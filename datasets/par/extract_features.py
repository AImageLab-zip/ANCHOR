"""Tile whole-slide images and embed the tissue patches with UNI2-h, in one pass (no patch images on disk).

Grid: non-overlapping full patches of `--patch-size` pixels at level 0, from (0, 0). A patch is background, and
dropped, when >= 70 % of its pixels have every band (alpha included, as read by libvips) above `--white-threshold`.
Kept patches go through PIL -> Resize(224) -> ToTensor -> ImageNet normalisation -> UNI2-h (1536-d CLS embedding).
Output: `<output-dir>/<slide stem>.npz` with `features` (n, 1536), `coords` (n, 2, top-left at level 0), `wsi_names`.

Patch size per scanner = round(512 x 0.262613 / mpp), i.e. 512 px at the resolution of the Leica GT450 (0.2626 um/px),
so every patch covers the same tissue area: PAR Leica 512, Hamamatsu 593, Grundium 537. White threshold 220
(Leica, Grundium) and 200 for the grey glass of the Hamamatsu scans (calibrated to match the Leica patch counts on the
same slides).

    python datasets/par/extract_features.py --slides-dir <PAR>/component_C_leica_svs --pattern '*.svs' \\
        --patch-size 512 --white-threshold 220 --weights <uni2-h>/pytorch_model.bin --output-dir <features>/leica

UNI2-h weights: https://huggingface.co/MahmoodLab/UNI2-h (gated; `pytorch_model.bin`).
"""
from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyvips
import timm
import torch
from PIL import Image
from torchvision import transforms

BACKGROUND_FRACTION = 0.70
_local = threading.local()


def load_uni2h(weights: Path, device: torch.device):
    model = timm.create_model(
        "vit_giant_patch14_224", pretrained=False, img_size=224, patch_size=14, depth=24, num_heads=24,
        init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    return model.to(device).eval(), transform


def _open(path: str) -> None:
    _local.slide = pyvips.Image.new_from_file(path)


def _patch(task):
    x, y, size, white, transform = task
    crop = _local.slide.crop(x, y, size, size)
    a = np.ndarray(buffer=crop.write_to_memory(), dtype=np.uint8, shape=[size, size, crop.bands])
    if np.all(a > white, axis=2).sum() / (size * size) >= BACKGROUND_FRACTION:
        return None
    rgb = a[:, :, :3] if a.shape[2] == 4 else a
    return transform(Image.fromarray(rgb)), (x, y)


@torch.no_grad()
def process(slide: Path, out: Path, model, transform, device, size: int, white: int, batch: int, workers: int) -> int:
    probe = pyvips.Image.new_from_file(str(slide))
    if probe.get("vips-loader") != "openslideload":
        raise RuntimeError(f"{slide}: expected the openslide loader, got {probe.get('vips-loader')}")
    grid = [(x, y) for y in range(0, probe.height - size + 1, size) for x in range(0, probe.width - size + 1, size)]
    feats, coords, pending = [], [], []

    def flush():
        if pending:
            feats.append(model(torch.stack([p[0] for p in pending]).to(device)).cpu().numpy())
            coords.extend(p[1] for p in pending)
            pending.clear()

    with ThreadPoolExecutor(max_workers=workers, initializer=_open, initargs=(str(slide),)) as pool:
        for start in range(0, len(grid), 4096):
            for r in pool.map(_patch, [(x, y, size, white, transform) for x, y in grid[start:start + 4096]]):
                if r is not None:
                    pending.append(r)
                    if len(pending) >= batch:
                        flush()
    flush()
    if not feats:
        print(f"{slide.stem}: no tissue patch")
        return 0
    features = np.concatenate(feats)
    np.savez(out, features=features, coords=np.array(coords, dtype=np.int64),
             wsi_names=np.array([slide.stem] * len(features)))
    return len(features)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slides-dir", type=Path, required=True)
    p.add_argument("--pattern", default="*.svs")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--white-threshold", type=int, default=220)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=7)
    p.add_argument("--only", nargs="*", help="Slide stems to process (default: all).")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, transform = load_uni2h(args.weights, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for slide in sorted(args.slides_dir.glob(args.pattern)):
        out = args.output_dir / f"{slide.stem}.npz"
        if (args.only and slide.stem not in args.only) or out.exists():
            continue
        n = process(slide, out, model, transform, device, args.patch_size, args.white_threshold, args.batch_size,
                    args.num_workers)
        print(f"{slide.stem}: {n} patches", flush=True)


if __name__ == "__main__":
    main()
