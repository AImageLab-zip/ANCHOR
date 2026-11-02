"""Post-extraction tissue filter for the PAR Hamamatsu and Grundium bags (the Leica bags are used as extracted).

Some Hamamatsu scans cover the whole glass: the black slide border and the grey or tinted glass pass the white filter
of `extract_features.py`. Each kept patch is re-read on the downsample-4 level and dropped when
  * >= 70 % of its pixels are dark (max RGB < 60), or
  * < 5 % of its pixels are tissue (HSV saturation > 0.15 and max RGB >= 60).
Features are per patch, so dropping rows equals filtering before the extraction. With this rule the Hamamatsu bags
hold 1.04x the Leica patch count of the same slides (1.40x before), Grundium loses ~1 % of its patches.

    python datasets/par/tissue_filter.py --slides-dir <PAR>/component_B_hamamatsu_ndpi --ext ndpi --patch-size 593 \\
        --src <features>/hamamatsu --dst <features>/hamamatsu_tf
"""
from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyvips

DS = 4
DARK, DARK_FRAC = 60, 0.70
SAT, MIN_TISSUE, BRIGHT = 0.15, 0.05, 60
_local = threading.local()


def ds_level(path: str) -> int:
    base = pyvips.Image.new_from_file(path)
    for lev in range(int(base.get("openslide.level-count"))):
        if abs(float(base.get(f"openslide.level[{lev}].downsample")) - DS) < 0.01:
            return lev
    raise ValueError(f"{path}: no downsample-{DS} level")


def measure(task) -> tuple[float, float]:
    """(dark fraction, tissue fraction) of one patch on the downsample-4 level."""
    path, lev, x, y, ps = task
    if getattr(_local, "key", None) != (path, lev):
        _local.slide, _local.key = pyvips.Image.new_from_file(path, level=lev), (path, lev)
    s = ps // DS
    rgb = np.ndarray(buffer=_local.slide.crop(x // DS, y // DS, s, s).write_to_memory(), dtype=np.uint8,
                     shape=[s, s, _local.slide.bands])[:, :, :3]
    a = rgb.astype(np.float32) / 255
    mx, mn = a.max(2), a.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    return float((mx < DARK / 255).mean()), float(((sat > SAT) & (mx >= BRIGHT / 255)).mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slides-dir", type=Path, required=True)
    p.add_argument("--ext", required=True, help="Slide file extension (ndpi, svs).")
    p.add_argument("--patch-size", type=int, required=True)
    p.add_argument("--src", type=Path, required=True, help="Folder of the extracted npz.")
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.threads) as pool:
        for npz in sorted(args.src.glob("*.npz")):
            out = args.dst / npz.name
            if out.exists():
                continue
            slide = str(args.slides_dir / f"{npz.stem}.{args.ext}")
            z = np.load(npz)
            lev = ds_level(slide)
            m = np.array(list(pool.map(measure, [(slide, lev, int(x), int(y), args.patch_size) for x, y in z["coords"]],
                                       chunksize=64)))
            black = m[:, 0] >= DARK_FRAC
            keep = ~black & (m[:, 1] >= MIN_TISSUE)
            assert keep.any(), f"{npz.stem}: the filter drops every patch"
            np.savez(out, **{k: z[k][keep] for k in z.files})
            print(f"{npz.stem}: {len(keep)} -> {int(keep.sum())} patches", flush=True)


if __name__ == "__main__":
    main()
