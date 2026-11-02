"""Load trained models and compute case scores s(x) of manifest bags."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from anchor.data import make_loader
from anchor.models import GlobalOrdinalMIL, MultiReaderOrdinalMIL


@torch.no_grad()
def score_bags(mil: torch.nn.Module, manifest: pd.DataFrame, features_root: Path, device, batch_size: int = 16,
               num_workers: int = 4) -> np.ndarray:
    """s(x) of every bag of the manifest, in manifest order (every patch, no subsampling)."""
    mil.eval()
    out = []
    for bags, _, _ in make_loader(manifest, features_root, [], batch_size, train=False, num_workers=num_workers):
        s, _ = mil([b.to(device) for b in bags])
        out.append(s.float().cpu().numpy().ravel())
    return np.concatenate(out) if out else np.zeros(0)


def load_phase1(fold_dir: Path) -> MultiReaderOrdinalMIL:
    """A fold of `anchor.train_central` (model.pt + config.yaml)."""
    from anchor.train_central import build_model

    cfg = yaml.safe_load((fold_dir / "config.yaml").read_text())
    model = build_model(argparse.Namespace(**cfg))
    model.load_state_dict(torch.load(fold_dir / "model.pt", map_location="cpu"))
    return model.eval()


def phase1_deltas(fold_dir: Path, centres: list[str]) -> np.ndarray:
    """Sum-zero phase-1 offsets (centres, K - 1) in the given order."""
    model = load_phase1(fold_dir)
    d = model.effective_delta().detach().numpy()
    return np.stack([d[model.offset_sources.index(c)] for c in centres])


def load_global(path: Path, **model_args) -> GlobalOrdinalMIL:
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    model = GlobalOrdinalMIL(**model_args)
    model.load_state_dict({k: v for k, v in state.items() if k.startswith(("mil.", "cutpoints."))})
    return model.eval()
