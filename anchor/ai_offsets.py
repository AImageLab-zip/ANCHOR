"""Fit the AI reader's offsets omega on the anchors at the end of phase 1 (one common profile for all centres).

For each fold, the phase-1 model scores the anchors; with s(x) and theta frozen, omega maximises the likelihood of
the AI grades of the anchors (`label_ai`, available on every centre's own anchors). The offsets stay fixed in
phase 2 (part of the phase-1 package).

    python -m anchor.ai_offsets --phase1-dir runs/par/phase1_N34_s7 --anchor-set <manifests>/merged/anchors34_d0 \\
        --features-root <features>/par3c --out runs/par/ai_offsets_N34_s7

Output: `<out>/<fold>/ai_offsets.json` (list of K - 1 values).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from anchor.data import fold_names
from anchor.heads import fit_ai_offsets
from anchor.scoring import load_phase1, score_bags


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--phase1-dir", type=Path, required=True)
    p.add_argument("--anchor-set", type=Path, required=True)
    p.add_argument("--features-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold in fold_names(args.phase1_dir):
        model = load_phase1(args.phase1_dir / fold).to(device)
        anchors = pd.read_csv(args.anchor_set / fold / "train_manifest.csv")
        anchors = anchors[anchors["label_ai"] >= 0].reset_index(drop=True)
        s = score_bags(model.mil, anchors, args.features_root, device)
        theta = model.cutpoints.thresholds().detach().cpu().double().numpy()
        omega = fit_ai_offsets(theta, s.astype(np.float64), anchors["label_ai"].to_numpy())
        (args.out / fold).mkdir(parents=True, exist_ok=True)
        (args.out / fold / "ai_offsets.json").write_text(json.dumps(omega.tolist()))
        print(f"{fold}: {len(anchors)} anchors, omega {np.round(omega, 3).tolist()}")


if __name__ == "__main__":
    main()
