"""Test predictions of a federated (or single-centre ANCHOR-Loc) run, every head, with and without the anchor refit.

For each fold, the final global model (MIL + theta) scores the test biopsies of every centre (`--test-set`, a merged
set). Centre heads use theta + delta_c, with delta_c
  * the phase-1 offsets (`--phase1-dir`; ANCHOR, fixed during training),
  * the clients' own trained offsets (`--offsets trainable`, the last round's `<client>/results/delta.json`; FedRep-type), or
  * 0 (`--offsets none`; BaseFed, GLC),
and the consensus head uses theta + mean_c delta_c. With `--glc`, centre c's head is p(z | x) C_c (C_c from the
client set's transition.json). With `--anchor-set`, the anchors are scored too and (theta, delta, alpha) are refitted on
their grades from every centre (the final exchange of ANCHOR-Fed / -Full, `anchor.heads.refit_on_anchors`).

    python -m anchor.fl.export --run runs/par/anchor_full_s7 --centres r1 r2 vir --features-root <features>/par3c \\
        --test-set <manifests>/merged/crossed --phase1-dir runs/par/phase1_N34_s7 --anchor-set <manifests>/merged/anchors34_d0

Writes `<run>/<fold>/test_predictions.csv` (heads as trained) and, with `--anchor-set`, `test_predictions_refit.csv`,
columns prob_<head>_<k> for head = consensus and every centre (layout of `anchor.train_central`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from anchor.fl.job import GLOBAL_MODEL
from anchor.heads import head_probs, refit_on_anchors
from anchor.scoring import load_global, phase1_deltas, score_bags


def write(path: Path, ids, heads: dict[str, np.ndarray]) -> None:
    data = {"bag_id": list(ids)}
    for h, p in heads.items():
        for k in range(p.shape[1]):
            data[f"prob_{h}_{k}"] = p[:, k]
    pd.DataFrame(data).to_csv(path, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--centres", nargs="+", required=True)
    p.add_argument("--features-root", type=Path, required=True)
    p.add_argument("--test-set", type=Path, required=True)
    p.add_argument("--phase1-dir", type=Path, default=None)
    p.add_argument("--offsets", choices=["phase1", "trainable", "none"], default=None,
                   help="Default: phase1 with --phase1-dir, none otherwise.")
    p.add_argument("--glc", type=Path, default=None, help="Manifest root: <root>/<c>_client/<set>/<fold>/transition.json")
    p.add_argument("--glc-set", default=None)
    p.add_argument("--anchor-set", type=Path, default=None)
    p.add_argument("--l2", type=float, default=0.01)
    args = p.parse_args()
    mode = args.offsets or ("phase1" if args.phase1_dir else "none")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold_dir in sorted(d for d in args.run.iterdir() if d.name.startswith("fold_")):
        fold = fold_dir.name
        model = load_global(fold_dir / GLOBAL_MODEL).to(device)
        theta = model.cutpoints.thresholds().detach().cpu().numpy()
        if mode == "phase1":
            delta = phase1_deltas(args.phase1_dir / fold, args.centres)
        elif mode == "trainable":
            delta = np.array([json.loads((fold_dir / f"{c}_client" / "results" / "delta.json").read_text()) for c in args.centres])
        else:
            delta = np.zeros((len(args.centres), len(theta)))
        test = pd.read_csv(args.test_set / fold / "test_manifest.csv")
        s = score_bags(model.mil, test, args.features_root, device)
        heads = {"consensus": head_probs(theta, s, offset=delta.mean(0))}
        for i, c in enumerate(args.centres):
            if args.glc:
                C = np.array(json.loads((args.glc / f"{c}_client" / args.glc_set / fold / "transition.json")
                                        .read_text())["C"])
                heads[c] = heads["consensus"] @ C
            else:
                heads[c] = head_probs(theta, s, offset=delta[i])
        write(fold_dir / "test_predictions.csv", test["bag_id"], heads)
        if args.anchor_set is not None:
            anchors = pd.read_csv(args.anchor_set / fold / "train_manifest.csv")
            sa = score_bags(model.mil, anchors, args.features_root, device)
            grades = anchors[[f"label_{c}" for c in args.centres]].to_numpy().astype(int)
            head, d, alpha = refit_on_anchors(theta, sa, grades, l2=args.l2)
            th = head.thresholds().detach().numpy()
            refit = {"consensus": head_probs(th, s, alpha)}
            refit.update({c: head_probs(th, s, alpha, d[i]) for i, c in enumerate(args.centres)})
            write(fold_dir / "test_predictions_refit.csv", test["bag_id"], refit)
            (fold_dir / "refit.json").write_text(json.dumps({"theta": th.tolist(), "delta": d.tolist(), "alpha": alpha}))
        print(f"{fold}: {len(test)} test biopsies" + (" + refit" if args.anchor_set is not None else ""), flush=True)


if __name__ == "__main__":
    main()
