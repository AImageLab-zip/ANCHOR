"""Table 1: consensus and own-head QWK / MAE on carcinoma biopsies, mean of seeds, paired patient bootstrap.

    python -m anchor.evaluate --table configs/par/table1.yaml --slides <manifests>/slides.csv --runs runs/par \\
        --out results/par_table1

`--table` lists the rows; each row has `runs`, run directories relative to `--runs` with `{seed}` (and `{centre}`)
placeholders, and optionally
  file:     prediction file in each fold directory (default test_predictions.csv; test_predictions_refit.csv for the
            refitted ANCHOR-Fed / -Full);
  ensemble: true to average the class probabilities of the row's runs of the same seed (ANCHOR-Loc ensemble);
            otherwise every run counts as one model and the row is the mean over models and seeds (ANCHOR-Loc single:
            three centre models x three seeds).
A run's head for centre c is `prob_<c>_k`; a run without it (MedianLabels) uses its consensus head for every centre.
Marks (computed separately for QWK and MAE, on the consensus column): ^ / v better / worse than `reference`
(95 % bootstrap CI of the difference excludes 0), + worse than `best`.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from anchor.metrics import LOWER_IS_BETTER, PatientBootstrap, table_metrics

K = 6
REFS: pd.DataFrame | None = None
CENTRES: list[str] = []
MODELS: dict[str, list[dict[str, np.ndarray]]] = {}


def load_run(run: Path, file: str) -> dict[str, np.ndarray]:
    files = sorted(run.glob(f"fold_*/{file}"))
    if not files:
        raise FileNotFoundError(f"{run}: no fold_*/{file}")
    df = pd.concat([pd.read_csv(f) for f in files]).set_index("bag_id")
    pos = REFS.index.get_indexer(df.index)
    assert (pos >= 0).all() and len(set(pos)) == len(pos) == len(REFS), f"{run}: test biopsies != slide table"
    out = {}
    for head in ("consensus", *CENTRES):
        cols = [f"prob_{head}_{k}" for k in range(K)]
        if not set(cols) <= set(df.columns):
            continue
        a = np.full((len(REFS), K), np.nan)
        a[pos] = np.clip(df[cols].to_numpy(), 1e-7, 1)
        out["cons" if head == "consensus" else head] = a
    for c in CENTRES:
        out.setdefault(c, out["cons"])
    return out


def row_models(row: dict, runs_root: Path, seeds: list[int]) -> list[dict[str, np.ndarray]]:
    file = row.get("file", "test_predictions.csv")
    models = []
    for seed in seeds:
        paths = sorted({r.format(seed=seed, centre=c) for r in row["runs"] for c in CENTRES})
        prs = [load_run(runs_root / p, file) for p in paths]
        if row.get("ensemble"):
            models.append({h: np.mean([p[h] for p in prs], axis=0) for h in prs[0]})
        else:
            models += prs
    return models


def stats(idx: np.ndarray) -> dict[str, pd.DataFrame]:
    """Per row: mean over its models of cons / own QWK and MAE on the rows idx."""
    out = {}
    for name, models in MODELS.items():
        per = []
        for pr in models:
            q = table_metrics(pr, REFS, CENTRES, idx, "qwk", consensus="consensus")
            m = table_metrics(pr, REFS, CENTRES, idx, "mae", consensus="consensus")
            per.append({"cons QWK": q["cons"], "cons MAE": m["cons"], "own QWK": q["own"], "own MAE": m["own"]})
        out[name] = pd.DataFrame(per).mean()
    return out


def main() -> None:
    global REFS, CENTRES
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--slides", type=Path, required=True, help="slides.csv of datasets/par/build_study.py")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Output prefix (.csv tables).")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    spec = yaml.safe_load(args.table.read_text())
    CENTRES[:] = spec["centres"]
    REFS = pd.read_csv(args.slides).set_index("bag_id").astype({c: float for c in [*spec["centres"], "consensus"]})
    for row in spec["rows"]:
        MODELS[row["name"]] = row_models(row, args.runs, spec["seeds"])
    point = stats(np.arange(len(REFS)))
    boot = PatientBootstrap(REFS["patient_id"], args.n_boot, 0)
    with ProcessPoolExecutor(args.workers) as pool:
        samples = list(pool.map(stats, boot.samples, chunksize=10))
    table = pd.DataFrame(point).T
    pairs, marks = [], {name: {"cons QWK": "", "cons MAE": ""} for name in MODELS}
    for name in MODELS:
        for other, mark in ((spec.get("reference"), "ref"), (spec.get("best"), "best")):
            if other is None or other == name or other not in MODELS:
                continue
            for metric in ("cons QWK", "cons MAE", "own QWK", "own MAE"):
                d0 = point[name][metric] - point[other][metric]
                d = np.array([s[name][metric] - s[other][metric] for s in samples])
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                pairs.append({"A": name, "B": other, "metric": metric, "diff": d0, "lo": lo, "hi": hi})
                if metric.startswith("cons") and (lo > 0 or hi < 0):
                    better = (d0 < 0) == LOWER_IS_BETTER[metric.split()[1].lower()]
                    if mark == "ref":
                        marks[name][metric] += "^" if better else "v"
                    elif not better:
                        marks[name][metric] += "+"
    shown = table.round(3).astype(str)
    for name, m in marks.items():
        for metric, s in m.items():
            shown.loc[name, metric] += s
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(f"{args.out}_points.csv")
    pd.DataFrame(pairs).to_csv(f"{args.out}_pairs.csv", index=False)
    shown.to_csv(f"{args.out}.csv")
    pd.set_option("display.width", 200)
    print(shown.to_string())


if __name__ == "__main__":
    main()
