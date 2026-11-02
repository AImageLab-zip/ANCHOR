"""Metrics of the paper's Table 1 and the paired patient bootstrap.

All metrics are computed on carcinoma biopsies (grade >= 1) with the median-rule prediction:
  * consensus: the consensus head against the consensus grade, on biopsies whose consensus is >= 1;
  * own: the head of centre c against c's own grade, on the biopsies of c's own site that c graded >= 1,
    averaged over the centres.
QWK uses the fixed label set 0 .. K - 1; MAE is in grade groups (lower is better).
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd

K = 6


def qwk(y: np.ndarray, p: np.ndarray, num_classes: int = K) -> float:
    """Quadratic weighted kappa on integer grades 0 .. K - 1 (nan for fewer than two cases)."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=int)
    if len(y) < 2:
        return np.nan
    observed = np.bincount(y * num_classes + p, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    observed = observed.astype(float)
    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    weights = np.subtract.outer(np.arange(num_classes), np.arange(num_classes)) ** 2 / (num_classes - 1) ** 2
    denom = (weights * expected).sum()
    return 1 - (weights * observed).sum() / denom if denom > 0 else np.nan


def mae(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 2:
        return np.nan
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


METRICS: dict[str, Callable] = {"qwk": qwk, "mae": mae}
LOWER_IS_BETTER = {"qwk": False, "mae": True}


def median_rule(probs: np.ndarray) -> np.ndarray:
    """(n, K) class probabilities -> predicted grade (median of the distribution); rows of nan -> -1."""
    ge = np.cumsum(probs[:, ::-1], axis=1)[:, ::-1][:, 1:]
    pred = (ge >= 0.5).sum(axis=1)
    return np.where(np.isnan(probs).any(axis=1), -1, pred)


def table_metrics(heads: Mapping[str, np.ndarray], refs: pd.DataFrame, centres: list[str], idx: np.ndarray,
                  metric: str = "qwk", consensus: str = "consensus") -> dict[str, float]:
    """Consensus and own-head metric of one run on the rows `idx` of `refs` (bootstrap sample or all rows).

    heads: {"cons": (n, K), <centre>: (n, K)} class probabilities aligned with refs; refs: one row per test biopsy with
    columns `site`, `<centre>` (that centre's grade, nan if missing) and `consensus`.
    """
    f = METRICS[metric]
    pred = {h: median_rule(p) for h, p in heads.items()}
    y = refs[consensus].to_numpy(float)[idx]
    p = pred["cons"][idx]
    m = ~np.isnan(y) & (y >= 1) & (p >= 0)
    out = {"cons": f(y[m].astype(int), p[m])}
    site = refs["site"].to_numpy()[idx]
    own = []
    for c in centres:
        if c not in pred:
            continue
        y, p = refs[c].to_numpy(float)[idx], pred[c][idx]
        m = (site == c) & ~np.isnan(y) & (y >= 1) & (p >= 0)
        own.append(f(y[m].astype(int), p[m]))
    out["own"] = float(np.mean(own)) if own else np.nan
    return out


class PatientBootstrap:
    """Resample patients with replacement; the same draws are used for every run, so differences are paired."""

    def __init__(self, patients: pd.Series, n_boot: int = 1000, seed: int = 0):
        codes, uniques = pd.factorize(patients)
        members = [np.flatnonzero(codes == i) for i in range(len(uniques))]
        rng = np.random.default_rng(seed)
        self.samples = [np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
                        for _ in range(n_boot)]
