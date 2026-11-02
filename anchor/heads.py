"""Closed-form-free head fits on frozen case scores (maximum likelihood, L-BFGS).

* `fit_ai_offsets`: after phase 1, the AI reader's offset profile omega (K - 1 values, common to every centre) on the
  anchors, with the phase-1 score and theta frozen:  P(y_AI <= k - 1 | x) = sigmoid(theta_k + omega_k - s(x)).
* `refit_on_anchors`: the final exchange of ANCHOR-Fed / -Full. With the network frozen, refit theta, the sum-zero
  centre offsets delta and a score scale alpha on the anchors' grades from every centre:
      min  sum_a sum_c NLL(y_{a,c} | alpha s(x_a), theta + delta_c) / n  +  lambda ||delta||^2.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from anchor.models import CumulativeLinkHead


def head_with(theta: np.ndarray | torch.Tensor, dtype=torch.float32) -> CumulativeLinkHead:
    head = CumulativeLinkHead(len(theta) + 1).to(dtype)
    head.set_thresholds(torch.as_tensor(theta, dtype=torch.float64))
    return head


def ordinal_nll(theta: torch.Tensor, offsets: torch.Tensor, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean cumulative-link NLL; theta (K - 1,), offsets (K - 1,) or (n, K - 1), s (n,), y (n,)."""
    cdf = torch.sigmoid(theta + offsets - s.unsqueeze(-1))
    ones = torch.ones(len(y), 1, dtype=cdf.dtype)
    cdf = torch.cat([torch.zeros_like(ones), cdf, ones], dim=1)
    p = torch.clamp(cdf[:, 1:] - cdf[:, :-1], min=1e-7)
    return -torch.log(p[torch.arange(len(y)), y]).mean()


def fit_ai_offsets(theta: np.ndarray, scores: np.ndarray, ai_grades: np.ndarray) -> np.ndarray:
    """omega (K - 1,) maximising the likelihood of the AI grades of the anchors (float64)."""
    theta = torch.as_tensor(theta, dtype=torch.float64)
    s = torch.as_tensor(scores, dtype=torch.float64)
    y = torch.as_tensor(ai_grades).long()
    omega = torch.zeros(len(theta), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([omega], lr=0.5, max_iter=1000, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = ordinal_nll(theta, omega, s, y)
        loss.backward()
        return loss

    opt.step(closure)
    return omega.detach().numpy()


def refit_on_anchors(theta0: np.ndarray, scores: np.ndarray, grades: np.ndarray, l2: float = 0.01,
                     max_iter: int = 500) -> tuple[CumulativeLinkHead, np.ndarray, float]:
    """Refit (theta, delta, alpha) on anchors with scores (n,) and grades (n, C) from C centres (-1 = missing).

    Starts from theta0 (the trained shared cut-points), delta = 0 and alpha = 1. Returns the fitted head (theta),
    the centred offsets delta (C, K - 1) and alpha.
    """
    head = head_with(theta0)
    s = torch.as_tensor(scores, dtype=torch.float32)
    grades = np.asarray(grades)
    per_centre = [(s[grades[:, c] >= 0], torch.as_tensor(grades[grades[:, c] >= 0, c]).long())
                  for c in range(grades.shape[1])]
    n = sum(len(y) for _, y in per_centre)
    delta = torch.zeros(grades.shape[1], len(theta0), requires_grad=True)
    log_scale = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS(list(head.parameters()) + [delta, log_scale], lr=0.5, max_iter=max_iter,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        d = delta - delta.mean(dim=0, keepdim=True)
        a = torch.exp(log_scale)
        loss = sum(F.nll_loss(head(a * sc, d[c].expand(len(sc), -1)), y, reduction="sum")
                   for c, (sc, y) in enumerate(per_centre)) / n + l2 * (d ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return head, (delta - delta.mean(dim=0, keepdim=True)).detach().numpy(), float(torch.exp(log_scale).item())


def head_probs(theta: np.ndarray, scores: np.ndarray, alpha: float = 1.0, offset: np.ndarray | None = None) -> np.ndarray:
    """Class probabilities (n, K) of the head theta + offset on scores alpha * s (numpy, as the evaluation)."""
    cut = np.asarray(theta)[None, :] + (0 if offset is None else np.asarray(offset)[None, :])
    ge = 1 / (1 + np.exp(-(alpha * np.asarray(scores)[:, None] - cut)))
    n = len(scores)
    return np.clip(-np.diff(np.concatenate([np.ones((n, 1)), ge, np.zeros((n, 1))], axis=1), axis=1), 0, 1)
