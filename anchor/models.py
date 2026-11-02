"""Gated-attention MIL with a cumulative-link (ordinal) head and reader-specific offsets.

A MIL network maps a bag of patch embeddings x to one scalar case score s(x). The cumulative-link head holds K - 1
ordered cut-points theta and models, for a head with offset vector d,

    P(y >= k | x) = sigmoid(s(x) - theta_k - d_k),        k = 1 .. K - 1.

d = 0 is the consensus head; d = delta_c is the head of reader (centre) c. Equivalently, as implemented below,
P(y <= k - 1 | x) = sigmoid(theta_k + d_k - s(x)).

State-dict keys (`mil.*`, `cutpoints.*`, `delta`, `fixed_delta`, `delta_raw`) are those of the original experiment
code, so its checkpoints load unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PROB_EPS = 1e-7


class GatedAttentionMIL(nn.Module):
    """Gated attention pooling (Ilse et al., 2018) followed by a small MLP that outputs one score per bag."""

    def __init__(self, input_dim: int = 1536, attention_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.5,
                 output_dim: int = 1):
        super().__init__()
        self.attention_v = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Sigmoid())
        self.attention_score = nn.Linear(attention_dim, 1)
        self.classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, output_dim))

    def forward(self, bags: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Scores (B,) for output_dim = 1, logits (B, output_dim) otherwise; attention weights per bag."""
        scores, weights = [], []
        for bag in bags:
            a = torch.softmax(self.attention_score(self.attention_v(bag) * self.attention_u(bag)).squeeze(-1), dim=0)
            out = self.classifier((a.unsqueeze(-1) * bag).sum(dim=0))
            scores.append(out.squeeze(-1) if out.shape[-1] == 1 else out)
            weights.append(a)
        return torch.stack(scores), weights


def cumulative_logits(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Logits of the cumulative class proportions P(y <= k), k = 0 .. K - 2 (add-one smoothing)."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(float) + 1.0
    cum = np.cumsum(counts)[:-1] / counts.sum()
    return np.log(cum / (1 - cum))


class CumulativeLinkHead(nn.Module):
    """K - 1 ordered cut-points: theta_1 free, theta_{k+1} = theta_k + softplus(gap_k)."""

    def __init__(self, num_classes: int):
        super().__init__()
        if num_classes < 3:
            raise ValueError("The ordinal head needs at least 3 classes.")
        self.num_classes = num_classes
        self.theta_first = nn.Parameter(torch.zeros(1))
        self.theta_gaps = nn.Parameter(torch.zeros(num_classes - 2))

    def thresholds(self) -> torch.Tensor:
        return torch.cat([self.theta_first, self.theta_first + torch.cumsum(F.softplus(self.theta_gaps), dim=0)])

    @torch.no_grad()
    def set_thresholds(self, theta: torch.Tensor | np.ndarray, min_gap: float = 1e-3) -> None:
        theta = torch.as_tensor(theta, dtype=self.theta_first.dtype)
        gaps = torch.clamp(theta.diff(), min=min_gap)
        self.theta_first.fill_(float(theta[0]))
        self.theta_gaps.copy_(torch.log(torch.expm1(gaps)))

    @torch.no_grad()
    def init_from_labels(self, labels: np.ndarray) -> None:
        """Cut-points at the logits of the cumulative grade proportions: with s(x) = 0 the head predicts the training
        grade distribution, so the rare high grades do not start collapsed onto benign."""
        self.set_thresholds(cumulative_logits(labels, self.num_classes))

    def forward(self, scores: torch.Tensor, offsets: torch.Tensor | None = None) -> torch.Tensor:
        """Class log-probabilities (B, K) for scores (B,) and optional offsets (B, K - 1) or (K - 1,)."""
        theta = self.thresholds().unsqueeze(0)
        if offsets is not None:
            theta = theta + offsets
        cdf = torch.sigmoid(theta - scores.unsqueeze(-1))
        ones = torch.ones_like(cdf[:, :1])
        cdf = torch.cat([torch.zeros_like(ones), cdf, ones], dim=1)
        return torch.log(torch.clamp(cdf[:, 1:] - cdf[:, :-1], min=PROB_EPS))


def median_grade(probs: np.ndarray) -> np.ndarray:
    """Predicted grade = median of the predictive distribution = number of k with P(y >= k) >= 1/2."""
    prob_ge = 1.0 - np.cumsum(probs, axis=-1)[..., :-1]
    return (prob_ge >= 0.5).sum(axis=-1)


def expected_grade(probs: np.ndarray) -> np.ndarray:
    return probs @ np.arange(probs.shape[-1])


class MultiReaderOrdinalMIL(nn.Module):
    """One case score, one head per label source ("reader").

    Sources fall in three groups:
      * offset sources (the centres): head theta + delta_s, delta learned; with `sum_zero=True` the applied offsets are
        centred over these sources at every threshold, so theta is the consensus (mean) convention;
      * fixed sources (e.g. the AI reader in phase 2): constant offsets given at construction, outside the centring;
      * consensus sources (e.g. the median of the reads): scored by theta itself (offset 0);
      * free sources (e.g. the AI reader in the Crossed + AI reference): learned offsets kept outside the centring.
    forward returns log-probabilities (B, 1 + S, K): head 0 = consensus (theta), head 1 + s = source s.
    """

    def __init__(self, mil: nn.Module, num_classes: int, sources: list[str], consensus_sources: list[str] = (),
                 sum_zero: bool = True, learn_offsets: bool = True,
                 fixed_offsets: dict[str, list[float]] | None = None, free_sources: list[str] = ()):
        super().__init__()
        self.mil = mil
        self.cutpoints = CumulativeLinkHead(num_classes)
        self.sources = list(sources)
        fixed_offsets = dict(fixed_offsets or {})
        self.fixed_sources = [s for s in self.sources if s in fixed_offsets]
        self.offset_sources = [s for s in self.sources if s not in set(consensus_sources) | set(self.fixed_sources)]
        self.free_sources = [s for s in self.offset_sources if s in set(free_sources)]
        self.sum_zero = sum_zero
        self.delta = nn.Parameter(torch.zeros(len(self.offset_sources), num_classes - 1), requires_grad=learn_offsets)
        self.register_buffer("centred_rows", torch.tensor([s not in self.free_sources for s in self.offset_sources],
                                                          dtype=torch.bool), persistent=False)
        if self.fixed_sources:
            self.register_buffer("fixed_delta", torch.tensor([fixed_offsets[s] for s in self.fixed_sources],
                                                             dtype=torch.float32))

    def effective_delta(self) -> torch.Tensor:
        """(offset sources, K - 1): the offsets actually applied."""
        if not self.sum_zero:
            return self.delta
        mean = self.delta[self.centred_rows].mean(dim=0, keepdim=True)
        return torch.where(self.centred_rows.unsqueeze(-1), self.delta - mean, self.delta)

    def source_offsets(self) -> torch.Tensor:
        delta = self.effective_delta()
        zeros = delta.new_zeros(delta.shape[1])
        return torch.stack([delta[self.offset_sources.index(s)] if s in self.offset_sources
                            else self.fixed_delta[self.fixed_sources.index(s)] if s in self.fixed_sources else zeros
                            for s in self.sources])

    def forward(self, bags: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        scores, attention = self.mil(bags)
        offsets = self.source_offsets()
        heads = [self.cutpoints(scores)] + [self.cutpoints(scores, offsets[s].expand(len(scores), -1))
                                             for s in range(len(self.sources))]
        return torch.stack(heads, dim=1), attention


class CentreOrdinalMIL(nn.Module):
    """The model held by one centre in federated training: shared MIL + theta, private offset delta (never sent).

    The applied offset is `offset_scale * delta_raw`; with Adam this gives delta an effective learning rate of
    `offset_scale * lr` inside a single optimiser. forward returns the log-probabilities of the centre's head.
    """

    def __init__(self, mil: nn.Module, num_classes: int, offset_scale: float = 1.0):
        super().__init__()
        self.mil = mil
        self.cutpoints = CumulativeLinkHead(num_classes)
        self.offset_scale = offset_scale
        self.delta_raw = nn.Parameter(torch.zeros(num_classes - 1))

    def delta(self) -> torch.Tensor:
        return self.offset_scale * self.delta_raw

    def forward(self, bags: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        scores, attention = self.mil(bags)
        return self.cutpoints(scores, self.delta().expand(len(scores), -1)), attention


class GlobalOrdinalMIL(nn.Module):
    """Server-side global model (MIL + theta, no offsets), built from plain arguments for NVFlare."""

    def __init__(self, num_classes: int = 6, input_dim: int = 1536, attention_dim: int = 128, hidden_dim: int = 64,
                 dropout: float = 0.5):
        super().__init__()
        self.mil = GatedAttentionMIL(input_dim, attention_dim, hidden_dim, dropout)
        self.cutpoints = CumulativeLinkHead(num_classes)

    def forward(self, bags: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        scores, attention = self.mil(bags)
        return self.cutpoints(scores), attention
