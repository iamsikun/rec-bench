"""Metric functions.

Ranking metrics take ``hits`` — a ``(users, K)`` float tensor marking whether
the item at each rank is relevant — and ``n_rel``, the number of relevant
items per user, and return one value per user.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import rankdata


def _discounts(k: int, device: torch.device) -> torch.Tensor:
    return 1.0 / torch.log2(torch.arange(2, k + 2, device=device, dtype=torch.float32))


def hr(hits: torch.Tensor, n_rel: torch.Tensor, k: int) -> torch.Tensor:
    return hits[:, :k].amax(1)


def recall(hits: torch.Tensor, n_rel: torch.Tensor, k: int) -> torch.Tensor:
    """Hits / number of relevant items (LightGCN, RecBole convention)."""
    return hits[:, :k].sum(1) / n_rel


def nrecall(hits: torch.Tensor, n_rel: torch.Tensor, k: int) -> torch.Tensor:
    """Hits / min(k, number of relevant items): Liang et al. (2018) normalized recall."""
    return hits[:, :k].sum(1) / n_rel.clamp(max=k)


def precision(hits: torch.Tensor, n_rel: torch.Tensor, k: int) -> torch.Tensor:
    return hits[:, :k].sum(1) / k


def ndcg(hits: torch.Tensor, n_rel: torch.Tensor, k: int) -> torch.Tensor:
    disc = _discounts(k, hits.device)
    dcg = (hits[:, :k] * disc).sum(1)
    idcg = disc.cumsum(0)[(n_rel.clamp(max=k) - 1).long()]
    return dcg / idcg


def mrr(hits: torch.Tensor, n_rel: torch.Tensor, k: int) -> torch.Tensor:
    ranks = torch.arange(1, k + 1, device=hits.device, dtype=torch.float32)
    return (hits[:, :k] / ranks).amax(1)


RANKING = {"hr": hr, "recall": recall, "nrecall": nrecall, "precision": precision, "ndcg": ndcg, "mrr": mrr}


def parse(name: str) -> tuple[str, int]:
    """Split ``"ndcg@10"`` into ``("ndcg", 10)``."""
    base, _, k = name.partition("@")
    if base not in RANKING or not k.isdigit():
        raise ValueError(f"Unknown ranking metric {name!r}; expected <{'|'.join(RANKING)}>@<k>")
    return base, int(k)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties get average ranks)."""
    pos = y > 0.5
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


CTR = {"auc": auc, "logloss": logloss}
