"""ItemKNN (Sarwar et al., WWW 2001): item-item cosine neighbourhoods, Dacrema et al. (RecSys 2019) settings."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from recsys.algorithms.base import Recommender
from recsys.device import resolve_device


def history_rows(history, users, device: torch.device) -> torch.Tensor:
    """Dense binary ``(len(users), n_items)`` history rows on ``device``."""
    return torch.as_tensor(history.matrix(users).toarray(), device=device)


def gram(X: sp.spmatrix, device: torch.device, weights: np.ndarray | None = None) -> torch.Tensor:
    """Dense item-item co-occurrence ``X^T diag(weights) X`` on ``device`` (sparse product on the CPU)."""
    Xw = X if weights is None else sp.diags(weights.astype(np.float32)) @ X
    return torch.as_tensor((X.T @ Xw).toarray().astype(np.float32), device=device)


def keep_topk(S: torch.Tensor, k: int, dim: int = 0) -> torch.Tensor:
    """A new matrix keeping the ``k`` largest entries of ``S`` along ``dim`` (others zero)."""
    k = min(k, S.shape[dim])
    values, idx = S.topk(k, dim=dim)
    return torch.zeros_like(S).scatter_(dim, idx, values)


class ItemKNN(Recommender):
    """ItemKNN (Sarwar et al., WWW 2001): score items by their similarity to the items a user has seen.

    Similarity is cosine with shrinkage, ``s_ij = x_i . x_j / (|x_i| |x_j| +
    shrink)`` (a plain dot product when ``normalize=False``), the diagonal is
    zeroed and only the ``k`` most similar neighbours of each item are kept.
    Scores are ``x_u @ S``. Defaults follow Dacrema et al. (RecSys 2019).
    """

    tasks = frozenset({"ranking"})
    inductive = True
    space = {"k": [20, 50, 100, 200], "shrink": [0.0, 10.0, 100.0, 500.0]}

    def __init__(self, k: int = 50, shrink: float = 100.0, normalize: bool = True, device: str = "auto") -> None:
        self.k = k
        self.shrink = shrink
        self.normalize = normalize
        self.device = resolve_device(device)

    def fit(self, data) -> None:
        G = gram(data.history.matrix(), self.device)
        if self.normalize:
            norms = G.diagonal().sqrt()
            G = G / (norms[:, None] * norms[None, :] + self.shrink + 1e-6)
        G.diagonal().zero_()
        self.S = keep_topk(G, self.k, dim=0)  # column j holds item j's neighbours

    def score(self, users, history):
        return history_rows(history, users, self.device) @ self.S
