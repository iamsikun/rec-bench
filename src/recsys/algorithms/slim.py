"""SLIM (Ning & Karypis, ICDM 2011): sparse non-negative item-item regression with an ElasticNet penalty."""

from __future__ import annotations

import logging

import torch

from recsys.algorithms.base import Recommender
from recsys.algorithms.itemknn import gram, history_rows, keep_topk
from recsys.device import resolve_device

LOGGER = logging.getLogger(__name__)


class SLIM(Recommender):
    """SLIM (Ning & Karypis, ICDM 2011): reconstruct each item column from the other items with a sparse non-negative W.

    For every item ``j``: ``min_w 1/(2 n_users) |x_j - X w|^2 + alpha * l1_ratio
    |w|_1 + alpha (1 - l1_ratio) / 2 |w|^2`` with ``w >= 0`` and ``w_j = 0``
    (scikit-learn's ``ElasticNet`` scaling, as in Dacrema et al., RecSys 2019).
    All columns are solved at once on the device by FISTA (accelerated
    projected proximal gradient; Beck & Teboulle 2009) on the Gram matrix
    ``G = X^T X``: each step is one ``n_items^2`` matmul, a soft-threshold and a
    clamp at zero. Each column then keeps its ``k`` largest weights; scores
    are ``x_u @ W``. Memory is ``O(n_items^2)``.
    """

    tasks = frozenset({"ranking"})
    inductive = True
    space = {"alpha": [1e-4, 1e-3, 1e-2], "l1_ratio": [0.01, 0.1, 0.5]}

    def __init__(self, alpha: float = 1e-3, l1_ratio: float = 0.1, k: int = 100, max_iter: int = 500,
                 tol: float = 1e-5, device: str = "auto") -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.device = resolve_device(device)

    def fit(self, data) -> None:
        G = gram(data.history.matrix(), self.device)
        l1 = self.alpha * self.l1_ratio * data.n_users
        l2 = self.alpha * (1 - self.l1_ratio) * data.n_users
        v = torch.ones(G.shape[0], device=self.device)
        for _ in range(30):  # power iteration: Lipschitz constant of the smooth part
            v = G @ v
            v /= v.norm()
        step = 1.0 / (float(v @ G @ v) * 1.01 + l2)
        diagonal = torch.eye(G.shape[0], dtype=torch.bool, device=self.device)
        W = torch.zeros_like(G)
        Y, t = W, 1.0
        for it in range(1, self.max_iter + 1):
            grad = G @ Y - G + l2 * Y
            W_next = (Y - step * (grad + l1)).clamp_min_(0).masked_fill_(diagonal, 0.0)
            t_next = (1 + (1 + 4 * t * t) ** 0.5) / 2
            Y = W_next + ((t - 1) / t_next) * (W_next - W)
            if it % 25 == 0:
                change = (W_next - W).abs().max().item()
                LOGGER.info("SLIM iteration %d: max coefficient change %.2e", it, change)
                if change < self.tol:
                    W = W_next
                    break
            W, t = W_next, t_next
        self.W = keep_topk(W, self.k, dim=0)

    def score(self, users, history):
        return history_rows(history, users, self.device) @ self.W
