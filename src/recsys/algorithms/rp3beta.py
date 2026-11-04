"""RP3beta (Paudel et al., TiiS 2016): popularity-penalized three-step random walk, Dacrema et al. (RecSys 2019)."""

from __future__ import annotations

import numpy as np

from recsys.algorithms.base import Recommender
from recsys.algorithms.itemknn import gram, history_rows, keep_topk
from recsys.device import resolve_device


class RP3beta(Recommender):
    """RP3beta (Paudel et al., TiiS 2016): item-to-item random-walk probabilities divided by popularity^beta.

    With binary interactions, ``W_ij = p(i -> u)^alpha p(u -> j)^alpha`` summed
    over users, i.e. ``deg_i^-alpha (X^T deg_u^-alpha X)_ij``, then divided by
    ``deg_j^beta``. The diagonal is zeroed and each item keeps its ``k``
    strongest out-edges; scores are ``x_u @ W``. Defaults follow Dacrema et
    al. (RecSys 2019).
    """

    tasks = frozenset({"ranking"})
    inductive = True
    space = {"alpha": [0.5, 0.8, 1.0], "beta": [0.2, 0.4, 0.6, 0.8], "k": [50, 100, 200]}

    def __init__(self, alpha: float = 1.0, beta: float = 0.6, k: int = 100, device: str = "auto") -> None:
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.device = resolve_device(device)

    def fit(self, data) -> None:
        X = data.history.matrix()
        user_deg = np.asarray(X.sum(1)).ravel()
        user_w = np.where(user_deg > 0, np.maximum(user_deg, 1) ** -self.alpha, 0.0)
        W = gram(X, self.device, user_w)
        item_deg = W.new_tensor(np.asarray(X.sum(0)).ravel()).clamp_min(1)
        W = W * item_deg[:, None] ** -self.alpha * item_deg[None, :] ** -self.beta
        W.diagonal().zero_()
        self.W = keep_topk(W, self.k, dim=1)  # row i holds item i's out-edges

    def score(self, users, history):
        return history_rows(history, users, self.device) @ self.W
