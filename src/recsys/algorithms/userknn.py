"""UserKNN (Resnick et al., CSCW 1994): user-user neighbourhoods, Dacrema et al. (RecSys 2019) settings."""

from __future__ import annotations

import torch

from recsys.algorithms.base import Recommender
from recsys.algorithms.itemknn import history_rows
from recsys.device import resolve_device


class UserKNN(Recommender):
    """UserKNN (Resnick et al., CSCW 1994): the k most similar training users vote for their items.

    At prediction time the user's history row is compared with every training
    user by cosine similarity with shrinkage, ``x_u . x_v / (|x_u| |x_v| +
    shrink)`` (dot product when ``normalize=False``); the user's own training
    row is excluded. Scores are the similarity-weighted sum of the ``k``
    nearest rows. Because similarity is computed from the history, unseen users
    are scored too. Keeps the training matrix dense on the device
    (``O(n_users * n_items)`` memory). Defaults follow Dacrema et al.
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
        self.X = torch.as_tensor(data.history.matrix().toarray(), device=self.device)
        self.norms = self.X.sum(1).sqrt()  # binary rows: |x|^2 = row sum

    def score(self, users, history):
        rows = history_rows(history, users, self.device)
        sim = rows @ self.X.T
        if self.normalize:
            sim /= rows.sum(1, keepdim=True).sqrt() * self.norms[None, :] + self.shrink + 1e-6
        sim[torch.arange(len(users), device=self.device), torch.as_tensor(users, device=self.device)] = 0.0
        values, idx = sim.topk(min(self.k, sim.shape[1]), dim=1)
        return torch.zeros_like(sim).scatter_(1, idx, values) @ self.X
