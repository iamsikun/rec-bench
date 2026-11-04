"""EASE (Steck, WWW 2019): closed-form item-item linear autoencoder."""

from __future__ import annotations

import torch

from recsys.algorithms.base import Recommender
from recsys.device import resolve_device


class EASE(Recommender):
    """EASE (Steck, WWW 2019): closed-form item-item ridge regression with a zero diagonal.

    ``B = I - P / diag(P)`` with ``P = (X^T X + reg * I)^-1``, computed on the
    GPU when available. Memory is ``O(n_items^2)``; suited to catalogs up to a
    few tens of thousands of items.
    """

    tasks = frozenset({"ranking"})
    inductive = True
    space = {"reg": [100.0, 250.0, 500.0, 1000.0, 3000.0]}

    def __init__(self, reg: float = 500.0, device: str = "auto") -> None:
        self.reg = reg
        self.device = resolve_device(device)

    def fit(self, data) -> None:
        X = data.history.matrix()
        gram = torch.as_tensor((X.T @ X).toarray(), device=self.device)
        gram.diagonal().add_(self.reg)
        P = torch.linalg.inv(gram)
        self.B = P / -P.diagonal()
        self.B.diagonal().zero_()

    def score(self, users, history):
        return torch.as_tensor(history.matrix(users).toarray(), device=self.device) @ self.B
