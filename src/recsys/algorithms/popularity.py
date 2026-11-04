"""Most-popular baseline."""

from __future__ import annotations

import numpy as np
import torch

from recsys.algorithms.base import Recommender
from recsys.device import resolve_device


class Popularity(Recommender):
    """Popularity (TopPop baseline): ranks items by frequency in the known interactions; CTR: smoothed item rate.

    ``prior`` is the pseudo-count pulling item CTRs toward the global rate.
    """

    tasks = frozenset({"ranking", "ctr"})
    inductive = True
    space = {"prior": [1.0, 10.0, 100.0]}

    def __init__(self, prior: float = 10.0, device: str = "auto") -> None:
        self.prior = prior
        self.device = resolve_device(device)

    def fit(self, data) -> None:
        items = data.train["item"].to_numpy()
        n = data.n_items
        if data.task == "ranking":
            counts = np.bincount(items, minlength=n).astype(np.float32)
            self.scores = torch.as_tensor(counts, device=self.device)
        else:
            labels = data.train["label"].to_numpy().astype(np.float64)
            clicks = np.bincount(items, weights=labels, minlength=n)
            shows = np.bincount(items, minlength=n)
            self.ctr = (clicks + self.prior * labels.mean()) / (shows + self.prior)

    def score(self, users, history):
        # Count everything known at prediction time (train, plus val when scoring test).
        if getattr(self, "_counted", None) is not history:
            counts = np.bincount(history.items, minlength=history.n_items).astype(np.float32)
            self.scores, self._counted = torch.as_tensor(counts, device=self.device), history
        return self.scores.expand(len(users), -1).clone()

    def predict(self, rows):
        return self.ctr[rows["item"].to_numpy()]
