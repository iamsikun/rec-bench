"""Random baseline: uniform scores, the floor every method must beat."""

from __future__ import annotations

import torch

from recsys.algorithms.base import Recommender
from recsys.device import resolve_device


class Random(Recommender):
    """Random (sanity floor, no paper): every item gets an i.i.d. uniform score.

    Scores come from a generator seeded with ``data.seed``, so runs are
    reproducible; they ignore the history, so any user can be scored.
    """

    tasks = frozenset({"ranking"})
    inductive = True
    space = {}

    def __init__(self, device: str = "auto") -> None:
        self.device = resolve_device(device)

    def fit(self, data) -> None:
        self.n_items = data.n_items
        self.generator = torch.Generator(device=self.device).manual_seed(data.seed)

    def score(self, users, history):
        return torch.rand((len(users), self.n_items), device=self.device, generator=self.generator)
