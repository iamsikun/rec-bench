"""Small fixtures shared by tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from recsys import Recommender
from recsys.benchmarks import Benchmark
from recsys.data import Interactions, Synthetic
from recsys.protocol import LeaveLastOut


@dataclass(frozen=True)
class FrameDataset(Synthetic):
    """A dataset served from an in-memory frame (keyed by ``tag`` for hashing)."""

    tag: str = "frame"
    frames: dict = field(default_factory=dict, hash=False, compare=False, repr=False)

    def read(self, root):
        return Interactions(self.frames["events"], self.frames.get("users"), self.frames.get("items"))


def ranking_benchmark(dataset=None, metrics=("hr@10", "ndcg@10"), split=None, **kwargs) -> Benchmark:
    return Benchmark("test-ranking", dataset or Synthetic(), "ranking", split or LeaveLastOut(), metrics, **kwargs)


class Constant(Recommender):
    tasks = frozenset({"ranking"})
    inductive = True

    def fit(self, data):
        self.n = data.n_items

    def score(self, users, history):
        return torch.zeros(len(users), history.n_items)


class Spy(Recommender):
    """Records every user/item reachable from TrainData and every scored batch."""

    tasks = frozenset({"ranking"})
    inductive = True

    def fit(self, data):
        self.seen = set(zip(data.train["user"].to_list(), data.train["item"].to_list()))
        h = data.history
        for u in range(h.n_users):
            self.seen |= {(u, int(i)) for i in h.items[h.indptr[u] : h.indptr[u + 1]]}
        self.batches = []

    def score(self, users, history):
        self.batches.append(len(users))
        return torch.as_tensor(np.random.default_rng(0).random((len(users), history.n_items)))
