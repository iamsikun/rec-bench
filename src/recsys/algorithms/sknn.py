"""SKNN (Jannach & Ludewig, RecSys 2017): session-based nearest neighbours over users' recent histories."""

from __future__ import annotations

import numpy as np
import polars as pl
import torch

from recsys.algorithms.base import Recommender
from recsys.device import resolve_device


class SKNN(Recommender):
    """SKNN (Jannach & Ludewig, RecSys 2017): the most similar past sessions vote for their items.

    The "session" is the user's last ``session_len`` items, weighted linearly
    by recency (most recent = 1, as in V-SKNN, Ludewig & Jannach 2018). Past
    sessions are the training users' histories. Candidates are the
    ``sample_size`` most recently active training users sharing at least one
    item with the session; the ``k`` most cosine-similar candidates (the user's
    own training row excluded) vote for their items with their similarity.
    Inductive: only the history is needed to score a user.
    """

    tasks = frozenset({"ranking"})
    inductive = True
    requires = frozenset({"timestamps"})
    space = {"k": [50, 100, 200], "sample_size": [500, 1000], "session_len": [10, 20, 50]}

    def __init__(self, k: int = 100, sample_size: int = 500, session_len: int = 20, device: str = "auto") -> None:
        self.k = k
        self.sample_size = sample_size
        self.session_len = session_len
        self.device = resolve_device(device)

    def fit(self, data) -> None:
        self.X = torch.as_tensor(data.history.matrix().toarray(), device=self.device)
        self.norms = self.X.sum(1).sqrt()
        last = data.train.group_by("user").agg(pl.col("timestamp").max())
        recency = np.full(data.n_users, -np.inf)
        recency[last["user"].to_numpy()] = last["timestamp"].to_numpy()
        # Rank (not raw time) keeps float32 exact; users without training rows are never candidates.
        order = np.argsort(np.argsort(recency, kind="stable"), kind="stable").astype(np.float32)
        order = np.where(np.isinf(recency), -np.inf, order).astype(np.float32)
        self.recency = torch.as_tensor(order, device=self.device)

    def score(self, users, history):
        seq = torch.as_tensor(history.sequences(users, self.session_len), device=self.device)
        valid = seq >= 0
        n = valid.sum(1, keepdim=True).clamp_min(1)
        age = valid.flip(1).cumsum(1).flip(1) - 1  # 0 for the most recent item
        weights = torch.where(valid, (n - age) / n, 0.0).float()
        session = torch.zeros((len(users), self.X.shape[1]), device=self.device)
        session.scatter_add_(1, seq.clamp_min(0), weights)
        sim = session @ self.X.T
        sim /= session.norm(dim=1, keepdim=True) * self.norms[None, :] + 1e-6
        sim[torch.arange(len(users), device=self.device), torch.as_tensor(users, device=self.device)] = 0.0
        recent = torch.where(sim > 0, self.recency[None, :], -torch.inf)
        r, cand = recent.topk(min(self.sample_size, sim.shape[1]), dim=1)
        cand_sim = torch.where(r > -torch.inf, sim.gather(1, cand), 0.0)
        values, j = cand_sim.topk(min(self.k, cand_sim.shape[1]), dim=1)
        votes = torch.zeros_like(sim).scatter_(1, cand.gather(1, j), values)
        return votes @ self.X
