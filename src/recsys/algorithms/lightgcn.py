"""LightGCN (He et al., SIGIR 2020): linear graph propagation over the user-item graph."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms.trainer import TorchRecommender, minibatches


class LightGCN(TorchRecommender):
    """LightGCN (He et al., SIGIR 2020): mean of linear propagations over the user-item graph, BPR loss."""

    tasks = frozenset({"ranking"})
    space = {"n_layers": [2, 3], "lr": [1e-3, 5e-3], "reg": [1e-5, 1e-4, 1e-3]}

    def __init__(self, dim: int = 64, n_layers: int = 3, reg: float = 1e-4, *, epochs: int = 200,
                 batch_size: int = 4096, lr: float = 1e-3, patience: int | None = 10, **kwargs) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, **kwargs)
        self.dim = dim
        self.n_layers = n_layers
        self.reg = reg
        self._cache: torch.Tensor | None = None

    def build(self, data) -> None:
        self.n_users, self.n_items = data.n_users, data.n_items
        self.emb = nn.Embedding(data.n_users + data.n_items, self.dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        R = data.history.matrix()
        R.data[:] = 1.0
        A = sp.bmat([[None, R], [R.T, None]], format="coo")
        deg = np.asarray(A.sum(1)).ravel()
        d = np.power(deg, -0.5, where=deg > 0, out=np.zeros_like(deg))
        values = d[A.row] * d[A.col]
        adj = torch.sparse_coo_tensor(np.vstack([A.row, A.col]), values.astype(np.float32), A.shape).coalesce()
        self.register_buffer("_adj", adj, persistent=False)
        self.register_buffer("_users", torch.tensor(data.train["user"].to_numpy()), persistent=False)
        self.register_buffer("_items", torch.tensor(data.train["item"].to_numpy()), persistent=False)

    def propagate(self) -> torch.Tensor:
        e = self.emb.weight
        total = e
        for _ in range(self.n_layers):
            e = torch.sparse.mm(self._adj, e)
            total = total + e
        return total / (self.n_layers + 1)

    def train(self, mode: bool = True):
        self._cache = None  # weights may change (training, best-state restore)
        return super().train(mode)

    def batches(self, data):
        for idx in minibatches(len(self._users), self.batch_size, self.device):
            neg = torch.randint(self.n_items, (len(idx),), device=self.device)
            yield self._users[idx], self._items[idx] + self.n_users, neg + self.n_users

    def loss(self, batch):
        u, i, j = batch
        out = self.propagate()
        margin = (out[u] * (out[i] - out[j])).sum(1)
        ego = self.emb(torch.cat([u, i, j]))
        return -F.logsigmoid(margin).mean() + self.reg * ego.square().sum() / len(u)

    def score(self, users, history):
        if self._cache is None:
            self._cache = self.propagate()
        out = self._cache
        u = torch.as_tensor(users, device=self.device)
        return out[u] @ out[self.n_users :].T
