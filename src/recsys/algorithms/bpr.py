"""BPR-MF (Rendle et al., UAI 2009): matrix factorization with the pairwise BPR loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms.trainer import TorchRecommender, minibatches


class BPR(TorchRecommender):
    """BPR-MF (Rendle et al., UAI 2009): dot-product embeddings trained with the pairwise BPR loss."""

    tasks = frozenset({"ranking"})
    space = {"dim": [64, 128], "lr": [1e-3, 5e-3], "reg": [1e-5, 1e-4, 1e-3]}

    def __init__(self, dim: int = 64, reg: float = 1e-4, *, epochs: int = 200, batch_size: int = 4096,
                 lr: float = 1e-3, patience: int | None = 10, **kwargs) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, **kwargs)
        self.dim = dim
        self.reg = reg

    def build(self, data) -> None:
        self.n_items = data.n_items
        self.user_emb = nn.Embedding(data.n_users, self.dim)
        self.item_emb = nn.Embedding(data.n_items, self.dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        self.register_buffer("_users", torch.tensor(data.train["user"].to_numpy()), persistent=False)
        self.register_buffer("_items", torch.tensor(data.train["item"].to_numpy()), persistent=False)

    def batches(self, data):
        for idx in minibatches(len(self._users), self.batch_size, self.device):
            neg = torch.randint(self.n_items, (len(idx),), device=self.device)
            yield self._users[idx], self._items[idx], neg

    def loss(self, batch):
        u, i, j = batch
        eu, ei, ej = self.user_emb(u), self.item_emb(i), self.item_emb(j)
        margin = (eu * (ei - ej)).sum(1)
        l2 = (eu.square().sum() + ei.square().sum() + ej.square().sum()) / len(u)
        return -F.logsigmoid(margin).mean() + self.reg * l2

    def score(self, users, history):
        u = torch.as_tensor(users, device=self.device)
        return self.user_emb.weight[u] @ self.item_emb.weight.T
