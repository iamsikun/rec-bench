"""NeuMF (He et al., WWW 2017): neural collaborative filtering fusing GMF and an MLP."""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms.trainer import TorchRecommender, minibatches


class NeuMF(TorchRecommender):
    """NeuMF (He et al., WWW 2017): GMF and MLP branches over ID embeddings, fused linearly.

    Trained pointwise with binary cross-entropy on each training interaction
    plus ``num_neg`` uniformly sampled negatives (the paper's setting; no
    GMF/MLP pre-training). ``layers`` are the MLP widths; the MLP embeddings
    have ``layers[0] // 2`` dimensions each.
    """

    tasks = frozenset({"ranking"})
    space: ClassVar[dict[str, list]] = {"dim": [16, 32, 64], "num_neg": [4, 8], "lr": [1e-3, 3e-3]}

    def __init__(self, dim: int = 32, layers: tuple[int, ...] = (128, 64, 32), num_neg: int = 4,
                 dropout: float = 0.0, *, epochs: int = 100, batch_size: int = 1024, lr: float = 1e-3,
                 patience: int | None = 5, **kwargs) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, **kwargs)
        self.dim = dim
        self.layers = tuple(layers)
        self.num_neg = num_neg
        self.dropout = dropout

    def build(self, data) -> None:
        self.n_items = data.n_items
        half = self.layers[0] // 2
        self.gmf_user = nn.Embedding(data.n_users, self.dim)
        self.gmf_item = nn.Embedding(data.n_items, self.dim)
        self.mlp_user = nn.Embedding(data.n_users, half)
        self.mlp_item = nn.Embedding(data.n_items, half)
        for emb in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(emb.weight, std=0.02)
        # The first MLP layer acts on [user; item], split into two halves so scoring can broadcast.
        self.first_user = nn.Linear(half, self.layers[1] if len(self.layers) > 1 else half)
        self.first_item = nn.Linear(half, self.first_user.out_features, bias=False)
        widths = [self.first_user.out_features, *self.layers[2:]]
        self.mlp = nn.Sequential(*[
            m for a, b in zip(widths[:-1], widths[1:]) for m in (nn.Dropout(self.dropout), nn.Linear(a, b), nn.ReLU())
        ])
        self.out_gmf = nn.Linear(self.dim, 1, bias=False)
        self.out_mlp = nn.Linear(widths[-1], 1)
        self.register_buffer("_users", torch.tensor(data.train["user"].to_numpy()), persistent=False)
        self.register_buffer("_items", torch.tensor(data.train["item"].to_numpy()), persistent=False)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Logits for aligned ``(user, item)`` pairs."""
        gmf = self.gmf_user(u) * self.gmf_item(i)
        h = F.relu(self.first_user(self.mlp_user(u)) + self.first_item(self.mlp_item(i)))
        return (self.out_gmf(gmf) + self.out_mlp(self.mlp(h))).squeeze(-1)

    def batches(self, data):
        for idx in minibatches(len(self._users), self.batch_size, self.device):
            u, i = self._users[idx], self._items[idx]
            neg = torch.randint(self.n_items, (len(idx), self.num_neg), device=self.device)
            users = torch.cat([u, u.repeat_interleave(self.num_neg)])
            items = torch.cat([i, neg.flatten()])
            labels = torch.cat([torch.ones_like(u, dtype=torch.float32),
                                torch.zeros(neg.numel(), device=self.device)])
            yield users, items, labels

    def loss(self, batch):
        u, i, y = batch
        return F.binary_cross_entropy_with_logits(self(u, i), y)

    def score(self, users, history):
        u = torch.as_tensor(users, device=self.device)
        # GMF branch is bilinear: (p_u * w) . q_i for all items at once.
        gmf = (self.gmf_user(u) * self.out_gmf.weight[0]) @ self.gmf_item.weight.T
        item_part = self.first_item(self.mlp_item.weight)  # (n_items, h)
        user_part = self.first_user(self.mlp_user(u))  # (B, h)
        chunk = max(1, 2**24 // (self.n_items * item_part.shape[1]))
        mlp = torch.cat([
            self.out_mlp(self.mlp(F.relu(user_part[s : s + chunk, None] + item_part[None]))).squeeze(-1)
            for s in range(0, len(u), chunk)
        ])
        return gmf + mlp
