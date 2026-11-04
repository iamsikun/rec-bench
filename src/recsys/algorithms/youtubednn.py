"""YouTube DNN candidate generation (Covington et al., RecSys 2016).

The user vector is a ReLU MLP over the average embedding of the user's recent
watches concatenated with user side-feature embeddings; it is trained as
extreme multiclass classification of the next watched item. The paper uses a
sampled softmax for its millions of videos; benchmark catalogs are small
enough for the exact full softmax (the default). ``n_negatives`` switches to a
sampled softmax with shared uniform negatives (constant proposal, so no logQ
term is needed). Following the paper, examples predict each next watch from
the watches before it only, and serving is a nearest-neighbor search over the
softmax item vectors (plus projected item side features when available).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms._towers import FeatureEmbedding, HistoryTwoTower, embedding


class YouTubeDNN(HistoryTwoTower):
    """YouTubeDNN (Covington et al., RecSys 2016): MLP over averaged watch-history embeddings, softmax over items."""

    space = {"dim": [64, 128], "max_len": [20, 50, 200], "lr": [1e-3, 3e-3]}

    def __init__(self, dim: int = 64, hidden: tuple[int, ...] = (128,), max_len: int = 50,
                 n_negatives: int | None = None, *, epochs: int = 100, batch_size: int = 4096,
                 lr: float = 1e-3, patience: int | None = 10, **kwargs) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, **kwargs)
        self.dim = dim
        self.hidden = tuple(hidden)
        self.max_len = max_len
        self.n_negatives = n_negatives

    def build(self, data) -> None:
        self.build_user_tower(data)
        self.item_emb = embedding(data.n_items, self.dim)
        self.item_feat = FeatureEmbedding(data.item_features, self.dim)
        out = self.item_feat.out_dim
        self.item_proj = nn.Linear(out, self.dim, bias=False) if out else None

    def item_vectors(self, items=None):
        items = torch.arange(self.n_items, device=self.device) if items is None else items
        v = self.item_emb(items)
        return v if self.item_proj is None else v + self.item_proj(self.item_feat(items))

    def loss(self, batch):
        users, seq, target = batch
        u = self.encode_user(users, seq)
        if self.n_negatives is None:
            return F.cross_entropy(u @ self.item_vectors().T, target)
        neg = torch.randint(self.n_items, (self.n_negatives,), device=self.device)
        pos = (u * self.item_vectors(target)).sum(-1, keepdim=True)
        rest = (u @ self.item_vectors(neg).T).masked_fill(neg[None, :] == target[:, None], -1e4)
        return F.cross_entropy(torch.cat([pos, rest], 1), torch.zeros_like(target))
