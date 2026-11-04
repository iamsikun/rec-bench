"""DSSM (Huang et al., CIKM 2013), in the standard two-tower adaptation for recommendation.

The original maps query and document text through letter-trigram MLPs and
trains a softmax of each clicked document against four uniformly sampled
unclicked ones. Here the "query" tower reads the user id and user side
features, the "document" tower the item id and item side features; both are
L2-normalized (cosine similarity, scaled by ``1 / temperature``). Negatives are
``n_negatives`` uniformly sampled items shared across the batch (a larger pool
than the paper's four; uniform sampling keeps the model's popularity signal,
which in-batch negatives would penalize). ``n_negatives=None`` switches to the
in-batch softmax common in later two-tower work.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from recsys.algorithms._towers import FeatureEmbedding, TwoTower, embedding, in_batch_softmax, mlp
from recsys.algorithms.trainer import minibatches


class DSSM(TwoTower):
    """DSSM (Huang et al., CIKM 2013): cosine two-tower over id + side-feature embeddings, in-batch softmax."""

    requires = frozenset({"user_features", "item_features"})
    space = {"dim": [64, 128], "temperature": [0.05, 0.1, 0.2], "n_negatives": [256, 1024]}

    def __init__(self, dim: int = 64, hidden: tuple[int, ...] = (128,), temperature: float = 0.05,
                 n_negatives: int | None = 256, *,
                 epochs: int = 100, batch_size: int = 4096, lr: float = 1e-3, patience: int | None = 10,
                 **kwargs) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, **kwargs)
        self.dim = dim
        self.hidden = tuple(hidden)
        self.temperature = temperature
        self.n_negatives = n_negatives

    def build(self, data) -> None:
        self.n_items = data.n_items
        self.user_emb = embedding(data.n_users, self.dim)
        self.item_emb = embedding(data.n_items, self.dim)
        self.user_feat = FeatureEmbedding(data.user_features, self.dim)
        self.item_feat = FeatureEmbedding(data.item_features, self.dim)
        self.user_tower = mlp([self.dim + self.user_feat.out_dim, *self.hidden, self.dim])
        self.item_tower = mlp([self.dim + self.item_feat.out_dim, *self.hidden, self.dim])
        self.register_buffer("_users", torch.as_tensor(data.train["user"].to_numpy()), persistent=False)
        self.register_buffer("_items", torch.as_tensor(data.train["item"].to_numpy()), persistent=False)

    def user_vectors(self, users, history=None):
        x = torch.cat([self.user_emb(users), self.user_feat(users)], -1)
        return F.normalize(self.user_tower(x), dim=-1)

    def item_vectors(self, items=None):
        items = torch.arange(self.n_items, device=self.device) if items is None else items
        x = torch.cat([self.item_emb(items), self.item_feat(items)], -1)
        return F.normalize(self.item_tower(x), dim=-1)

    def batches(self, data):
        for idx in minibatches(len(self._users), self.batch_size, self.device):
            yield self._users[idx], self._items[idx]

    def loss(self, batch):
        users, items = batch
        u = self.user_vectors(users)
        if self.n_negatives is None:
            return in_batch_softmax(u, self.item_vectors(items), items, self.temperature)
        neg = torch.randint(self.n_items, (self.n_negatives,), device=self.device)
        pos = (u * self.item_vectors(items)).sum(-1, keepdim=True)
        rest = (u @ self.item_vectors(neg).T).masked_fill(neg[None, :] == items[:, None], -1e4)
        logits = torch.cat([pos, rest], 1) / self.temperature
        return F.cross_entropy(logits, torch.zeros_like(items))
