"""Sampling-bias-corrected two-tower retrieval (Yi et al., RecSys 2019).

Both towers are MLPs with L2-normalized outputs and a temperature. Training
uses the in-batch softmax, whose negatives are drawn with the item's in-batch
frequency; each logit is corrected by subtracting ``log p_j``, where ``p_j`` is
estimated by the paper's streaming algorithm (Algorithm 2): per item, ``A``
holds the last step it was seen and ``B`` an exponential moving average of the
gap between sightings, so ``p_j = 1 / B[j]``. Items are indexed directly
instead of hashed. The correction applies only in training; serving is plain
cosine MIPS. The user tower reads the averaged recent history plus user
features (the paper's seed-plus-history query), so the model is inductive.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from recsys.algorithms._towers import FeatureEmbedding, HistoryTwoTower, embedding, in_batch_softmax, mlp


class TwoTowerLogQ(HistoryTwoTower):
    """TwoTowerLogQ (Yi et al., RecSys 2019): in-batch softmax two-tower with streaming-frequency logQ correction."""

    normalize = True
    space = {"dim": [64, 128], "temperature": [0.05, 0.1, 0.2], "max_len": [20, 50, 200]}

    def __init__(self, dim: int = 64, hidden: tuple[int, ...] = (128,), max_len: int = 50,
                 temperature: float = 0.05, alpha: float = 0.01, *, epochs: int = 100, batch_size: int = 4096,
                 lr: float = 1e-3, patience: int | None = 10, **kwargs) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, **kwargs)
        self.dim = dim
        self.hidden = tuple(hidden)
        self.max_len = max_len
        self.temperature = temperature
        self.alpha = alpha

    def build(self, data) -> None:
        self.build_user_tower(data)
        self.item_emb = embedding(data.n_items, self.dim)
        self.item_feat = FeatureEmbedding(data.item_features, self.dim)
        self.item_tower = mlp([self.dim + self.item_feat.out_dim, *self.hidden, self.dim])
        self.register_buffer("_last_seen", torch.full((data.n_items,), -1.0), persistent=False)
        self.register_buffer("_gap", torch.ones(data.n_items), persistent=False)
        self.register_buffer("_step", torch.zeros(()), persistent=False)

    def item_vectors(self, items=None):
        items = torch.arange(self.n_items, device=self.device) if items is None else items
        x = torch.cat([self.item_emb(items), self.item_feat(items)], -1)
        return F.normalize(self.item_tower(x), dim=-1)

    @torch.no_grad()
    def log_frequency(self, items: torch.Tensor) -> torch.Tensor:
        """Update the streaming estimates with this batch and return ``log p`` for ``items``."""
        self._step += 1
        seen = items.unique()
        last = self._last_seen[seen]
        gap = (1 - self.alpha) * self._gap[seen] + self.alpha * (self._step - last)
        self._gap[seen] = torch.where(last >= 0, gap, self._step.expand_as(last))  # first sighting: gap so far
        self._last_seen[seen] = self._step
        return -torch.log(self._gap[items])

    def loss(self, batch):
        users, seq, target = batch
        log_q = self.log_frequency(target)
        u = self.encode_user(users, seq)
        return in_batch_softmax(u, self.item_vectors(target), target, self.temperature, log_q)
