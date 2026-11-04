"""Shared training for sequence encoders whose token mixing is not causal (FMLP-Rec, BSARec).

A global filter along the sequence lets every position see later items, so
next-item prediction at every position would leak targets. As in the BSARec
reference code (which also re-implements FMLP-Rec), each training example is
one prefix whose target is among the user's last ``max_len`` training items,
padded on the left to exactly ``max_len``, and full-softmax cross-entropy is
applied at the final position only. Inputs are always ``max_len`` long, so
scores depend on how much left padding a short sequence carries.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms.trainer import TorchRecommender, minibatches


class FeedForward(nn.Module):
    """Post-norm point-wise feed-forward block: ``LayerNorm(x + Dropout(W2 GELU(W1 x)))``."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(dim, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class LastPositionRecommender(TorchRecommender):
    """Base for encoders trained on prefixes with cross-entropy at the last position; subclasses add ``layers``."""

    tasks = frozenset({"ranking"})
    inductive = True
    requires = frozenset({"timestamps"})

    def __init__(self, dim: int, n_layers: int, max_len: int, dropout: float, *, epochs: int = 200,
                 batch_size: int = 256, lr: float = 1e-3, patience: int | None = 10, **kwargs) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, **kwargs)
        self.dim, self.n_layers, self.max_len, self.dropout = dim, n_layers, max_len, dropout

    def build(self, data) -> None:
        # Item ids are shifted by one so that 0 is padding.
        self.item_emb = nn.Embedding(data.n_items + 1, self.dim, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_len, self.dim)
        for emb in (self.item_emb, self.pos_emb):
            nn.init.normal_(emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.norm = nn.LayerNorm(self.dim, eps=1e-12)
        h = data.history
        starts, stops = np.repeat(h.indptr[:-1], h.lengths()), np.repeat(h.indptr[1:], h.lengths())
        ends = np.arange(len(h.items))
        keep = (ends > starts) & (ends >= stops - self.max_len)  # targets: the last max_len items, not the first
        self.register_buffer("_items", torch.as_tensor(h.items + 1), persistent=False)
        self.register_buffer("_starts", torch.as_tensor(starts[keep]), persistent=False)
        self.register_buffer("_ends", torch.as_tensor(ends[keep]), persistent=False)

    def embed(self, seq: torch.Tensor) -> torch.Tensor:
        """Item plus absolute position embeddings (left to right, as in the reference code), normalized."""
        x = self.item_emb(seq) + self.pos_emb(torch.arange(seq.shape[1], device=seq.device))
        return F.dropout(self.norm(x), self.dropout, self.training)

    @abstractmethod
    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """Hidden states ``(B, max_len, dim)`` for left-padded item sequences ``(B, max_len)``."""

    def batches(self, data):
        offsets = torch.arange(-self.max_len, 0, device=self.device)
        for idx in minibatches(len(self._ends), self.batch_size, self.device):
            ends = self._ends[idx]
            pos = ends[:, None] + offsets
            seq = torch.where(pos >= self._starts[idx, None], self._items[pos.clamp(min=0)], 0)
            yield seq, self._items[ends]

    def loss(self, batch):
        seq, target = batch
        return F.cross_entropy(self.encode(seq)[:, -1] @ self.item_emb.weight.T, target)

    def score(self, users, history):
        seq = torch.as_tensor(history.sequences(users, self.max_len) + 1, device=self.device)
        return self.encode(seq)[:, -1] @ self.item_emb.weight[1:].T
