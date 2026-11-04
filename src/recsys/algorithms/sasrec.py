"""SASRec (Kang & McAuley, ICDM 2018): causal self-attention over the user's item sequence.

Trained with full-softmax cross-entropy over the catalog, which is much
stronger than the original one-negative BCE (Klenitskiy & Vasilev, RecSys 2023).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms.trainer import TorchRecommender, minibatches


class SASRec(TorchRecommender):
    """SASRec (Kang & McAuley, ICDM 2018): causal self-attention over the item sequence, full-softmax CE."""

    tasks = frozenset({"ranking"})
    inductive = True
    requires = frozenset({"timestamps"})
    space = {"max_len": [50, 200], "dropout": [0.1, 0.2, 0.5], "n_layers": [2, 3]}

    def __init__(self, dim: int = 64, n_layers: int = 2, n_heads: int = 1, max_len: int = 50,
                 dropout: float = 0.2, *, epochs: int = 200, batch_size: int = 128, lr: float = 1e-3,
                 patience: int | None = 10, **kwargs) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, **kwargs)
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_len = max_len
        self.dropout = dropout

    def build(self, data) -> None:
        # Item ids are shifted by one so that 0 is padding.
        self.item_emb = nn.Embedding(data.n_items + 1, self.dim, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_len, self.dim)
        for emb in (self.item_emb, self.pos_emb):
            nn.init.normal_(emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        layer = nn.TransformerEncoderLayer(
            self.dim, self.n_heads, dim_feedforward=self.dim, dropout=self.dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, self.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(self.dim)
        self.drop = nn.Dropout(self.dropout)
        users = np.flatnonzero(data.history.lengths() >= 2)
        seqs = data.history.sequences(users, self.max_len + 1) + 1
        self.register_buffer("_seqs", torch.as_tensor(seqs), persistent=False)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """Hidden states ``(B, L, dim)`` for left-padded item sequences ``(B, L)``.

        Positions count back from the most recent item and padding keys are
        masked, so outputs depend neither on future items nor on the amount
        of padding.
        """
        B, L = seq.shape
        positions = torch.arange(L - 1, -1, -1, device=seq.device)
        x = self.item_emb(seq) * math.sqrt(self.dim) + self.pos_emb(positions)
        x = self.drop(x) * (seq > 0).unsqueeze(-1)
        eye = torch.eye(L, dtype=torch.bool, device=seq.device)
        blocked = (torch.ones(L, L, dtype=torch.bool, device=seq.device).triu(1) | (seq == 0)[:, None, :]) & ~eye
        mask = torch.zeros(blocked.shape, device=seq.device).masked_fill(blocked, -torch.inf)
        return self.norm(self.encoder(x, mask=mask.repeat_interleave(self.n_heads, dim=0)))

    def batches(self, data):
        for idx in minibatches(len(self._seqs), self.batch_size, self.device):
            yield self._seqs[idx]

    def loss(self, seq):
        inputs, targets = seq[:, :-1], seq[:, 1:]
        targets = targets.masked_fill(inputs == 0, 0)
        logits = self.encode(inputs) @ self.item_emb.weight.T
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=0)

    def score(self, users, history):
        seq = torch.as_tensor(history.sequences(users, self.max_len) + 1, device=self.device)
        return self.encode(seq)[:, -1] @ self.item_emb.weight[1:].T
