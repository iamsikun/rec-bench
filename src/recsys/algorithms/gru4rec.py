"""GRU4Rec (Hidasi et al., ICLR 2016; CIKM 2018): a GRU over the item sequence predicts the next item."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms.trainer import TorchRecommender, minibatches


def left_align(seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Move the left padding (``0``) of right-aligned ``(B, L)`` rows to the right; also return lengths."""
    lengths = (seq > 0).sum(1)
    idx = (torch.arange(seq.shape[1], device=seq.device)[None] + seq.shape[1] - lengths[:, None]) % seq.shape[1]
    return seq.gather(1, idx), lengths


class GRU4Rec(TorchRecommender):
    """GRU4Rec (Hidasi et al., ICLR 2016): a GRU over the recent items; its state scores the next item.

    Each sequence is left-aligned before the GRU, so the hidden state starts
    at zero on the first real item and padding never enters the recurrence.
    ``loss="ce"`` is softmax cross-entropy over the full catalog at every
    position (Hidasi & Czapp, RecSys 2023 found CE the strongest loss);
    ``loss="sampled"`` uses ``n_samples`` uniform negatives shared across the
    batch plus in-sequence targets (the CIKM 2018 "additional samples" softmax).
    """

    tasks = frozenset({"ranking"})
    requires = frozenset({"timestamps"})
    inductive = True
    space: ClassVar[dict[str, list]] = {"hidden": [64, 128, 256], "dropout": [0.1, 0.2, 0.3], "lr": [1e-3, 3e-3]}

    def __init__(self, dim: int = 64, hidden: int = 128, n_layers: int = 1, max_len: int = 50,
                 dropout: float = 0.2, loss: str = "ce", n_samples: int = 2048, *, epochs: int = 200,
                 batch_size: int = 128, lr: float = 1e-3, patience: int | None = 10, **kwargs) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, **kwargs)
        if loss not in ("ce", "sampled"):
            raise ValueError(f"loss must be 'ce' or 'sampled', got {loss!r}")
        self.dim = dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.max_len = max_len
        self.dropout = dropout
        self.loss_type = loss
        self.n_samples = n_samples

    def build(self, data) -> None:
        self.n_items = data.n_items
        self.item_emb = nn.Embedding(data.n_items + 1, self.dim, padding_idx=0)  # 0 = padding
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.gru = nn.GRU(self.dim, self.hidden, self.n_layers, batch_first=True,
                          dropout=self.dropout if self.n_layers > 1 else 0.0)
        self.drop = nn.Dropout(self.dropout)
        self.out = nn.Linear(self.hidden, self.dim)
        self.out_bias = nn.Parameter(torch.zeros(data.n_items))
        users = np.flatnonzero(data.history.lengths() >= 2)
        seqs = data.history.sequences(users, self.max_len + 1) + 1
        self.register_buffer("_seqs", torch.as_tensor(seqs), persistent=False)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """Per-position outputs ``(B, L, dim)`` for *left-aligned* sequences (padding on the right)."""
        h, _ = self.gru(self.drop(self.item_emb(seq)))
        return self.out(self.drop(h))

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.item_emb.weight[1:].T + self.out_bias

    def batches(self, data):
        for idx in minibatches(len(self._seqs), self.batch_size, self.device):
            yield left_align(self._seqs[idx])[0]

    def loss(self, seq):
        inputs, targets = seq[:, :-1], seq[:, 1:]
        valid = (inputs > 0) & (targets > 0)
        h = self.encode(inputs)[valid]
        target = targets[valid] - 1
        if self.loss_type == "ce":
            return F.cross_entropy(self.logits(h), target)
        # Sampled softmax: the batch's targets plus shared uniform negatives as the candidate set.
        negatives = torch.randint(self.n_items, (self.n_samples,), device=seq.device)
        candidates = torch.cat([target, negatives])
        logits = h @ self.item_emb.weight[candidates + 1].T + self.out_bias[candidates]
        logits = logits.masked_fill(candidates[None] == target[:, None], -torch.inf)
        logits[torch.arange(len(target)), torch.arange(len(target))] = (h * self.item_emb(target + 1)).sum(1) \
            + self.out_bias[target]
        return F.cross_entropy(logits, torch.arange(len(target), device=seq.device))

    def score(self, users, history):
        seq = torch.as_tensor(history.sequences(users, self.max_len) + 1, device=self.device)
        aligned, lengths = left_align(seq)
        h = self.encode(aligned)
        last = h[torch.arange(len(seq), device=self.device), (lengths - 1).clamp(min=0)]
        return self.logits(last)
