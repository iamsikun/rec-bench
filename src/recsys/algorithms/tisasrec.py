"""TiSASRec (Li, Wang & McAuley, WSDM 2020): self-attention aware of the time intervals between items."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms.trainer import TorchRecommender, minibatches


def _heads(t: torch.Tensor, n_heads: int) -> torch.Tensor:
    """``(..., N, D)`` to ``(..., H, N, D / H)``."""
    return t.view(*t.shape[:-1], n_heads, t.shape[-1] // n_heads).transpose(-3, -2)


class TimeBlock(nn.Module):
    """One TiSASRec block: interval- and position-aware causal attention, then a point-wise feed-forward network."""

    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads, self.dropout = n_heads, dropout
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.norm_attn, self.norm_ffn = nn.LayerNorm(dim, eps=1e-8), nn.LayerNorm(dim, eps=1e-8)
        self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout), nn.ReLU(), nn.Linear(dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, x, pos_k, pos_v, time_k, time_v, buckets, blocked):
        B, L, D = x.shape
        H = self.n_heads
        query = self.norm_attn(x)
        q = _heads(self.q(query), H)
        k, v = _heads(self.k(x) + pos_k, H), _heads(self.v(x) + pos_v, H)  # (B, H, L, D/H)
        tk, tv = _heads(time_k, H), _heads(time_v, H)  # (H, span + 1, D/H)
        idx = buckets.unsqueeze(1).expand(B, H, L, L).contiguous()
        # q . E_K[bucket(i, j)] read from q @ E_K^T instead of materializing (B, L, L, D) interval embeddings.
        scores = (q @ k.transpose(-1, -2) + (q @ tk.transpose(-1, -2)).gather(-1, idx)) / math.sqrt(D // H)
        attn = F.dropout(scores.masked_fill(blocked.unsqueeze(1), -torch.inf).softmax(-1), self.dropout, self.training)
        # sum_j a_ij E_V[bucket(i, j)] = (attention mass per bucket) @ E_V.
        per_bucket = attn.new_zeros(B, H, L, tv.shape[-2]).scatter_add(-1, idx, attn)
        out = (attn @ v + per_bucket @ tv).transpose(1, 2).reshape(B, L, D)
        x = self.norm_ffn(query + out)
        return x + self.ffn(x)


class TiSASRec(TorchRecommender):
    """TiSASRec (Li, Wang & McAuley, WSDM 2020): SASRec whose attention sees clipped personal time intervals.

    The interval between items ``i`` and ``j`` of a user is
    ``min(round(|t_i - t_j| / r_u), time_span)``, where ``r_u`` is the
    user's smallest non-zero gap between consecutive items. Learned interval
    embeddings and absolute position embeddings are added to keys and to
    values; attention is causal and ignores padding. Training uses
    full-softmax cross-entropy at every position, as our SASRec does.

    Deviations from the paper: full-softmax CE replaces one-negative BCE;
    positions count back from the most recent item (so left padding does
    not change scores) and ``r_u`` is taken over the ``max_len`` window
    being encoded rather than the whole history; intervals round the scaled
    difference instead of differencing rounded scaled timestamps; dropout
    acts on the interval embedding tables rather than per lookup.
    """

    tasks = frozenset({"ranking"})
    inductive = True
    requires = frozenset({"timestamps"})
    space: ClassVar[dict[str, list]] = {"time_span": [64, 256, 1024], "dropout": [0.2, 0.5], "max_len": [50, 200]}

    def __init__(self, dim: int = 50, n_layers: int = 2, n_heads: int = 1, max_len: int = 50, time_span: int = 256,
                 dropout: float = 0.2, *, epochs: int = 200, batch_size: int = 128, lr: float = 1e-3,
                 patience: int | None = 10, **kwargs) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, **kwargs)
        self.dim, self.n_layers, self.n_heads, self.max_len = dim, n_layers, n_heads, max_len
        self.time_span, self.dropout = time_span, dropout

    def build(self, data) -> None:
        # Item ids are shifted by one so that 0 is padding.
        self.item_emb = nn.Embedding(data.n_items + 1, self.dim, padding_idx=0)
        self.pos_k, self.pos_v = nn.Embedding(self.max_len, self.dim), nn.Embedding(self.max_len, self.dim)
        self.time_k = nn.Embedding(self.time_span + 1, self.dim)
        self.time_v = nn.Embedding(self.time_span + 1, self.dim)
        for emb in (self.item_emb, self.pos_k, self.pos_v, self.time_k, self.time_v):
            nn.init.normal_(emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.blocks = nn.ModuleList(TimeBlock(self.dim, self.n_heads, self.dropout) for _ in range(self.n_layers))
        self.norm = nn.LayerNorm(self.dim, eps=1e-8)
        users = np.flatnonzero(data.history.lengths() >= 2)
        self.register_buffer("_seqs", torch.as_tensor(data.history.sequences(users, self.max_len + 1) + 1),
                             persistent=False)
        self.register_buffer("_times", torch.as_tensor(data.history.time_sequences(users, self.max_len + 1)),
                             persistent=False)

    def intervals(self, seq: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Clipped personal interval buckets ``(B, L, L)`` for right-aligned items and their timestamps."""
        valid = seq > 0
        gaps = (times[:, 1:] - times[:, :-1]).float()
        scale = gaps.masked_fill(~valid[:, :-1] | (gaps <= 0), torch.inf).amin(1)
        scale = scale.masked_fill(scale.isinf(), 1.0)[:, None, None]
        diff = (times[:, :, None] - times[:, None, :]).abs().float()
        return (diff / scale).round().clamp(max=self.time_span).long()

    def encode(self, seq: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Hidden states ``(B, L, dim)``; position ``t`` sees items ``<= t`` (and the window's time scale)."""
        B, L = seq.shape
        valid = seq > 0
        buckets = self.intervals(seq, times)
        positions = torch.arange(L - 1, -1, -1, device=seq.device)
        def drop(t: torch.Tensor) -> torch.Tensor:
            return F.dropout(t, self.dropout, self.training)

        pos_k, pos_v = drop(self.pos_k(positions)), drop(self.pos_v(positions))
        time_k, time_v = drop(self.time_k.weight), drop(self.time_v.weight)
        x = drop(self.item_emb(seq) * math.sqrt(self.dim)) * valid.unsqueeze(-1)
        eye = torch.eye(L, dtype=torch.bool, device=seq.device)
        blocked = (torch.ones(L, L, dtype=torch.bool, device=seq.device).triu(1) | ~valid[:, None, :]) & ~eye
        for block in self.blocks:
            x = block(x, pos_k, pos_v, time_k, time_v, buckets, blocked) * valid.unsqueeze(-1)
        return self.norm(x)

    def batches(self, data):
        for idx in minibatches(len(self._seqs), self.batch_size, self.device):
            yield self._seqs[idx], self._times[idx]

    def loss(self, batch):
        seq, times = batch
        inputs, targets = seq[:, :-1], seq[:, 1:]
        targets = targets.masked_fill(inputs == 0, 0)
        logits = self.encode(inputs, times[:, :-1]) @ self.item_emb.weight.T
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=0)

    def score(self, users, history):
        seq = torch.as_tensor(history.sequences(users, self.max_len) + 1, device=self.device)
        times = torch.as_tensor(history.time_sequences(users, self.max_len), device=self.device)
        return self.encode(seq, times)[:, -1] @ self.item_emb.weight[1:].T
