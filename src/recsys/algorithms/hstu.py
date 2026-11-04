"""HSTU (Zhai et al., ICML 2024): Hierarchical Sequential Transduction Units for generative recommendation.

Each layer projects the sequence into gating ``U``, values ``V``, queries and
keys with one SiLU-activated linear map, replaces softmax attention with
pointwise ``SiLU(QK^T + rab) / n``, and gates the normalized attention output
with ``U`` before a residual output projection.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms.trainer import TorchRecommender, minibatches


class HSTULayer(nn.Module):
    """One HSTU block: pointwise projection, spatial aggregation with relative position bias, pointwise transform."""

    def __init__(self, dim: int, n_heads: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.n_heads, self.max_len, self.dropout = n_heads, max_len, dropout
        self.norm_in, self.norm_attn = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.uvqk = nn.Linear(dim, 4 * dim)
        self.out = nn.Linear(dim, dim)
        self.rel_pos = nn.Parameter(torch.zeros(n_heads, 2 * max_len - 1))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H = self.n_heads
        u, v, q, k = F.silu(self.uvqk(self.norm_in(x))).chunk(4, dim=-1)
        q, k, v = (t.view(B, L, H, D // H).transpose(1, 2) for t in (q, k, v))
        pos = torch.arange(L, device=x.device)
        rab = self.rel_pos[:, (pos[None, :] - pos[:, None]) + self.max_len - 1]  # (H, L, L)
        causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
        mask = causal & valid[:, None, None, :]  # (B, 1, L, L): attend to earlier, non-padding keys
        attn = F.silu(q @ k.transpose(-1, -2) + rab) * mask / L
        y = (attn @ v).transpose(1, 2).reshape(B, L, D)
        y = self.out(F.dropout(self.norm_attn(y) * u, self.dropout, self.training))
        return x + y


class HSTU(TorchRecommender):
    """HSTU (Zhai et al., ICML 2024): causal pointwise-attention transducer for next-item retrieval.

    Scores are dot products of the last hidden state with (optionally
    L2-normalized) item embeddings. Training uses full softmax over the
    catalog by default, or sampled softmax with ``n_negatives`` uniform
    negatives shared across the batch (the paper's large-catalog setting).

    Deviations from the paper: the relative attention bias uses positions
    only, because ``History`` passed to :meth:`score` carries no timestamps
    (the paper adds a bucketized time-difference bias); no action/feature
    tokens are interleaved (items only); attention is normalized by the
    padded length ``L`` as in the reference implementation.
    """

    tasks = frozenset({"ranking"})
    requires = frozenset({"timestamps"})
    inductive = True
    space = {"max_len": [50, 200], "dropout": [0.1, 0.2, 0.5], "n_layers": [2, 4]}

    def __init__(self, dim: int = 64, n_layers: int = 2, n_heads: int = 1, max_len: int = 200,
                 dropout: float = 0.2, n_negatives: int | None = None, temperature: float = 0.05,
                 *, epochs: int = 200, batch_size: int = 128, lr: float = 1e-3, patience: int | None = 10,
                 **kwargs) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, **kwargs)
        self.dim, self.n_layers, self.n_heads, self.max_len = dim, n_layers, n_heads, max_len
        self.dropout, self.n_negatives, self.temperature = dropout, n_negatives, temperature

    def build(self, data) -> None:
        # Item ids are shifted by one so that 0 is padding.
        self.item_emb = nn.Embedding(data.n_items + 1, self.dim, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.layers = nn.ModuleList(
            HSTULayer(self.dim, self.n_heads, self.max_len, self.dropout) for _ in range(self.n_layers))
        self.norm = nn.LayerNorm(self.dim)
        users = np.flatnonzero(data.history.lengths() >= 2)
        seqs = data.history.sequences(users, self.max_len + 1) + 1
        self.register_buffer("_seqs", torch.as_tensor(seqs), persistent=False)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """Hidden states ``(B, L, dim)`` for padded item sequences ``(B, L)``; position ``t`` sees items ``<= t``."""
        valid = seq > 0
        x = F.dropout(self.item_emb(seq) * self.dim**0.5, self.dropout, self.training) * valid.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, valid)
        return self.norm(x)

    def _items(self) -> torch.Tensor:
        """Output item embeddings, L2-normalized under sampled softmax (as in the reference code)."""
        w = self.item_emb.weight
        return F.normalize(w, dim=-1) if self.n_negatives else w

    def batches(self, data):
        for idx in minibatches(len(self._seqs), self.batch_size, self.device):
            yield self._seqs[idx]

    def loss(self, seq):
        # Fixed shapes (no boolean indexing): variable-size tensors fragment the MPS allocator cache.
        inputs, targets = seq[:, :-1], seq[:, 1:]
        targets = targets.masked_fill(inputs == 0, 0).flatten()
        h = self.encode(inputs).flatten(0, 1)
        if not self.n_negatives:
            return F.cross_entropy(h @ self.item_emb.weight.T, targets, ignore_index=0)
        h, items = F.normalize(h, dim=-1), self._items()
        neg = torch.randint(1, len(items), (self.n_negatives,), device=h.device)
        pos_logit = (h * items[targets]).sum(-1, keepdim=True)
        neg_logit = (h @ items[neg].T).masked_fill(neg[None, :] == targets[:, None], -torch.inf)
        logits = torch.cat([pos_logit, neg_logit], dim=1) / self.temperature
        return F.cross_entropy(logits, torch.zeros_like(targets).masked_fill(targets == 0, -100))

    def score(self, users, history):
        seq = torch.as_tensor(history.sequences(users, self.max_len) + 1, device=self.device)
        h = self.encode(seq)[:, -1]
        if self.n_negatives:
            h = F.normalize(h, dim=-1)
        return h @ self._items()[1:].T
