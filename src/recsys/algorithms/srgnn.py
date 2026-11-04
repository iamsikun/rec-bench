"""SR-GNN (Wu et al., AAAI 2019): a gated graph network over the session graph with attention readout."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms.trainer import TorchRecommender, minibatches


class SRGNN(TorchRecommender):
    """SRGNN (Wu et al., AAAI 2019): gated GNN over the session graph with attention readout.

    The user's most recent ``max_len`` items form the session: unique items
    are nodes, consecutive pairs are directed edges, with in/out adjacency
    normalized by degree. ``steps`` gated propagation steps update the node
    states; the session vector combines the last item's state with a soft
    attention over all positions. Training uses every prefix of every user's
    history (the paper's sequence augmentation) with softmax cross-entropy.
    Graphs are built as batched dense ``(B, L, L)`` adjacency on the device.
    """

    tasks = frozenset({"ranking"})
    requires = frozenset({"timestamps"})
    inductive = True
    space: ClassVar[dict[str, list]] = {"dim": [64, 100], "steps": [1, 2], "max_len": [20, 50]}

    def __init__(self, dim: int = 100, steps: int = 1, max_len: int = 50, *, epochs: int = 30,
                 batch_size: int = 512, lr: float = 1e-3, weight_decay: float = 1e-5,
                 patience: int | None = 5, **kwargs) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                         patience=patience, **kwargs)
        self.dim = dim
        self.steps = steps
        self.max_len = max_len

    def build(self, data) -> None:
        d = self.dim
        self.item_emb = nn.Embedding(data.n_items + 1, d, padding_idx=0)  # 0 = padding
        self.edge_in, self.edge_out = nn.Linear(d, d), nn.Linear(d, d)
        self.cell = nn.GRUCell(2 * d, d)
        self.w1, self.w2 = nn.Linear(d, d, bias=False), nn.Linear(d, d)
        self.q = nn.Linear(d, 1, bias=False)
        self.w3 = nn.Linear(2 * d, d, bias=False)
        for p in self.parameters():  # reference implementation: U(-1/sqrt(d), 1/sqrt(d))
            nn.init.uniform_(p, -1 / math.sqrt(d), 1 / math.sqrt(d))
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        h = data.history
        # One training sample per position t >= 1 of each history: prefix items[start:t] -> items[t].
        ends = np.concatenate([np.arange(s + 1, e) for s, e in zip(h.indptr[:-1], h.indptr[1:])] or [np.zeros(0)])
        starts = np.repeat(h.indptr[:-1], np.maximum(np.diff(h.indptr) - 1, 0))
        self.register_buffer("_items", torch.as_tensor(h.items), persistent=False)
        self.register_buffer("_ends", torch.as_tensor(ends.astype(np.int64)), persistent=False)
        self.register_buffer("_starts", torch.as_tensor(starts.astype(np.int64)), persistent=False)

    def prefixes(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-aligned prefixes ``(B, max_len)`` (shifted ids, 0 = pad) and next-item targets for samples ``idx``."""
        ends, starts = self._ends[idx], self._starts[idx]
        pos = ends[:, None] - self.max_len + torch.arange(self.max_len, device=idx.device)[None]
        seq = torch.where(pos >= starts[:, None], self._items[pos.clamp(min=0)] + 1, 0)
        return seq, self._items[ends]

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """Session embeddings ``(B, dim)`` for right-aligned sequences ``(B, L)``."""
        B, L = seq.shape
        valid = seq > 0
        # alias[b, j]: first position holding the same item as position j; first occurrences are the nodes.
        positions = torch.arange(L, device=seq.device)
        alias = torch.where(seq[:, :, None] == seq[:, None, :], positions, L).amin(2)
        node_item = torch.where(alias == positions, seq, 0)
        edge = valid[:, :-1] & valid[:, 1:]
        # Scatter-max avoids boolean indexing (a host sync per step on GPUs).
        flat = (alias[:, :-1] * L + alias[:, 1:])
        A = torch.zeros(B, L * L, device=seq.device).scatter_reduce(1, flat, edge.float(), "amax").view(B, L, L)
        a_out = A / A.sum(2, keepdim=True).clamp(min=1)
        a_in = A.transpose(1, 2) / A.sum(1)[:, :, None].clamp(min=1)
        h = self.item_emb(node_item)
        for _ in range(self.steps):
            msg = torch.cat([a_in @ self.edge_in(h), a_out @ self.edge_out(h)], -1)
            h = self.cell(msg.flatten(0, 1), h.flatten(0, 1)).view(B, L, -1)
        hs = h.gather(1, alias[:, :, None].expand(-1, -1, self.dim))  # (B, L, d) states per position
        last = hs[:, -1]
        alpha = self.q(torch.sigmoid(self.w1(last)[:, None] + self.w2(hs)))
        glob = (alpha * hs * valid[:, :, None]).sum(1)
        return self.w3(torch.cat([last, glob], -1))

    def batches(self, data):
        for idx in minibatches(len(self._ends), self.batch_size, self.device):
            yield self.prefixes(idx)

    def loss(self, batch):
        seq, target = batch
        return F.cross_entropy(self.encode(seq) @ self.item_emb.weight[1:].T, target)

    def score(self, users, history):
        seq = torch.as_tensor(history.sequences(users, self.max_len) + 1, device=self.device)
        return self.encode(seq) @ self.item_emb.weight[1:].T
