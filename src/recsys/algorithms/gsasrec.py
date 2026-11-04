"""gSASRec (Petrov & Macdonald, RecSys 2023): SASRec trained with generalized BCE and many negatives."""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms.sasrec import SASRec


class GSASRec(SASRec):
    """GSASRec (Petrov & Macdonald, RecSys 2023): SASRec with ``k`` sampled negatives per position and the gBCE loss.

    gBCE raises the positive's sigmoid to the power ``beta = alpha * (t * (1 - 1/alpha) + 1/alpha)``,
    with sampling rate ``alpha = k / (n_items - 1)``, which counters the
    overconfidence that negative sampling induces; ``t = 0`` is plain BCE and
    ``t = 1`` recovers full-softmax calibration. Negatives are drawn uniformly
    per sequence and shared across its positions. As in the reference code,
    output item embeddings are separate from input ones unless ``tie_embeddings``.
    Encoder and scoring are SASRec's.
    """

    requires = frozenset({"timestamps"})
    space: ClassVar[dict[str, list]] = {"num_negatives": [64, 256], "t": [0.5, 0.75, 1.0], "lr": [1e-3, 3e-3]}

    def __init__(self, dim: int = 64, n_layers: int = 2, n_heads: int = 1, max_len: int = 50,
                 dropout: float = 0.2, num_negatives: int = 256, t: float = 0.75, tie_embeddings: bool = False,
                 **kwargs) -> None:
        # Many uniform negatives make early training slow (a popularity plateau), so allow a longer patience.
        super().__init__(dim, n_layers, n_heads, max_len, dropout, **{"patience": 30, **kwargs})
        self.num_negatives = num_negatives
        self.t = t
        self.tie_embeddings = tie_embeddings

    def build(self, data) -> None:
        super().build(data)
        self.n_items = data.n_items
        self.k = max(1, min(self.num_negatives, data.n_items - 1))  # alpha must stay <= 1
        alpha = self.k / max(data.n_items - 1, 1)
        self.beta = alpha * (self.t * (1 - 1 / alpha) + 1 / alpha)
        self.out_emb = self.item_emb if self.tie_embeddings else nn.Embedding(data.n_items + 1, self.dim)
        if not self.tie_embeddings:
            nn.init.normal_(self.out_emb.weight, std=0.02)

    def loss(self, seq):
        inputs, targets = seq[:, :-1], seq[:, 1:]
        valid = (inputs > 0) & (targets > 0)
        h = self.encode(inputs)  # (B, L, d)
        pos = (h * self.out_emb(targets)).sum(-1)  # (B, L)
        neg_ids = torch.randint(1, self.n_items + 1, (len(seq), self.k), device=seq.device)
        neg = torch.einsum("bld,bkd->blk", h, self.out_emb(neg_ids))  # (B, L, k)
        # BCE with the positive probability raised to beta: -beta * log sigma(s+) - sum log(1 - sigma(s-)).
        per_pos = -self.beta * F.logsigmoid(pos) - F.logsigmoid(-neg).sum(-1)
        return (per_pos[valid] / (self.k + 1)).mean()

    def score(self, users, history):
        seq = torch.as_tensor(history.sequences(users, self.max_len) + 1, device=self.device)
        return self.encode(seq)[:, -1] @ self.out_emb.weight[1:].T
