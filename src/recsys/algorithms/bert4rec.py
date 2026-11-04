"""BERT4Rec (Sun et al., CIKM 2019): a bidirectional Transformer trained with a cloze objective."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms.trainer import TorchRecommender, minibatches


class BERT4Rec(TorchRecommender):
    """BERT4Rec (Sun et al., CIKM 2019): bidirectional self-attention trained to fill masked items.

    Each training step masks every item of the user's last ``max_len`` items
    with probability ``mask_prob`` (at least one per sequence) and applies
    full-softmax cross-entropy at the masked positions. To score, the last
    ``max_len - 1`` known items are followed by ``[MASK]`` and the masked
    position's output ranks the catalog. Positions are counted from the end
    of the sequence and padding is excluded from attention, so the output
    does not depend on how much left padding a sequence carries.
    """

    tasks = frozenset({"ranking"})
    requires = frozenset({"timestamps"})
    inductive = True
    space: ClassVar[dict[str, list]] = {"mask_prob": [0.1, 0.2, 0.3], "dropout": [0.1, 0.2], "max_len": [50, 200]}

    def __init__(self, dim: int = 64, n_layers: int = 2, n_heads: int = 2, max_len: int = 200,
                 dropout: float = 0.1, mask_prob: float = 0.2, *, epochs: int = 400, batch_size: int = 128,
                 lr: float = 1e-3, patience: int | None = 20, eval_every: int = 2, **kwargs) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
                         eval_every=eval_every, **kwargs)
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_len = max_len
        self.dropout = dropout
        self.mask_prob = mask_prob

    def build(self, data) -> None:
        # Token ids: 0 = padding, 1..n_items = items, n_items + 1 = [MASK].
        self.n_items = data.n_items
        self.mask_id = data.n_items + 1
        self.item_emb = nn.Embedding(data.n_items + 2, self.dim, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_len, self.dim)
        for emb in (self.item_emb, self.pos_emb):
            nn.init.normal_(emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        layer = nn.TransformerEncoderLayer(
            self.dim, self.n_heads, dim_feedforward=4 * self.dim, dropout=self.dropout,
            activation="gelu", batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, self.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(self.dim)
        self.drop = nn.Dropout(self.dropout)
        self.head = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.LayerNorm(self.dim))
        self.out_bias = nn.Parameter(torch.zeros(data.n_items))
        users = np.flatnonzero(data.history.lengths() >= 2)
        seqs = data.history.sequences(users, self.max_len) + 1
        self.register_buffer("_seqs", torch.as_tensor(seqs), persistent=False)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """Hidden states ``(B, L, dim)`` for right-aligned token sequences ``(B, L)``, ``L <= max_len``."""
        L = seq.shape[1]
        pos = torch.arange(L - 1, -1, -1, device=seq.device)  # distance from the end
        x = self.drop(self.norm(self.item_emb(seq) + self.pos_emb(pos)))
        return self.head(self.encoder(x, src_key_padding_mask=seq == 0))

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.item_emb.weight[1 : self.n_items + 1].T + self.out_bias

    def batches(self, data):
        for idx in minibatches(len(self._seqs), self.batch_size, self.device):
            seq = self._seqs[idx]
            valid = seq > 0
            masked = (torch.rand(seq.shape, device=self.device) < self.mask_prob) & valid
            # Guarantee one masked item per sequence: mask the last item where none was drawn.
            masked[:, -1] |= ~masked.any(1)
            yield seq.masked_fill(masked, self.mask_id), seq, masked

    def loss(self, batch):
        inputs, targets, masked = batch
        h = self.encode(inputs)[masked]
        return F.cross_entropy(self.logits(h), targets[masked] - 1)

    def score(self, users, history):
        seq = torch.as_tensor(history.sequences(users, self.max_len - 1) + 1, device=self.device)
        mask = torch.full((len(seq), 1), self.mask_id, device=self.device, dtype=seq.dtype)
        return self.logits(self.encode(torch.cat([seq, mask], 1))[:, -1])
