"""Behavior Sequence Transformer for CTR."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms._ctr import CTRModel, mlp


class BST(CTRModel):
    """BST (Chen et al., DLP-KDD 2019): a Transformer over the behavior sequence plus the target item.

    The history and the target item (last position) get learned position
    embeddings and pass through ``layers`` Transformer encoder layers
    (post-norm, LeakyReLU feed-forward of width ``dim``) with padding masked.
    ``pooling="mean"`` averages the valid outputs (FuxiCTR's default);
    ``"concat"`` concatenates all positions as in the paper. The result joins
    all field embeddings in an MLP. Defaults follow FuxiCTR's BST
    configuration (2 heads, 1 layer, MLP 256-128-64). Deviation: positions are
    indices from the most recent behavior, not the paper's time-interval
    buckets, because histories carry no timestamps.
    """

    requires = frozenset({"behavior"})
    space = {"layers": [1, 2], "pooling": ["mean", "concat"], "dropout": [0.0, 0.1], "embedding_reg": [1e-5, 1e-4]}

    def __init__(self, dim: int = 16, hidden: tuple[int, ...] = (256, 128, 64), heads: int = 2, layers: int = 1,
                 pooling: str = "mean", dropout: float = 0.0, att_dropout: float = 0.0, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        if pooling not in ("mean", "concat"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.hidden = tuple(hidden)
        self.heads, self.layers = heads, layers
        self.pooling = pooling
        self.dropout, self.att_dropout = dropout, att_dropout

    def network(self, data) -> None:
        length = data.train["history"].dtype.size + 1
        self.pos = nn.Parameter(torch.randn(length, self.dim) * self.init_std)
        layer = nn.TransformerEncoderLayer(self.dim, self.heads, self.dim, self.att_dropout, F.leaky_relu,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, self.layers, enable_nested_tensor=False)
        pooled = self.dim * (length if self.pooling == "concat" else 1)
        self.mlp = mlp(self.n_fields * self.dim + pooled, self.hidden, self.dropout)

    def forward(self, x, h):
        e = self.embed(x)
        keys, mask = self.behavior(h)
        valid = torch.cat([mask, mask.new_ones(len(x), 1)], 1)
        out = self.encoder(torch.cat([keys, e[:, self.item_field, None]], 1) + self.pos, src_key_padding_mask=~valid)
        out = out * valid.unsqueeze(-1)
        pooled = out.flatten(1) if self.pooling == "concat" else out.sum(1) / valid.sum(1, keepdim=True)
        return self.mlp(torch.cat([e.flatten(1), pooled], 1)).squeeze(1)
