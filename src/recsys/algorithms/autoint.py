"""AutoInt / AutoInt+ for CTR."""

from __future__ import annotations

import torch
from torch import nn

from recsys.algorithms._ctr import CTRModel, mlp


class InteractingLayer(nn.Module):
    """Multi-head self-attention across fields with a projected residual connection and ReLU."""

    def __init__(self, width: int, att_dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * att_dim, bias=False)
        self.res = nn.Linear(width, att_dim, bias=False)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        b, f, _ = e.shape
        q, k, v = (t.reshape(b, f, self.heads, -1).transpose(1, 2) for t in self.qkv(e).chunk(3, -1))
        att = torch.softmax(q @ k.transpose(-1, -2), -1)
        out = (att @ v).transpose(1, 2).reshape(b, f, -1)
        return torch.relu(out + self.res(e))


class AutoInt(CTRModel):
    """AutoInt (Song et al., CIKM 2019): stacked multi-head self-attention learns high-order field interactions.

    ``dnn=True`` gives AutoInt+, which adds an MLP over the concatenated
    embeddings to the attention logit. ``att_dim`` is the total width over
    all ``heads``.
    """

    space = {"layers": [2, 3], "heads": [1, 2], "dnn": [False, True], "embedding_reg": [1e-5, 1e-4]}

    def __init__(self, dim: int = 16, layers: int = 3, heads: int = 2, att_dim: int = 64, dnn: bool = False,
                 hidden: tuple[int, ...] = (400, 400, 400), dropout: float = 0.2, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        if att_dim % heads:
            raise ValueError("att_dim must be divisible by heads")
        self.layers, self.heads, self.att_dim = layers, heads, att_dim
        self.dnn = dnn
        self.hidden = tuple(hidden)
        self.dropout = dropout

    def network(self, data) -> None:
        widths = [self.dim] + [self.att_dim] * self.layers
        self.attention = nn.Sequential(*(InteractingLayer(w, self.att_dim, self.heads) for w in widths[:-1]))
        self.out = nn.Linear(self.n_fields * widths[-1], 1)
        self.mlp = mlp(self.n_fields * self.dim, self.hidden, self.dropout) if self.dnn else None

    def forward(self, x, h=None):
        e = self.embed(x)
        logit = self.out(self.attention(e).flatten(1)).squeeze(1)
        if self.mlp is not None:
            logit = logit + self.mlp(e.flatten(1)).squeeze(1)
        return logit
