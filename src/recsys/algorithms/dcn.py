"""Deep & Cross Network for CTR."""

from __future__ import annotations

import torch
from torch import nn

from recsys.algorithms._ctr import CTRModel, mlp


class DCN(CTRModel):
    """DCN (Wang et al., ADKDD 2017): an explicit cross network in parallel with an MLP.

    Cross layer: ``x_{l+1} = x_0 * (x_l . w_l) + b_l + x_l`` over the concatenated
    embeddings ``x_0``; the cross and MLP outputs are concatenated into the logit.
    """

    space = {"cross_layers": [2, 3, 4], "dropout": [0.1, 0.2], "embedding_reg": [1e-5, 1e-4]}

    def __init__(self, dim: int = 16, cross_layers: int = 3, hidden: tuple[int, ...] = (400, 400, 400),
                 dropout: float = 0.2, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self.cross_layers = cross_layers
        self.hidden = tuple(hidden)
        self.dropout = dropout

    def network(self, data) -> None:
        width = self.n_fields * self.dim
        self.w = nn.Parameter(torch.randn(self.cross_layers, width) * width ** -0.5)
        self.b = nn.Parameter(torch.zeros(self.cross_layers, width))
        self.mlp = mlp(width, self.hidden, self.dropout, out=None)
        self.out = nn.Linear(width + (self.hidden[-1] if self.hidden else width), 1)

    def forward(self, x, h=None):
        x0 = self.embed(x).flatten(1)
        xl = x0
        for w, b in zip(self.w, self.b):
            xl = x0 * (xl @ w).unsqueeze(1) + b + xl
        return self.out(torch.cat([xl, self.mlp(x0)], 1)).squeeze(1)
