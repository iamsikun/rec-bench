"""xDeepFM for CTR."""

from __future__ import annotations

import torch
from torch import nn

from recsys.algorithms._ctr import CTRModel, mlp


class XDeepFM(CTRModel):
    """XDeepFM (Lian et al., KDD 2018): a Compressed Interaction Network (CIN) plus linear terms and an MLP.

    CIN layer ``k`` takes the outer products of its input feature maps with
    the field embeddings along each embedding dimension and compresses them with
    ``cin[k]`` 1x1 convolutions; every layer is sum-pooled over the embedding
    dimension and the pooled maps feed the logit.
    """

    first_order = True
    space = {"cin": [(100, 100, 100), (200, 200)], "dropout": [0.1, 0.2], "embedding_reg": [1e-5, 1e-4]}

    def __init__(self, dim: int = 16, cin: tuple[int, ...] = (100, 100, 100),
                 hidden: tuple[int, ...] = (400, 400, 400), dropout: float = 0.2, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self.cin = tuple(cin)
        self.hidden = tuple(hidden)
        self.dropout = dropout

    def network(self, data) -> None:
        m, prev, convs = self.n_fields, self.n_fields, []
        for h in self.cin:
            convs.append(nn.Linear(prev * m, h))  # a 1x1 convolution over feature maps
            prev = h
        self.convs = nn.ModuleList(convs)
        self.cin_out = nn.Linear(sum(self.cin), 1)
        self.mlp = mlp(self.n_fields * self.dim, self.hidden, self.dropout)

    def forward(self, x, h=None):
        x0 = self.embed(x)
        b, d = x0.shape[0], x0.shape[2]
        xk, pooled = x0, []
        for conv in self.convs:
            z = (xk.unsqueeze(2) * x0.unsqueeze(1)).reshape(b, -1, d)
            xk = torch.relu(conv(z.transpose(1, 2))).transpose(1, 2)
            pooled.append(xk.sum(2))
        cin = self.cin_out(torch.cat(pooled, 1)).squeeze(1)
        return self.linear(x) + cin + self.mlp(x0.flatten(1)).squeeze(1)
