"""Gated Deep Cross Network for CTR."""

from __future__ import annotations

import torch
from torch import nn

from recsys.algorithms._ctr import CTRModel, mlp


class GatedCrossLayer(nn.Module):
    """``c_{l+1} = c_0 * (W c_l + b) * sigmoid(W_g c_l) + c_l``: a DCN-V2 cross whose terms pass an information gate."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width)
        self.gate = nn.Linear(width, width, bias=False)

    def forward(self, c0: torch.Tensor, cl: torch.Tensor) -> torch.Tensor:
        return c0 * self.proj(cl) * torch.sigmoid(self.gate(cl)) + cl


class GDCN(CTRModel):
    """GDCN (Wang et al., CIKM 2023): a gated cross network filters noisy high-order crosses, beside or under an MLP.

    Each cross layer multiplies the DCN-V2 cross term by a learned sigmoid
    information gate. ``structure`` is ``"parallel"`` (GDCN-P: cross network
    and MLP side by side, outputs concatenated) or ``"stacked"`` (GDCN-S: the
    MLP on top of the cross output). Defaults follow the official code and
    FuxiCTR (3 cross layers, MLP 400-400-400). Not implemented: the paper's
    field-level dimension optimization, a separate post-hoc procedure that
    shrinks each field's embedding size by PCA of a trained model's
    embeddings; every field here uses ``dim``.
    """

    space = {"structure": ["parallel", "stacked"], "cross_layers": [3, 5], "dropout": [0.1, 0.2],
             "embedding_reg": [1e-5, 1e-4]}

    def __init__(self, dim: int = 16, cross_layers: int = 3, hidden: tuple[int, ...] = (400, 400, 400),
                 dropout: float = 0.2, structure: str = "parallel", **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        if structure not in ("parallel", "stacked"):
            raise ValueError(f"unknown structure {structure!r}")
        self.cross_layers = cross_layers
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.structure = structure

    def network(self, data) -> None:
        width = self.n_fields * self.dim
        self.cross = nn.ModuleList(GatedCrossLayer(width) for _ in range(self.cross_layers))
        self.mlp = mlp(width, self.hidden, self.dropout, out=None)
        deep = self.hidden[-1] if self.hidden else width
        self.out = nn.Linear(width + deep if self.structure == "parallel" else deep, 1)

    def forward(self, x, h=None):
        c0 = self.embed(x).flatten(1)
        cl = c0
        for layer in self.cross:
            cl = layer(c0, cl)
        z = torch.cat([cl, self.mlp(c0)], 1) if self.structure == "parallel" else self.mlp(cl)
        return self.out(z).squeeze(1)
