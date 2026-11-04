"""DCN-V2 for CTR."""

from __future__ import annotations

import torch
from torch import nn

from recsys.algorithms._ctr import CTRModel, mlp


class CrossLayerV2(nn.Module):
    """``x_{l+1} = x_0 * (W x_l + b) + x_l`` with a full-rank ``W`` or a low-rank ``U V^T``."""

    def __init__(self, width: int, rank: int | None) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width) if rank is None else nn.Sequential(
            nn.Linear(width, rank, bias=False), nn.Linear(rank, width))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.proj(xl) + xl


class DCNv2(CTRModel):
    """DCNv2 (Wang et al., WWW 2021): a matrix-valued cross network, stacked on or parallel to an MLP.

    ``structure`` is ``"parallel"`` (cross and MLP side by side, outputs
    concatenated), ``"stacked"`` (MLP on top of the cross output) or
    ``"crossnet"`` (cross network only). ``low_rank`` sets the rank of the
    factorized cross matrix (``None`` = full rank).
    """

    space = {"structure": ["parallel", "stacked"], "low_rank": [None, 32], "dropout": [0.1, 0.2],
             "embedding_reg": [1e-5, 1e-4]}

    def __init__(self, dim: int = 16, cross_layers: int = 3, hidden: tuple[int, ...] = (400, 400, 400),
                 dropout: float = 0.2, structure: str = "parallel", low_rank: int | None = None,
                 **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        if structure not in ("parallel", "stacked", "crossnet"):
            raise ValueError(f"unknown structure {structure!r}")
        self.cross_layers = cross_layers
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.structure = structure
        self.low_rank = low_rank

    def network(self, data) -> None:
        width = self.n_fields * self.dim
        self.cross = nn.ModuleList(CrossLayerV2(width, self.low_rank) for _ in range(self.cross_layers))
        hidden = self.hidden if self.structure != "crossnet" else ()
        self.mlp = mlp(width, hidden, self.dropout, out=None)
        deep = hidden[-1] if hidden else width
        out = {"parallel": width + deep, "stacked": deep, "crossnet": width}[self.structure]
        self.out = nn.Linear(out, 1)

    def forward(self, x, h=None):
        x0 = self.embed(x).flatten(1)
        xl = x0
        for layer in self.cross:
            xl = layer(x0, xl)
        if self.structure == "parallel":
            z = torch.cat([xl, self.mlp(x0)], 1)
        else:
            z = self.mlp(xl)
        return self.out(z).squeeze(1)
