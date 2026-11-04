"""FinalMLP for CTR."""

from __future__ import annotations

import torch
from torch import nn

from recsys.algorithms._ctr import CTRModel, mlp


class FeatureSelection(nn.Module):
    """Gate ``2 * sigmoid(MLP(context))`` over the flattened embeddings; empty context = a learned vector."""

    def __init__(self, fields: list[int], dim: int, width: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        self.fields = fields
        self.bias = nn.Parameter(torch.zeros(1, dim)) if not fields else None
        self.gate = mlp(max(len(fields), 1) * dim, hidden, out=width)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        ctx = e[:, self.fields].flatten(1) if self.fields else self.bias.expand(e.shape[0], -1)
        return e.flatten(1) * 2 * torch.sigmoid(self.gate(ctx))


class Aggregation(nn.Module):
    """Multi-head bilinear fusion: ``w_x^T x + w_y^T y + sum_h x_h^T W_h y_h``."""

    def __init__(self, dx: int, dy: int, heads: int) -> None:
        super().__init__()
        if dx % heads or dy % heads:
            raise ValueError("stream widths must be divisible by heads")
        self.heads = heads
        self.wx, self.wy = nn.Linear(dx, 1), nn.Linear(dy, 1, bias=False)
        self.w = nn.Parameter(torch.zeros(heads, dx // heads, dy // heads))  # zero init: no logit blow-up at start

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xh, yh = x.reshape(len(x), self.heads, -1), y.reshape(len(y), self.heads, -1)
        bilinear = torch.einsum("bhi,hij,bhj->b", xh, self.w, yh)
        return self.wx(x).squeeze(1) + self.wy(y).squeeze(1) + bilinear


class FinalMLP(CTRModel):
    """FinalMLP (Mao et al., AAAI 2023): two MLP streams with stream-specific feature gating and bilinear fusion.

    Each stream sees the embeddings re-weighted by a gate computed from its
    context fields (``fs1_context`` / ``fs2_context``; empty = a learned
    vector), and the two stream outputs are fused by multi-head bilinear
    aggregation. Defaults follow the BARS Criteo configuration.
    """

    space = {"heads": [1, 10], "dropout": [0.1, 0.2], "embedding_reg": [1e-5, 1e-4]}

    def __init__(self, dim: int = 16, mlp1: tuple[int, ...] = (400, 400, 400), mlp2: tuple[int, ...] = (800,),
                 fs_hidden: tuple[int, ...] = (800,), fs1_context: tuple[str, ...] = (),
                 fs2_context: tuple[str, ...] = (), heads: int = 10, dropout: float = 0.2,
                 batch_norm: bool = True, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self.mlp1, self.mlp2, self.fs_hidden = tuple(mlp1), tuple(mlp2), tuple(fs_hidden)
        self.fs1_context, self.fs2_context = tuple(fs1_context), tuple(fs2_context)
        self.heads, self.dropout, self.batch_norm = heads, dropout, batch_norm

    def network(self, data) -> None:
        width = self.n_fields * self.dim
        ctx = lambda names: [self.field_names.index(n) for n in names]  # noqa: E731
        self.fs1 = FeatureSelection(ctx(self.fs1_context), self.dim, width, self.fs_hidden)
        self.fs2 = FeatureSelection(ctx(self.fs2_context), self.dim, width, self.fs_hidden)
        self.stream1 = mlp(width, self.mlp1, self.dropout, out=None, batch_norm=self.batch_norm)
        self.stream2 = mlp(width, self.mlp2, self.dropout, out=None, batch_norm=self.batch_norm)
        self.fusion = Aggregation(self.mlp1[-1], self.mlp2[-1], self.heads)

    def forward(self, x, h=None):
        e = self.embed(x)
        return self.fusion(self.stream1(self.fs1(e)), self.stream2(self.fs2(e)))
