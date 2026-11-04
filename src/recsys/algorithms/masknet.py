"""MaskNet for CTR."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms._ctr import CTRModel, mlp


class MaskBlock(nn.Module):
    """``ReLU(LN(W (mask(e) * v)))`` with the instance-guided mask ``mask(e) = W2 ReLU(W1 e)`` from all embeddings."""

    def __init__(self, emb_width: int, in_width: int, out_width: int, reduction: float, dropout: float) -> None:
        super().__init__()
        self.mask = nn.Sequential(nn.Linear(emb_width, int(in_width * reduction)), nn.ReLU(),
                                  nn.Linear(int(in_width * reduction), in_width))
        self.hidden = nn.Sequential(nn.Linear(in_width, out_width, bias=False), nn.LayerNorm(out_width), nn.ReLU(),
                                    nn.Dropout(dropout))

    def forward(self, e: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.hidden(self.mask(e) * v)


class MaskNet(CTRModel):
    """MaskNet (Wang et al., DLP-KDD 2021): instance-guided masks multiply feature embeddings and hidden layers.

    Every MaskBlock gates its input element-wise with a mask computed from the
    raw flattened embeddings, then applies a layer-normalized linear layer.
    ``structure="serial"`` (SerMaskNet) stacks one block per ``hidden`` width,
    the first acting on the layer-normalized embeddings; ``"parallel"``
    (ParaMaskNet) runs ``blocks`` blocks of width ``block_dim`` on the
    embeddings and feeds their concatenation to an MLP of ``hidden`` widths.
    ``reduction`` is the mask's aggregation-layer ratio. Defaults follow
    FuxiCTR's MaskNet configuration (64-64-64, one parallel block, ratio 1).
    """

    space = {"structure": ["serial", "parallel"], "reduction": [1.0, 2.0], "dropout": [0.0, 0.1],
             "embedding_reg": [1e-5, 1e-4]}

    def __init__(self, dim: int = 16, hidden: tuple[int, ...] = (64, 64, 64), structure: str = "serial",
                 blocks: int = 1, block_dim: int = 64, reduction: float = 1.0, dropout: float = 0.0,
                 **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        if structure not in ("serial", "parallel"):
            raise ValueError(f"unknown structure {structure!r}")
        self.hidden = tuple(hidden)
        self.structure = structure
        self.blocks, self.block_dim = blocks, block_dim
        self.reduction = reduction
        self.dropout = dropout

    def network(self, data) -> None:
        width = self.n_fields * self.dim
        # One LayerNorm per field (as in FuxiCTR): shared normalization, per-field gain and bias.
        self.norm_gain = nn.Parameter(torch.ones(self.n_fields, self.dim))
        self.norm_bias = nn.Parameter(torch.zeros(self.n_fields, self.dim))
        if self.structure == "serial":
            widths = [width, *self.hidden]
            self.mask_blocks = nn.ModuleList(MaskBlock(width, a, b, self.reduction, self.dropout)
                                             for a, b in zip(widths, widths[1:]))
            self.head = nn.Linear(widths[-1], 1)
        else:
            self.mask_blocks = nn.ModuleList(MaskBlock(width, width, self.block_dim, self.reduction, self.dropout)
                                             for _ in range(self.blocks))
            self.head = mlp(self.blocks * self.block_dim, self.hidden, self.dropout)

    def forward(self, x, h=None):
        e = self.embed(x)
        v = (F.layer_norm(e, (self.dim,)) * self.norm_gain + self.norm_bias).flatten(1)
        e = e.flatten(1)
        if self.structure == "serial":
            for block in self.mask_blocks:
                v = block(e, v)
            return self.head(v).squeeze(1)
        return self.head(torch.cat([block(e, v) for block in self.mask_blocks], 1)).squeeze(1)
