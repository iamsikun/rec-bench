"""FinalNet for CTR."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms._ctr import CTRModel


class FactorizedInteraction(nn.Module):
    """A linear layer split in halves ``h1, h2``; returns ``[h2, h1 * h2]`` : one more multiplicative order."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        if d_out % 2:
            raise ValueError("FINAL block widths must be even")
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h2, h1 = self.linear(x).chunk(2, -1)
        return torch.cat([h2, h1 * h2], -1)


def final_block(width: int, hidden: tuple[int, ...], dropout: float, batch_norm: bool) -> nn.Sequential:
    """FINAL block: an MLP whose linear layers are factorized interaction layers, then a logit layer."""
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [FactorizedInteraction(width, h), *([nn.BatchNorm1d(h)] if batch_norm else []), nn.ReLU(),
                   nn.Dropout(dropout)]
        width = h
    return nn.Sequential(*layers, nn.Linear(width, 1))


class FinalNet(CTRModel):
    """FinalNet (Zhu et al., SIGIR 2023): factorized interaction layers in place of linear layers, one or two blocks.

    Each FINAL block is an MLP whose layers split their output in two halves
    and return ``[h2, h1 * h2]`` (FuxiCTR's ``residual_type="concat"``), so
    the interaction order grows exponentially with depth. ``blocks=2`` adds a
    second block over the plain embeddings while the first sees the
    feature-gated embeddings ``[e, e * g]`` (gates ``g`` linear across fields,
    initialized to one); the prediction averages the two block probabilities,
    and the block-wise self-knowledge-distillation loss adds
    ``BCE(p_i, p.detach())`` for each block ``i`` (as in FuxiCTR). Defaults
    follow FuxiCTR's FinalNet configuration (``2B``, 64-64-64, gating on).
    """

    space = {"blocks": [1, 2], "dropout": [0.0, 0.1], "embedding_reg": [1e-5, 1e-4]}

    def __init__(self, dim: int = 16, blocks: int = 2, hidden1: tuple[int, ...] = (64, 64, 64),
                 hidden2: tuple[int, ...] = (64, 64, 64), dropout: float = 0.0, gating: bool = True,
                 batch_norm: bool = True, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        if blocks not in (1, 2):
            raise ValueError("blocks must be 1 or 2")
        self.blocks = blocks
        self.hidden1, self.hidden2 = tuple(hidden1), tuple(hidden2)
        self.dropout = dropout
        self.gating = gating
        self.batch_norm = batch_norm

    def network(self, data) -> None:
        width = self.n_fields * self.dim
        if self.gating:
            self.gate = nn.Linear(self.n_fields, self.n_fields)
            nn.init.zeros_(self.gate.weight)
            nn.init.ones_(self.gate.bias)
        self.block1 = final_block(width * (2 if self.gating else 1), self.hidden1, self.dropout, self.batch_norm)
        if self.blocks == 2:
            self.block2 = final_block(width, self.hidden2, self.dropout, self.batch_norm)

    def block_logits(self, x: torch.Tensor) -> list[torch.Tensor]:
        e = self.embed(x)
        g = torch.cat([e, e * self.gate(e.transpose(1, 2)).transpose(1, 2)], 1) if self.gating else e
        logits = [self.block1(g.flatten(1)).squeeze(1)]
        if self.blocks == 2:
            logits.append(self.block2(e.flatten(1)).squeeze(1))
        return logits

    @staticmethod
    def ensemble(logits: list[torch.Tensor]) -> torch.Tensor:
        """Logit of the mean block probability, computed in log space."""
        if len(logits) == 1:
            return logits[0]
        z = torch.stack(logits)
        return torch.logsumexp(F.logsigmoid(z), 0) - torch.logsumexp(F.logsigmoid(-z), 0)

    def forward(self, x, h=None):
        return self.ensemble(self.block_logits(x))

    def loss(self, batch):
        x, _, y = batch
        logits = self.block_logits(x)
        z = self.ensemble(logits)
        loss = F.binary_cross_entropy_with_logits(z, y) + self.reg()
        if self.blocks == 2:
            teacher = torch.sigmoid(z).detach()
            loss = loss + sum(F.binary_cross_entropy_with_logits(zi, teacher) for zi in logits)
        return loss

