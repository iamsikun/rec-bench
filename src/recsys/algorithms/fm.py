"""Factorization machine for CTR."""

from __future__ import annotations

import torch

from recsys.algorithms._ctr import CTRModel


def fm_interaction(e: torch.Tensor) -> torch.Tensor:
    """Sum of pairwise inner products of ``(B, F, D)`` field embeddings, in O(F * D)."""
    return 0.5 * (e.sum(1).square() - e.square().sum(1)).sum(1)


class FM(CTRModel):
    """FM (Rendle, ICDM 2010): linear terms plus factorized pairwise interactions of all fields.

    Logit = bias + sum_i w_i + sum_{i<j} <v_i, v_j>.
    """

    first_order = True
    space = {"dim": [16, 32], "embedding_reg": [0.0, 1e-5, 1e-4]}

    def forward(self, x, h=None):
        return self.linear(x) + fm_interaction(self.embed(x))
