"""DeepFM for CTR."""

from __future__ import annotations

from recsys.algorithms._ctr import CTRModel, mlp
from recsys.algorithms.fm import fm_interaction


class DeepFM(CTRModel):
    """DeepFM (Guo et al., IJCAI 2017): factorization machine plus an MLP over shared field embeddings.

    Logit = bias + first-order weights + FM pairwise interactions + MLP(concatenated embeddings).
    """

    first_order = True
    space = {"dropout": [0.0, 0.1, 0.2], "embedding_reg": [1e-6, 1e-5, 1e-4]}

    def __init__(self, dim: int = 16, hidden: tuple[int, ...] = (400, 400, 400), dropout: float = 0.2,
                 **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self.hidden = tuple(hidden)
        self.dropout = dropout

    def network(self, data) -> None:
        self.mlp = mlp(self.n_fields * self.dim, self.hidden, self.dropout)

    def forward(self, x, h=None):
        e = self.embed(x)
        return self.linear(x) + fm_interaction(e) + self.mlp(e.flatten(1)).squeeze(1)
