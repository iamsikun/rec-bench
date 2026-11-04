"""Wide & Deep for CTR."""

from __future__ import annotations

from recsys.algorithms._ctr import CTRModel, mlp


class WideDeep(CTRModel):
    """WideDeep (Cheng et al., DLRS@RecSys 2016): a wide linear model jointly trained with a deep MLP.

    Logit = bias + first-order weights on one-hot fields (wide) + MLP(concatenated embeddings) (deep).
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
        return self.linear(x) + self.mlp(self.embed(x).flatten(1)).squeeze(1)
