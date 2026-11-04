"""Logistic regression on one-hot categorical fields: the CTR floor."""

from __future__ import annotations

from recsys.algorithms._ctr import CTRModel


class LR(CTRModel):
    """LR (Richardson et al., WWW 2007): logistic regression on one-hot encoded fields.

    Logit = bias + sum of one weight per (field, value). No interactions.
    """

    first_order = True
    space = {"lr": [1e-3, 1e-2], "embedding_reg": [0.0, 1e-6, 1e-5]}

    def __init__(self, *, lr: float = 1e-2, **kwargs) -> None:
        super().__init__(dim=1, lr=lr, **kwargs)

    def forward(self, x, h=None):
        return self.linear(x)
