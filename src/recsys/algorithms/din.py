"""Deep Interest Network for CTR."""

from __future__ import annotations

import torch
from torch import nn

from recsys.algorithms._ctr import CTRModel, Dice, mlp


class DIN(CTRModel):
    """DIN (Zhou et al., KDD 2018): target-aware attention pools the user's behavior history.

    Each history item ``k`` gets weight ``MLP([q, k, q - k, q * k])`` against
    the target item ``q`` (no softmax, as in the paper); the weighted sum is
    concatenated with all field embeddings and fed to an MLP. History items
    share the ``item`` field's embedding table. ``dice=True`` uses the Dice
    activation instead of ReLU; ``max_len`` keeps only the most recent items.
    """

    requires = frozenset({"behavior"})
    space = {"dice": [False, True], "dropout": [0.1, 0.2], "embedding_reg": [1e-5, 1e-4]}

    def __init__(self, dim: int = 16, hidden: tuple[int, ...] = (200, 80), att_hidden: tuple[int, ...] = (80, 40),
                 dropout: float = 0.2, dice: bool = False, max_len: int | None = None, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self.hidden, self.att_hidden = tuple(hidden), tuple(att_hidden)
        self.dropout = dropout
        self.dice = dice
        self.max_len = max_len

    def network(self, data) -> None:
        act = Dice if self.dice else (lambda _: nn.ReLU())
        self.att = mlp(4 * self.dim, self.att_hidden, act=act)
        self.mlp = mlp((self.n_fields + 1) * self.dim, self.hidden, self.dropout, act=act)

    def forward(self, x, h):
        e = self.embed(x)
        q = e[:, self.item_field]
        keys, mask = self.behavior(h if self.max_len is None else h[:, -self.max_len:])
        qe = q.unsqueeze(1).expand_as(keys)
        w = self.att(torch.cat([qe, keys, qe - keys, qe * keys], -1)).squeeze(-1) * mask
        interest = (w.unsqueeze(-1) * keys).sum(1)
        return self.mlp(torch.cat([e.flatten(1), interest], 1)).squeeze(1)
