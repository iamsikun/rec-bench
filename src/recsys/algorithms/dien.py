"""Deep Interest Evolution Network for CTR."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms._ctr import CTRModel, Dice, mlp


class AUGRUCell(nn.Module):
    """GRU cell whose update gate is scaled by an attention score: ``u' = a * u``."""

    def __init__(self, d_in: int, d_h: int) -> None:
        super().__init__()
        self.x2h, self.h2h = nn.Linear(d_in, 3 * d_h), nn.Linear(d_h, 3 * d_h)

    def forward(self, gx: torch.Tensor, h: torch.Tensor, a: torch.Tensor | None = None) -> torch.Tensor:
        xr, xz, xn = gx.chunk(3, -1)
        hr, hz, hn = self.h2h(h).chunk(3, -1)
        r, z = torch.sigmoid(xr + hr), torch.sigmoid(xz + hz)
        z = z if a is None else z * a.unsqueeze(-1)
        return (1 - z) * h + z * torch.tanh(xn + r * hn)


class DIEN(CTRModel):
    """DIEN (Zhou et al., AAAI 2019): a GRU extracts interest states, an attention-gated GRU (AUGRU) evolves them.

    The extractor GRU runs over the behavior history; attention ``softmax(h_t W q)`` against the target item gates the
    AUGRU, whose final state joins the field embeddings in an MLP.
    ``aux_weight > 0`` adds the auxiliary loss: ``h_t`` must score the next
    clicked item above a uniformly sampled item.
    """

    requires = frozenset({"behavior"})
    space = {"aux_weight": [0.0, 1.0], "dropout": [0.1, 0.2], "embedding_reg": [1e-5, 1e-4]}

    def __init__(self, dim: int = 16, hidden: tuple[int, ...] = (200, 80), dropout: float = 0.2,
                 dice: bool = False, aux_weight: float = 1.0, max_len: int | None = None, **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.dice = dice
        self.aux_weight = aux_weight
        self.max_len = max_len

    def network(self, data) -> None:
        self.n_items = data.fields["item"]
        self.extractor = nn.GRU(self.dim, self.dim, batch_first=True)
        self.evolver = AUGRUCell(self.dim, self.dim)
        self.att = nn.Linear(self.dim, self.dim, bias=False)
        act = Dice if self.dice else (lambda _: nn.ReLU())
        self.mlp = mlp((self.n_fields + 2) * self.dim, self.hidden, self.dropout, act=act)

    def forward(self, x, h):
        e = self.embed(x)
        q = e[:, self.item_field]
        h = h if self.max_len is None else h[:, -self.max_len:]
        # Left-align the right-aligned history so padding trails: a plain GRU is then exact on valid steps.
        lengths = (h >= 0).sum(1)
        steps = torch.arange(h.shape[1], device=h.device)
        h = torch.gather(h, 1, (steps + h.shape[1] - lengths.unsqueeze(1)).clamp(max=h.shape[1] - 1))
        seq, _ = self.behavior(h)
        mask = steps < lengths.unsqueeze(1)
        n = max(int(lengths.max()), 1)
        seq, mask = seq[:, :n], mask[:, :n]
        states, _ = self.extractor(seq)
        scores = (self.att(states) * q.unsqueeze(1)).sum(-1).masked_fill(~mask, -torch.inf)
        a = torch.nan_to_num(torch.softmax(scores, 1))
        gx, z, out = self.evolver.x2h(states), states.new_zeros(len(x), self.dim), []
        for t in range(n):
            z = self.evolver(gx[:, t], z, a[:, t])
            out.append(z)
        last = (lengths.clamp(max=n) - 1).clamp(min=0)
        interest = torch.stack(out, 1)[torch.arange(len(x), device=x.device), last] * (lengths > 0).unsqueeze(1)
        if self.training and self.aux_weight:
            self._aux = self._aux_loss(states, seq, mask)
        return self.mlp(torch.cat([e.flatten(1), interest, interest * q], 1)).squeeze(1)

    def _aux_loss(self, states: torch.Tensor, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pair = mask[:, :-1] & mask[:, 1:]
        if not pair.any():
            return states.new_zeros(())
        neg = torch.randint(1, max(self.n_items, 2), pair.shape, device=seq.device) + self.item_offset
        hs = states[:, :-1][pair]
        pos = (hs * seq[:, 1:][pair]).sum(-1)
        neg = (hs * self.emb(neg)[pair]).sum(-1)
        logits = torch.cat([pos, neg])
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        return F.binary_cross_entropy_with_logits(logits, labels)

    def loss(self, batch):
        main = super().loss(batch)
        return main + self.aux_weight * self._aux if self.aux_weight else main
