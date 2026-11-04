"""Search-based Interest Model for CTR, and the two-stage retrieval base shared with ETA."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms._ctr import CTRModel, mlp


class TargetAttention(nn.Module):
    """Multi-head scaled dot-product attention of the target over behaviors (FuxiCTR's MultiHeadTargetAttention)."""

    def __init__(self, dim: int, att_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if att_dim % heads:
            raise ValueError("att_dim must be divisible by heads")
        self.heads = heads
        self.q = nn.Linear(dim, att_dim, bias=False)
        self.kv = nn.Linear(dim, 2 * att_dim, bias=False)
        self.out = nn.Linear(att_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = keys.shape
        qh = self.q(q).reshape(b, self.heads, 1, -1)
        k, v = (t.reshape(b, n, self.heads, -1).transpose(1, 2) for t in self.kv(keys).chunk(2, -1))
        s = (qh @ k.transpose(-1, -2) / math.sqrt(qh.shape[-1])).masked_fill(~mask[:, None, None], -torch.inf)
        a = torch.nan_to_num(torch.softmax(s, -1))  # no valid behavior -> zero interest
        return self.out((self.drop(a) @ v).reshape(b, -1))


class TwoStageInterest(CTRModel):
    """Long-term interest from the ``top_k`` behaviors that :meth:`search` ranks highest, plus a short-term interest.

    Both interests are multi-head target attention of the target item over
    behaviors: the short-term one over the ``short_len`` most recent, the
    long-term one over the retrieved top-k. They join all field embeddings in
    an MLP. Subclasses define the search unit.
    """

    requires = frozenset({"behavior"})

    def __init__(self, dim: int = 16, hidden: tuple[int, ...] = (512, 128, 64), att_dim: int = 64, heads: int = 1,
                 top_k: int = 20, short_len: int = 10, dropout: float = 0.0, att_dropout: float = 0.0,
                 **kwargs) -> None:
        super().__init__(dim=dim, **kwargs)
        self.hidden = tuple(hidden)
        self.att_dim, self.heads = att_dim, heads
        self.top_k, self.short_len = top_k, short_len
        self.dropout, self.att_dropout = dropout, att_dropout

    def network(self, data) -> None:
        self.short_att = TargetAttention(self.dim, self.att_dim, self.heads, self.att_dropout)
        self.long_att = TargetAttention(self.dim, self.att_dim, self.heads, self.att_dropout)
        self.mlp = mlp((self.n_fields + 2) * self.dim, self.hidden, self.dropout)

    def search(self, q: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Relevance ``(B, L)`` of each behavior to the target ``q``; higher is retrieved first."""
        raise NotImplementedError

    def forward(self, x, h):
        e = self.embed(x)
        q = e[:, self.item_field]
        keys, mask = self.behavior(h)
        short = self.short_att(q, keys[:, -self.short_len:], mask[:, -self.short_len:])
        scores = self.search(q, keys, mask).masked_fill(~mask, -torch.inf)
        idx = scores.topk(min(self.top_k, keys.shape[1]), 1).indices
        top = torch.gather(keys, 1, idx.unsqueeze(-1).expand(-1, -1, self.dim))
        long = self.long_att(q, top, torch.gather(mask, 1, idx))
        return self.mlp(torch.cat([e.flatten(1), short, long], 1)).squeeze(1)


class SIM(TwoStageInterest):
    """SIM (Pi et al., CIKM 2020): a General Search Unit retrieves relevant long-term behaviors for exact attention.

    The General Search Unit uses soft search: relevance is the inner product
    ``(W_b e_i) . (W_a e_target)``, trained by the paper's auxiliary CTR task
    on ``[sum_i r_i e_i, e_target]`` (weight ``gsu_weight``; ``r`` is a
    softmax over valid behaviors here). The Exact Search Unit is multi-head
    target attention over the ``top_k`` retrieved behaviors, and a short-term
    branch attends over the most recent ``short_len`` behaviors. Deviations:
    hard search (by category) is not implemented because our histories carry
    item ids only; no time-interval embeddings (histories have no
    timestamps); and since histories hold at most 50 items, ``top_k=20`` and
    ``short_len=10`` replace the paper's thousands-long setting, which limits
    the gain from retrieval. MLP and attention sizes follow FuxiCTR's ETA.
    """

    space = {"top_k": [10, 20], "gsu_weight": [0.1, 1.0], "dropout": [0.0, 0.1], "embedding_reg": [1e-5, 1e-4]}

    def __init__(self, gsu_weight: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gsu_weight = gsu_weight

    def network(self, data) -> None:
        super().network(data)
        self.wa = nn.Linear(self.dim, self.dim, bias=False)
        self.wb = nn.Linear(self.dim, self.dim, bias=False)
        self.gsu_mlp = mlp(2 * self.dim, self.hidden, self.dropout)

    def search(self, q, keys, mask):
        r = (self.wb(keys) * self.wa(q).unsqueeze(1)).sum(-1)
        if self.training and self.gsu_weight:
            w = torch.nan_to_num(torch.softmax(r.masked_fill(~mask, -torch.inf), 1))
            self._gsu = self.gsu_mlp(torch.cat([(w.unsqueeze(-1) * keys).sum(1), q], 1)).squeeze(1)
        return r

    def loss(self, batch):
        loss = super().loss(batch)
        if self.gsu_weight:
            loss = loss + self.gsu_weight * F.binary_cross_entropy_with_logits(self._gsu, batch[2])
        return loss
