"""DuoRec (Qiu et al., WSDM 2022): SASRec regularized by dropout views and same-target sequences."""

from __future__ import annotations

from typing import ClassVar

import torch

from recsys.algorithms.cl4srec import info_nce
from recsys.algorithms.sasrec import SASRec
from recsys.algorithms.trainer import minibatches


class DuoRec(SASRec):
    """DuoRec (Qiu et al., WSDM 2022): contrastive regularization with model-level and supervised positives.

    For each training sequence a random prefix (ending at position ``t``)
    is the anchor. Its unsupervised view is the same sequence encoded again
    under a fresh dropout mask; its supervised positive is a randomly drawn
    training prefix (from any user) whose next item is the same target.
    As in the reference code's ``us_x`` setting, the dropout view and the
    supervised view are contrasted with NT-Xent (dot product, temperature
    ``tau``, in-batch negatives) and added to the next-item cross-entropy
    with weight ``cl_weight``.

    Deviations from the paper: the recommendation loss is SASRec's
    full-softmax CE at every position of the user's sequence rather than at
    the last position of every prefix; the encoder's feed-forward width
    equals ``dim`` as in our SASRec.
    """

    space: ClassVar[dict[str, list]] = {"cl_weight": [0.05, 0.1, 0.3], "dropout": [0.1, 0.2, 0.5], "tau": [0.5, 1.0]}

    def __init__(self, dim: int = 64, n_layers: int = 2, n_heads: int = 2, max_len: int = 50,
                 dropout: float = 0.2, cl_weight: float = 0.1, tau: float = 1.0, **kwargs) -> None:
        super().__init__(dim, n_layers, n_heads, max_len, dropout, **kwargs)
        self.cl_weight, self.tau = cl_weight, tau

    def build(self, data) -> None:
        super().build(data)
        # Every (row, position) with a next item, flattened and grouped by that target item.
        inputs, targets = self._seqs[:, :-1].flatten(), self._seqs[:, 1:].flatten()
        flat = torch.nonzero(inputs > 0).squeeze(1)
        order = torch.argsort(targets[flat], stable=True)
        first = torch.searchsorted(targets[flat][order], torch.arange(data.n_items + 2))
        self.register_buffer("_by_target", flat[order], persistent=False)
        self.register_buffer("_first", first, persistent=False)

    def batches(self, data):
        L = self.max_len
        for idx in minibatches(len(self._seqs), self.batch_size, self.device):
            seq = self._seqs[idx]
            n = (seq[:, :-1] > 0).sum(1)
            t = L - n + (torch.rand(len(idx), device=self.device) * n).long()  # a random prefix end per row
            target = seq[torch.arange(len(idx), device=self.device), t + 1]
            lo, hi = self._first[target], self._first[target + 1]
            other = self._by_target[lo + (torch.rand(len(idx), device=self.device) * (hi - lo)).long()]
            yield seq, t, other

    def loss(self, batch):
        seq, t, other = batch
        rows = torch.arange(len(seq), device=seq.device)
        L = self.max_len
        aug = self.encode(seq[:, :-1])[rows, t]  # second pass: a different dropout mask
        sem = self.encode(self._seqs[other // L, :-1])[rows, other % L]
        return super().loss(seq) + self.cl_weight * info_nce(aug, sem, self.tau)
