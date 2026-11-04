"""CL4SRec (Xie et al., ICDE 2022): SASRec plus contrastive learning between augmented views of each sequence."""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms.sasrec import SASRec


def info_nce(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    """NT-Xent over ``2B`` rows: ``a[i]`` and ``b[i]`` are positives, all other rows of the batch are negatives."""
    z = F.normalize(torch.cat([a, b]), dim=1)
    n = len(a)
    eye = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = (z @ z.T / tau).masked_fill(eye, -torch.inf)  # cosine similarity
    labels = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device)
    return F.cross_entropy(sim, labels)


class CL4SRec(SASRec):
    """CL4SRec (Xie et al., ICDE 2022): SASRec with an InfoNCE loss between two randomly augmented views.

    Each training sequence gets two views, each made by one augmentation
    drawn uniformly per sequence: *crop* a random contiguous run of
    ``eta * n`` items, *mask* each item with probability ``gamma`` (a
    learned ``[mask]`` token), or *reorder* (shuffle) a random run of
    ``beta * n`` items. The final hidden states of the two views are
    positives under NT-Xent with cosine similarity, temperature ``tau``
    and the other ``2B - 2`` in-batch views as negatives; the loss is the
    next-item cross-entropy plus ``cl_weight`` times InfoNCE. Augmentation
    ratios follow the reference implementation (DuoRec repository).

    Deviations from the paper: the recommendation loss is SASRec's
    full-softmax CE at every position (the paper uses one-negative BCE),
    so a batch holds whole user sequences; the encoder's feed-forward width
    equals ``dim`` as in our SASRec; cosine similarity replaces the
    reference code's dot product, which on ML-1M kept the cross-entropy at
    the popularity plateau (one sequence per user gives far fewer steps per
    epoch than the reference's prefixes); ``cl_weight = 0.02`` instead of
    0.1, because with full-softmax CE a larger weight lowered ML-1M NDCG@10
    (0.105 at 0.1 vs 0.161 at 0.02, SASRec 0.160).
    """

    space: ClassVar[dict[str, list]] = {
        "cl_weight": [0.01, 0.02, 0.1], "dropout": [0.1, 0.2, 0.5], "tau": [0.2, 0.5, 1.0],
    }

    def __init__(self, dim: int = 64, n_layers: int = 2, n_heads: int = 2, max_len: int = 50,
                 dropout: float = 0.2, cl_weight: float = 0.02, tau: float = 1.0, eta: float = 0.6,
                 gamma: float = 0.3, beta: float = 0.6, **kwargs) -> None:
        super().__init__(dim, n_layers, n_heads, max_len, dropout, **kwargs)
        self.cl_weight, self.tau = cl_weight, tau
        self.eta, self.gamma, self.beta = eta, gamma, beta

    def build(self, data) -> None:
        super().build(data)
        # Token ids: 0 = padding, 1..n_items = items, n_items + 1 = [mask].
        self.mask_id = data.n_items + 1
        self.item_emb = nn.Embedding(data.n_items + 2, self.dim, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def augment(self, seq: torch.Tensor) -> torch.Tensor:
        """One random crop / mask / reorder view per right-aligned ``(B, L)`` row (``0`` = padding)."""
        B, L = seq.shape
        dev = seq.device
        n = (seq > 0).sum(1, keepdim=True)
        pos = torch.arange(L, device=dev)[None]
        first = L - n
        # Crop: a random contiguous run of max(1, eta * n) items, right-aligned.
        c = (self.eta * n).long().clamp(min=1)
        start = first + (torch.rand(B, 1, device=dev) * (n - c + 1)).long()
        crop = torch.where(pos >= L - c, seq.gather(1, (start + pos - (L - c)).clamp(0, L - 1)), 0)
        # Mask: each item becomes [mask] with probability gamma.
        mask = seq.masked_fill((torch.rand(B, L, device=dev) < self.gamma) & (seq > 0), self.mask_id)
        # Reorder: shuffle a random contiguous run of beta * n items (sort keys stay inside the run).
        r = (self.beta * n).long()
        s = first + (torch.rand(B, 1, device=dev) * (n - r + 1)).long()
        inside = (pos >= s) & (pos < s + r)
        key = torch.where(inside, s + torch.rand(B, L, device=dev) * r, pos.float())
        reorder = seq.gather(1, key.argsort(1))
        choice = torch.randint(3, (B, 1), device=dev)
        return torch.where(choice == 0, crop, torch.where(choice == 1, mask, reorder))

    def loss(self, seq):
        inputs, targets = seq[:, :-1], seq[:, 1:]
        targets = targets.masked_fill(inputs == 0, 0)
        logits = self.encode(inputs) @ self.item_emb.weight[: self.mask_id].T
        ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=0)
        views = [self.encode(self.augment(inputs))[:, -1] for _ in range(2)]
        return ce + self.cl_weight * info_nce(*views, self.tau)

    def score(self, users, history):
        seq = torch.as_tensor(history.sequences(users, self.max_len) + 1, device=self.device)
        return self.encode(seq)[:, -1] @ self.item_emb.weight[1 : self.mask_id].T
