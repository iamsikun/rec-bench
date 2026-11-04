"""PureSVD (Cremonesi et al., RecSys 2010): truncated SVD of the zero-filled interaction matrix."""

from __future__ import annotations

import torch

from recsys.algorithms.base import Recommender
from recsys.algorithms.itemknn import gram, history_rows
from recsys.device import resolve_device


class PureSVD(Recommender):
    """PureSVD (Cremonesi et al., RecSys 2010): project the history onto the top-k right singular vectors.

    ``X ~ U S V^T`` with ``V`` the top ``factors`` right singular vectors of the
    binary user-item matrix; scores are ``x_u V V^T`` (folding-in, so unseen
    users can be scored). ``V`` is obtained with ``torch.svd_lowrank`` on the
    item Gram matrix ``X^T X`` (same right singular vectors, faster spectral
    decay), so memory is ``O(n_items^2)`` and nothing user-sized is densified.
    """

    tasks = frozenset({"ranking"})
    inductive = True
    space = {"factors": [20, 50, 100, 200]}

    def __init__(self, factors: int = 50, oversample: int = 20, niter: int = 6, device: str = "auto") -> None:
        self.factors = factors
        self.oversample = oversample
        self.niter = niter
        self.device = resolve_device(device)

    def fit(self, data) -> None:
        G = gram(data.history.matrix(), self.device)
        q = min(self.factors + self.oversample, G.shape[0])
        # QR has no MPS kernel, so on Apple GPUs the randomized SVD runs on the CPU.
        _, _, V = torch.svd_lowrank(G.cpu() if G.device.type == "mps" else G, q=q, niter=self.niter)
        self.V = V[:, : self.factors].contiguous().to(self.device)

    def score(self, users, history):
        return (history_rows(history, users, self.device) @ self.V) @ self.V.T
