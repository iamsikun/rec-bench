"""GF-CF (Shen et al., CIKM 2021): training-free graph filters on the normalized item-item graph."""

from __future__ import annotations

import numpy as np
import torch

from recsys.algorithms.base import Recommender
from recsys.algorithms.itemknn import gram, history_rows
from recsys.device import resolve_device


def inv_sqrt(deg: np.ndarray) -> np.ndarray:
    """``deg ** -0.5`` with ``0`` for zero degrees."""
    return np.power(deg, -0.5, where=deg > 0, out=np.zeros_like(deg, dtype=np.float64))


class GFCF(Recommender):
    """GF-CF (Shen et al., CIKM 2021): linear low-pass filter plus an ideal low-pass filter, no training.

    With ``R~ = D_U^-1/2 R D_I^-1/2`` the degree-normalized interaction
    matrix, a user's scores are ``r_u (R~^T R~ + alpha D_I^-1/2 V V^T D_I^1/2)``,
    where ``V`` holds the top ``k`` right singular vectors of ``R~``. The item
    filter is a dense ``n_items x n_items`` matrix (memory as EASE); ``V``
    comes from ``torch.svd_lowrank`` on ``R~^T R~`` (same right singular
    vectors). Scores depend only on the history, so unseen users work.
    Defaults follow the official code (``alpha = 0.3``, ``k = 256``).
    """

    tasks = frozenset({"ranking"})
    inductive = True
    space = {"alpha": [0.0, 0.1, 0.3, 1.0], "k": [64, 128, 256, 512]}

    def __init__(self, alpha: float = 0.3, k: int = 256, oversample: int = 20, niter: int = 6,
                 device: str = "auto") -> None:
        self.alpha = alpha
        self.k = k
        self.oversample = oversample
        self.niter = niter
        self.device = resolve_device(device)

    def fit(self, data) -> None:
        X = data.history.matrix()
        item_deg = np.asarray(X.sum(0)).ravel()
        d = torch.as_tensor(inv_sqrt(item_deg), dtype=torch.float32, device=self.device)
        G = gram(X, self.device, inv_sqrt(np.asarray(X.sum(1)).ravel()) ** 2)  # R^T D_U^-1 R
        W = d[:, None] * G * d[None, :]  # R~^T R~
        k = min(self.k, W.shape[0])
        if self.alpha and k:
            # QR has no MPS kernel, so on Apple GPUs the randomized SVD runs on the CPU.
            _, _, V = torch.svd_lowrank(W.cpu() if W.device.type == "mps" else W, q=min(k + self.oversample, len(W)),
                                        niter=self.niter)
            V = V[:, :k].to(self.device)
            d_inv = torch.where(d > 0, 1 / d, torch.zeros_like(d))
            W = W + self.alpha * (d[:, None] * V) @ (V.T * d_inv[None, :])
        self.W = W

    def score(self, users, history):
        return history_rows(history, users, self.device) @ self.W
