"""iALS (Hu et al., ICDM 2008), in the formulation of Rendle et al. (RecSys 2022, "Revisiting iALS")."""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.sparse as sp
import torch

from recsys.algorithms.base import Recommender
from recsys.device import resolve_device

LOGGER = logging.getLogger(__name__)


def _cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    L = torch.linalg.cholesky(A)
    if L.device.type == "mps":  # no MPS kernel for cholesky_solve: triangular solves on the CPU
        return torch.cholesky_solve(b.cpu(), L.cpu()).to(L.device)
    return torch.cholesky_solve(b, L)


class IALS(Recommender):
    """iALS (Hu et al., ICDM 2008): matrix factorization fitting all user-item cells, unobserved ones down-weighted.

    Minimizes (Rendle et al., RecSys 2022) ``sum_{observed} (w_u.h_i - 1)^2 +
    alpha0 sum_{all u,i} (w_u.h_i)^2 + reg * (sum_u (|I_u| + alpha0 |I|)^nu
    |w_u|^2 + sum_i (|U_i| + alpha0 |U|)^nu |h_i|^2)`` by alternating least
    squares. Each row solves ``(alpha0 F^T F + F_S^T F_S + reg_r I) x =
    F_S^T 1``; rows are batched by length (padded gathers, ``bmm``) and solved
    exactly by batched Cholesky on the device, or, with ``cg_steps``, by that
    many warm-started conjugate-gradient steps (cheap for large ``dim``).
    Users are always scored by an exact fold-in solve against the item
    factors, so held-out users work. ``losses`` records the objective per epoch.
    """

    tasks = frozenset({"ranking"})
    inductive = True
    space = {"dim": [64, 128, 256], "reg": [1e-3, 3e-3, 1e-2], "alpha0": [0.03, 0.1, 0.3]}

    def __init__(self, dim: int = 64, alpha0: float = 0.1, reg: float = 0.003, nu: float = 1.0, epochs: int = 16,
                 init_std: float = 0.1, cg_steps: int | None = None, budget: int = 2**25, device: str = "auto") -> None:
        self.dim = dim
        self.alpha0 = alpha0
        self.reg = reg
        self.nu = nu
        self.epochs = epochs
        self.init_std = init_std
        self.cg_steps = cg_steps
        self.budget = budget  # max floats per padded gather / batch of d x d systems
        self.device = resolve_device(device)

    def _batches(self, X: sp.csr_matrix, exact: bool) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Row ids and ``-1``-padded column ids; rows sorted by length, at most 2x padding, within ``budget``."""
        lens = np.diff(X.indptr)
        order = np.argsort(-lens, kind="stable")
        neg_sorted = -lens[order]
        max_rows = max(1, self.budget // self.dim**2) if exact else 1 << 16
        out, s = [], 0
        while s < len(order):
            L = max(int(lens[order[s]]), 1)
            similar = int(np.searchsorted(neg_sorted, -(L // 2), side="right")) - s  # rows longer than L / 2
            rows = order[s : s + max(1, min(max_rows, self.budget // (L * self.dim), similar))]
            off = np.arange(L)[None, :]
            pos = np.minimum(X.indptr[rows][:, None] + off, max(X.nnz - 1, 0))
            cols = np.where(off < lens[rows][:, None], X.indices[pos] if X.nnz else 0, -1)
            out.append((torch.as_tensor(rows, device=self.device), torch.as_tensor(cols, device=self.device)))
            s += len(rows)
        return out

    def _solve(self, F: torch.Tensor, batches, out: torch.Tensor, exact: bool) -> torch.Tensor:
        """Least-squares rows against fixed factors ``F``; ``out`` holds warm starts and receives the result."""
        gram = self.alpha0 * (F.T @ F)
        F_pad = torch.cat([F, F.new_zeros(1, self.dim)])  # column id -1 gathers the zero row
        for rows, cols in batches:
            Fs = F_pad[cols]  # (B, L, d)
            n = (cols >= 0).sum(1, keepdim=True).float()
            lam = self.reg * (n + self.alpha0 * F.shape[0]) ** self.nu
            b = Fs.sum(1)
            if exact:
                A = torch.baddbmm(gram.expand(len(rows), -1, -1), Fs.mT, Fs)
                A.diagonal(dim1=1, dim2=2).add_(lam)
                out[rows] = _cholesky_solve(A, b[:, :, None])[:, :, 0]
                continue
            def matvec(v, Fs=Fs, lam=lam):
                return v @ gram + (Fs.mT @ (Fs @ v[:, :, None]))[:, :, 0] + lam * v

            x = out[rows]
            r = b - matvec(x)
            p, rs = r, (r * r).sum(1, keepdim=True)
            for _ in range(self.cg_steps):
                Ap = matvec(p)
                a = rs / (p * Ap).sum(1, keepdim=True).clamp_min(1e-20)
                x, r = x + a * p, r - a * Ap
                rs_new = (r * r).sum(1, keepdim=True)
                p, rs = r + rs_new / rs.clamp_min(1e-20) * p, rs_new
            out[rows] = x
        return out

    def objective(self) -> float:
        """The training loss above for the current factors."""
        W, H, total = self.W, self.H, 0.0
        H_pad = torch.cat([H, H.new_zeros(1, self.dim)])
        for rows, cols in self._user_batches:
            pred = (H_pad[cols] @ W[rows][:, :, None])[:, :, 0]
            total += torch.where(cols >= 0, (pred - 1) ** 2, 0.0).sum().item()
        total += self.alpha0 * ((W.T @ W) * (H.T @ H)).sum().item()
        for E, n, other in ((W, self._user_n, H), (H, self._item_n, W)):
            weight = (n + self.alpha0 * other.shape[0]) ** self.nu
            total += self.reg * (weight * E.square().sum(1)).sum().item()
        return total

    def fit(self, data) -> None:
        X = data.history.matrix().astype(np.float32)
        X.sum_duplicates()
        exact = self.cg_steps is None
        self._user_batches = self._batches(X, exact)
        item_batches = self._batches(X.T.tocsr(), exact)
        self._user_n = torch.as_tensor(np.diff(X.indptr), device=self.device).float()
        self._item_n = torch.as_tensor(np.diff(X.T.tocsr().indptr), device=self.device).float()
        g = torch.Generator(device=self.device).manual_seed(data.seed)
        std = self.init_std / self.dim**0.5
        self.W = torch.randn(data.n_users, self.dim, device=self.device, generator=g) * std
        self.H = torch.randn(data.n_items, self.dim, device=self.device, generator=g) * std
        self.losses = []
        for epoch in range(1, self.epochs + 1):
            start = time.perf_counter()
            self._solve(self.H, self._user_batches, self.W, exact)
            self._solve(self.W, item_batches, self.H, exact)
            self.losses.append(self.objective())
            LOGGER.info("IALS epoch %d: objective %.2f (%.1fs)", epoch, self.losses[-1], time.perf_counter() - start)

    def score(self, users, history):
        X = history.matrix(users)
        W = self._solve(self.H, self._batches(X, True), torch.zeros(len(users), self.dim, device=self.device), True)
        return W @ self.H.T
