"""Semantic-ID tokenizer for generative recommenders: item content vectors, RQ-VAE, prefix tables."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn


def _svd_factors(m, q: int, seed: int, right: bool = True) -> torch.Tensor:
    """Rank-``q`` factors of a scipy matrix, scaled by the square roots of the singular values."""
    m = m.tocoo()
    x = torch.sparse_coo_tensor(np.vstack([m.row, m.col]), m.data, m.shape, dtype=torch.float32).coalesce()
    q = max(1, min(q, min(m.shape) - 1))
    torch.manual_seed(seed)
    u, s, v = torch.svd_lowrank(x, q=q, niter=4)
    return (v if right else u) * s.sqrt()


def item_content(data, n_factors: int = 64, seed: int = 0) -> torch.Tensor:
    """Item vectors from what a benchmark offers: truncated SVD of the train matrix plus one-hot side features.

    Stands in for the text embeddings (Sentence-T5) of the TIGER paper, which
    our benchmarks don't have. Columns are standardized.
    """
    parts = [_svd_factors(data.history.matrix(), n_factors, seed)]
    for values in data.item_features.values():
        onehot = F.one_hot(torch.tensor(values, dtype=torch.long), int(values.max()) + 1).float()[:, 1:]
        parts.append(onehot * 0.5)
    out = torch.cat(parts, dim=1)
    return (out - out.mean(0)) / (out.std(0) + 1e-6)


def item_transitions(data, dim: int = 32, seed: int = 0) -> torch.Tensor:
    """Sequential collaborative item embeddings: truncated SVD of the symmetric next-item transition counts.

    A cheap stand-in for the SASRec item embeddings LETTER aligns its codes
    with; unlike :func:`item_content` it sees order (who follows whom), not co-consumption. Standardized.
    """
    h = data.history
    nxt = np.arange(1, len(h.items))
    nxt = nxt[~np.isin(nxt, h.indptr)]  # pairs inside one user's sequence
    n = data.n_items
    a = sp.coo_matrix((np.ones(len(nxt)), (h.items[nxt - 1], h.items[nxt])), shape=(n, n))
    out = _svd_factors((a + a.T).tocsr(), dim, seed, right=False)
    out = (out - out.mean(0)) / (out.std(0) + 1e-6)
    return F.pad(out, (0, dim - out.shape[1]))  # tiny catalogs have rank below ``dim``


def kmeans(x: torch.Tensor, k: int, iters: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd's k-means from ``k`` random points; returns ``(centroids (k, d), assignments (N,))``."""
    c = x[torch.randint(len(x), (k,), device=x.device)] + 0.01 * torch.randn(k, x.shape[1], device=x.device)
    for _ in range(iters):
        assign = F.one_hot(torch.cdist(x, c).argmin(dim=1), k).float()
        count = assign.sum(0)[:, None]
        c = torch.where(count > 0, assign.T @ x / count.clamp(min=1), c)
    return c, torch.cdist(x, c).argmin(dim=1)


class RQVAE(nn.Module):
    """Residual-quantized autoencoder (Zeghidour et al. 2021; Rajput et al. 2023) with ``levels`` codebooks."""

    def __init__(self, in_dim: int, latent: int = 32, hidden: int = 256, levels: int = 3, codes: int = 256,
                 beta: float = 0.25) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                     nn.Linear(hidden // 2, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, hidden),
                                     nn.ReLU(), nn.Linear(hidden, in_dim))
        self.codebooks = nn.Parameter(torch.zeros(levels, codes, latent))
        self.beta = beta

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(codes (N, levels), straight-through quantized z, codebook + commitment loss)``."""
        residual, total, loss, codes = z, torch.zeros_like(z), z.new_zeros(()), []
        for book in self.codebooks:
            idx = torch.cdist(residual.detach(), book).argmin(dim=1)
            q = book[idx]
            loss = loss + F.mse_loss(q, residual.detach()) + self.beta * F.mse_loss(residual, q.detach())
            codes.append(idx)
            total = total + q
            residual = residual - q.detach()
        return torch.stack(codes, 1), z + (total - z).detach(), loss

    @torch.no_grad()
    def init_codebooks(self, x: torch.Tensor) -> None:
        """k-means init (as in the paper): each codebook is fit by Lloyd iterations on the level's residuals."""
        residual = self.encoder(x)
        for book in self.codebooks:
            centroids, assign = kmeans(residual, book.shape[0])
            book.copy_(centroids)
            residual = residual - book[assign]

    def fit(self, x: torch.Tensor, steps: int = 1000, lr: float = 1e-3, regularizer=None,
            refit: bool = True) -> torch.Tensor:
        """Full-batch training on item vectors ``x``; returns codes ``(N, levels)``.

        ``regularizer(z, zq, codes)`` adds a loss term (LETTER's collaborative and
        diversity terms). ``refit`` re-fits the codebooks by k-means on the trained
        latents, which cuts collisions but discards what the codebooks learned.
        """
        self.init_codebooks(x)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(steps):
            z = self.encoder(x)
            codes, zq, qloss = self.quantize(z)
            loss = F.mse_loss(self.decoder(zq), x) + qloss
            if regularizer is not None:
                loss = loss + regularizer(z, zq, codes)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            if refit:
                self.init_codebooks(x)
            return self.quantize(self.encoder(x))[0]


def disambiguate(codes: np.ndarray) -> np.ndarray:
    """Append a collision index so every item's semantic ID is unique (the TIGER paper's extra token)."""
    order = np.lexsort(codes.T[::-1])
    sorted_codes = codes[order]
    new_group = np.r_[True, np.any(sorted_codes[1:] != sorted_codes[:-1], axis=1)]
    group_start = np.maximum.accumulate(np.where(new_group, np.arange(len(codes)), 0))
    extra = np.empty(len(codes), dtype=np.int64)
    extra[order] = np.arange(len(codes)) - group_start
    return np.column_stack([codes, extra])


def prefix_tables(sids: np.ndarray, sizes: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Prefix trie as dense tables for constrained decoding.

    ``child[t][p, c]`` is the id of prefix ``p`` extended by code ``c`` at
    level ``t`` (``-1`` = no such item). Level 0 has the single empty prefix
    ``0``; at the last level the child id is the item id, so decoding only
    reaches real items. ``path[t][i]`` is item ``i``'s prefix id after ``t + 1`` codes.
    """
    tables, paths, prefix = [], [], np.zeros(len(sids), dtype=np.int64)
    for t, size in enumerate(sizes):
        table = np.full((prefix.max() + 1, size), -1, dtype=np.int64)
        if t == len(sizes) - 1:
            nxt = np.arange(len(sids))
        else:
            nxt = np.unique(np.column_stack([prefix, sids[:, t]]), axis=0, return_inverse=True)[1].reshape(-1)
        table[prefix, sids[:, t]] = nxt
        tables.append(table)
        paths.append(nxt)
        prefix = nxt
    return tables, paths
