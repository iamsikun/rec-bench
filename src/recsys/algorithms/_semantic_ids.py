"""Semantic-ID tokenizer for generative recommenders: item content vectors, RQ-VAE, prefix tables."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def item_content(data, n_factors: int = 64, seed: int = 0) -> torch.Tensor:
    """Item vectors from what a benchmark offers: truncated SVD of the train matrix plus one-hot side features.

    Stands in for the text embeddings (Sentence-T5) of the TIGER paper, which
    our benchmarks don't have. Columns are standardized.
    """
    m = data.history.matrix().tocoo()
    x = torch.sparse_coo_tensor(np.vstack([m.row, m.col]), m.data, m.shape).coalesce()
    q = max(1, min(n_factors, min(m.shape) - 1))
    torch.manual_seed(seed)
    _, s, v = torch.svd_lowrank(x, q=q, niter=4)
    parts = [v * s.sqrt()]
    for values in data.item_features.values():
        onehot = F.one_hot(torch.tensor(values, dtype=torch.long), int(values.max()) + 1).float()[:, 1:]
        parts.append(onehot * 0.5)
    out = torch.cat(parts, dim=1)
    return (out - out.mean(0)) / (out.std(0) + 1e-6)


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
            pick = torch.randint(len(residual), (book.shape[0],), device=x.device)
            book.copy_(residual[pick] + 0.01 * torch.randn_like(book))
            for _ in range(20):
                assign = F.one_hot(torch.cdist(residual, book).argmin(dim=1), book.shape[0]).float()
                count = assign.sum(0)[:, None]
                book.copy_(torch.where(count > 0, assign.T @ residual / count.clamp(min=1), book))
            residual = residual - book[torch.cdist(residual, book).argmin(dim=1)]

    def fit(self, x: torch.Tensor, steps: int = 1000, lr: float = 1e-3) -> torch.Tensor:
        """Full-batch training on item vectors ``x``; returns codes ``(N, levels)``."""
        self.init_codebooks(x)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(steps):
            _, zq, qloss = self.quantize(self.encoder(x))
            loss = F.mse_loss(self.decoder(zq), x) + qloss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():  # refit codebooks on the learned latents: fewer collisions than the trained ones
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
