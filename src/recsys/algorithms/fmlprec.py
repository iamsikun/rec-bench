"""FMLP-Rec (Zhou et al., WWW 2022): an all-MLP sequential recommender with learnable frequency-domain filters."""

from __future__ import annotations

import math
from typing import ClassVar

import torch
from torch import nn

from recsys.algorithms._lastpos import FeedForward, LastPositionRecommender


class FilterLayer(nn.Module):
    """Learnable global filter along the sequence: ``LayerNorm(x + Dropout(irfft(W * rfft(x))))``.

    The orthonormal real FFT and its inverse are written as products with
    fixed cosine/sine matrices, and the complex weight ``W`` as its real and
    imaginary parts, so no complex tensors are needed on any device (cheap
    for ``max_len <= 200``).
    """

    def __init__(self, dim: int, max_len: int, dropout: float) -> None:
        super().__init__()
        n_freq = max_len // 2 + 1
        self.weight = nn.Parameter(torch.randn(2, n_freq, dim) * 0.02)
        angle = 2 * math.pi * torch.arange(n_freq)[:, None] * torch.arange(max_len)[None] / max_len
        cos, sin = torch.cos(angle) / math.sqrt(max_len), torch.sin(angle) / math.sqrt(max_len)
        # irfft counts each non-DC, non-Nyquist frequency twice (Hermitian symmetry).
        twice = torch.full((n_freq, 1), 2.0)
        twice[0] = 1.0
        if max_len % 2 == 0:
            twice[-1] = 1.0
        for name, value in {"cos": cos, "sin": sin, "icos": (twice * cos).T, "isin": (twice * sin).T}.items():
            self.register_buffer(name, value.contiguous(), persistent=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        re, im = self.cos @ x, -(self.sin @ x)  # (B, n_freq, dim)
        wr, wi = self.weight
        y = self.icos @ (re * wr - im * wi) - self.isin @ (re * wi + im * wr)
        return self.norm(x + self.drop(y))


class FMLPRec(LastPositionRecommender):
    """FMLP-Rec (Zhou et al., WWW 2022): filter-enhanced MLP blocks in place of self-attention.

    Item plus position embeddings pass through ``n_layers`` blocks, each a
    learnable complex filter applied in the frequency domain of the
    sequence (residual, dropout, LayerNorm) followed by a GELU feed-forward
    network. The final position's state ranks the catalog by dot product
    with the item embeddings.

    Deviations from the paper: the filter mixes the whole sequence, so the
    reference code's next-item loss at every position lets each position see
    its own target; we instead train on prefixes with full-softmax
    cross-entropy at the last position (as the BSARec authors' FMLP-Rec
    does) rather than one-negative BCE at every position.
    """

    space: ClassVar[dict[str, list]] = {"dropout": [0.2, 0.5], "n_layers": [1, 2, 4], "lr": [1e-3, 3e-3]}

    def __init__(self, dim: int = 64, n_layers: int = 2, max_len: int = 50, dropout: float = 0.5, **kwargs) -> None:
        super().__init__(dim, n_layers, max_len, dropout, **kwargs)

    def build(self, data) -> None:
        super().build(data)
        self.layers = nn.ModuleList(
            nn.Sequential(FilterLayer(self.dim, self.max_len, self.dropout), FeedForward(self.dim, self.dropout))
            for _ in range(self.n_layers))

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.embed(seq)
        for layer in self.layers:
            x = layer(x)
        return x
