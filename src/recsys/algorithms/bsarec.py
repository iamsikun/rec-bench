"""BSARec (Shin et al., AAAI 2024): self-attention mixed with a frequency-rescaling inductive bias."""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms._lastpos import FeedForward, LastPositionRecommender


class BSALayer(nn.Module):
    """One BSARec block: ``alpha * frequency branch + (1 - alpha) * attention branch``, then a feed-forward network.

    The frequency branch keeps the lowest ``c // 2 + 1`` frequencies of the
    sequence (a fixed low-pass matrix, computed once with the real FFT) and
    rescales the remainder by a learnable per-channel ``beta``.
    """

    def __init__(self, dim: int, n_heads: int, max_len: int, c: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.n_heads, self.alpha, self.dropout = n_heads, alpha, dropout
        spectrum = torch.fft.rfft(torch.eye(max_len), dim=0, norm="ortho")
        spectrum[c // 2 + 1 :] = 0
        self.register_buffer("low_pass", torch.fft.irfft(spectrum, n=max_len, dim=0, norm="ortho"), persistent=False)
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, dim))
        self.qkv, self.out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.norm_freq, self.norm_attn = nn.LayerNorm(dim, eps=1e-12), nn.LayerNorm(dim, eps=1e-12)
        self.ffn = FeedForward(dim, dropout)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        low = self.low_pass @ x
        freq = self.norm_freq(x + F.dropout(low + self.sqrt_beta**2 * (x - low), self.dropout, self.training))
        q, k, v = (t.view(B, L, self.n_heads, -1).transpose(1, 2) for t in self.qkv(x).chunk(3, dim=-1))
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                              dropout_p=self.dropout if self.training else 0.0)
        attn = self.out(attn.transpose(1, 2).reshape(B, L, D))
        attn = self.norm_attn(x + F.dropout(attn, self.dropout, self.training))
        return self.ffn(self.alpha * freq + (1 - self.alpha) * attn)


class BSARec(LastPositionRecommender):
    """BSARec (Shin et al., AAAI 2024): Beyond Self-Attention, causal attention plus a low/high-frequency rescaler.

    Each block averages, with weight ``alpha``, a post-norm causal
    self-attention branch and a filter branch ``low + beta^2 * (x - low)``
    whose low-pass keeps ``c // 2 + 1`` frequencies of the sequence. Like
    the reference code, training uses prefixes with full-softmax
    cross-entropy at the last position (the filter mixes the whole
    sequence, so earlier positions would see their targets) and the final
    position ranks the catalog.
    """

    space: ClassVar[dict[str, list]] = {"alpha": [0.3, 0.5, 0.7, 0.9], "c": [1, 3, 5, 7, 9], "dropout": [0.2, 0.5]}

    def __init__(self, dim: int = 64, n_layers: int = 2, n_heads: int = 1, max_len: int = 50, c: int = 5,
                 alpha: float = 0.7, dropout: float = 0.5, **kwargs) -> None:
        super().__init__(dim, n_layers, max_len, dropout, **kwargs)
        self.n_heads, self.c, self.alpha = n_heads, c, alpha

    def build(self, data) -> None:
        super().build(data)
        self.layers = nn.ModuleList(BSALayer(self.dim, self.n_heads, self.max_len, self.c, self.alpha, self.dropout)
                                    for _ in range(self.n_layers))

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        L = seq.shape[1]
        eye = torch.eye(L, dtype=torch.bool, device=seq.device)
        causal = torch.ones(L, L, dtype=torch.bool, device=seq.device).tril()
        allowed = ((causal & (seq > 0)[:, None, :]) | eye).unsqueeze(1)  # (B, 1, L, L): earlier, non-padding keys
        x = self.embed(seq)
        for layer in self.layers:
            x = layer(x, allowed)
        return x
