"""Mamba4Rec (Liu et al., 2024): selective state-space (Mamba) blocks for sequential recommendation."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms.gru4rec import left_align
from recsys.algorithms.trainer import TorchRecommender, minibatches


class Mamba(nn.Module):
    """The Mamba S6 block (Gu & Dao, 2023) in pure torch.

    Input projection to a gate ``z`` and a stream ``u``; a depthwise causal
    convolution and SiLU on ``u``; input-dependent step ``delta``, ``B`` and
    ``C``; the recurrence ``h_t = exp(delta_t A) h_{t-1} + delta_t B_t u_t``,
    ``y_t = C_t h_t + D u_t`` as a sequential scan (one fused ``addcmul``
    per step); output ``y * SiLU(z)`` projected back. Initialization follows
    the reference ``mamba_ssm`` code.
    """

    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        inner, self.rank, self.d_state = expand * dim, math.ceil(dim / 16), d_state
        self.in_proj = nn.Linear(dim, 2 * inner, bias=False)
        self.conv = nn.Conv1d(inner, inner, d_conv, groups=inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(inner, self.rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.rank, inner)
        nn.init.uniform_(self.dt_proj.weight, -self.rank**-0.5, self.rank**-0.5)
        dt = torch.exp(torch.rand(inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # softplus(bias) = dt
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float)).repeat(inner, 1))
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2))  # causal: drop the right overhang
        dt, b, c = self.x_proj(u).split([self.rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))  # (B, L, inner)
        decay = torch.exp(delta.unsqueeze(-1) * -torch.exp(self.A_log))  # (B, L, inner, d_state)
        drive = (delta * u).unsqueeze(-1) * b.unsqueeze(2)
        h, states = torch.zeros_like(decay[:, 0]), []
        for a_t, b_t in zip(decay.unbind(1), drive.unbind(1)):  # unbind: one gradient stack, not L scatters
            h = torch.addcmul(b_t, a_t, h)
            states.append(h)
        y = torch.einsum("bldn,bln->bld", torch.stack(states, 1), c) + self.D * u
        return self.out_proj(y * F.silu(z))


class MambaLayer(nn.Module):
    """Mamba, dropout and LayerNorm (residual only when stacked), then a GELU feed-forward network."""

    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int, dropout: float, residual: bool) -> None:
        super().__init__()
        self.mamba, self.residual = Mamba(dim, d_state, d_conv, expand), residual
        self.drop = nn.Dropout(dropout)
        self.norm, self.norm_ffn = nn.LayerNorm(dim, eps=1e-12), nn.LayerNorm(dim, eps=1e-12)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.mamba(x))
        h = self.norm(h + x if self.residual else h)
        return self.norm_ffn(h + self.ffn(h))


class Mamba4Rec(TorchRecommender):
    """Mamba4Rec (Liu et al., 2024): a linear-time selective SSM in place of self-attention, full-softmax CE.

    Item embeddings (dropout, LayerNorm) pass through ``n_layers`` Mamba
    layers; the state after the user's last item ranks the catalog by dot
    product with the item embeddings. Sequences are left-aligned (padding at
    the end, as in the reference RecBole code), so padding never enters the
    recurrence and scores do not depend on ``max_len``.

    Deviations from the paper: the selective scan is a pure-torch
    sequential loop instead of the fused ``mamba_ssm`` CUDA kernel (same
    function, slower); because the model is causal and has no position
    embeddings, cross-entropy is applied at every position of each user's
    sequence instead of at the last position of every prefix (one forward
    pass instead of ``max_len``), so ``batch_size`` counts users, not prefixes.
    """

    tasks = frozenset({"ranking"})
    inductive = True
    requires = frozenset({"timestamps"})
    space: ClassVar[dict[str, list]] = {"dropout": [0.2, 0.4], "n_layers": [1, 2], "max_len": [50, 200]}

    def __init__(self, dim: int = 64, n_layers: int = 1, d_state: int = 32, d_conv: int = 4, expand: int = 2,
                 max_len: int = 200, dropout: float = 0.2, *, epochs: int = 200, batch_size: int = 128,
                 lr: float = 1e-3, patience: int | None = 10, **kwargs) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, **kwargs)
        self.dim, self.n_layers, self.max_len, self.dropout = dim, n_layers, max_len, dropout
        self.d_state, self.d_conv, self.expand = d_state, d_conv, expand

    def build(self, data) -> None:
        # Item ids are shifted by one so that 0 is padding.
        self.item_emb = nn.Embedding(data.n_items + 1, self.dim, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.norm = nn.LayerNorm(self.dim, eps=1e-12)
        self.layers = nn.ModuleList(
            MambaLayer(self.dim, self.d_state, self.d_conv, self.expand, self.dropout, self.n_layers > 1)
            for _ in range(self.n_layers))
        users = np.flatnonzero(data.history.lengths() >= 2)
        seqs = torch.as_tensor(data.history.sequences(users, self.max_len + 1) + 1)
        self.register_buffer("_seqs", left_align(seqs)[0], persistent=False)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """Hidden states ``(B, L, dim)`` for left-aligned sequences ``(B, L)``; position ``t`` sees items ``<= t``."""
        x = self.norm(F.dropout(self.item_emb(seq), self.dropout, self.training))
        for layer in self.layers:
            x = layer(x)
        return x

    def batches(self, data):
        for idx in minibatches(len(self._seqs), self.batch_size, self.device):
            seq = self._seqs[idx]
            yield seq[:, : int((seq > 0).sum(1).max())]  # trailing all-padding columns cannot change anything

    def loss(self, seq):
        logits = self.encode(seq[:, :-1]) @ self.item_emb.weight.T  # padding is at the end: targets 0 are ignored
        return F.cross_entropy(logits.flatten(0, 1), seq[:, 1:].flatten(), ignore_index=0)

    def score(self, users, history):
        seq, lengths = left_align(torch.as_tensor(history.sequences(users, self.max_len) + 1, device=self.device))
        h = self.encode(seq[:, : max(int(lengths.max()), 1)])
        h = h[torch.arange(len(seq), device=self.device), (lengths - 1).clamp(min=0)]
        return h @ self.item_emb.weight[1:].T
