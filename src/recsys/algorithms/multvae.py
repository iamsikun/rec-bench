"""Mult-VAE and Mult-DAE (Liang et al., WWW 2018): autoencoders with a multinomial likelihood."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms.trainer import TorchRecommender, minibatches


class MultVAE(TorchRecommender):
    """MultVAE (Liang et al., WWW 2018): variational autoencoder over a user's item row with a multinomial likelihood.

    Encoder ``n_items -> hidden -> 2 * latent`` (tanh) on the L2-normalized
    row after ``input_dropout``; decoder mirrors it. The KL weight anneals
    linearly from 0 to ``anneal_cap`` over ``total_anneal_steps`` updates.
    Scoring encodes the user's known items, so unseen users work (inductive).
    """

    tasks = frozenset({"ranking"})
    inductive = True
    space: ClassVar[dict[str, list]] = {
        "input_dropout": [0.3, 0.5], "anneal_cap": [0.1, 0.2, 0.3], "lr": [1e-3, 3e-3],
    }

    def __init__(self, hidden: int = 600, latent: int = 200, input_dropout: float = 0.5,
                 anneal_cap: float = 0.2, total_anneal_steps: int = 200_000, *, epochs: int = 200,
                 batch_size: int = 500, lr: float = 1e-3, patience: int | None = 10, **kwargs) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, **kwargs)
        self.hidden = hidden
        self.latent = latent
        self.input_dropout = input_dropout
        self.anneal_cap = anneal_cap
        self.total_anneal_steps = total_anneal_steps
        self.steps = 0

    def _layers(self, n_items: int) -> None:
        self.encoder = nn.Sequential(nn.Linear(n_items, self.hidden), nn.Tanh(),
                                     nn.Linear(self.hidden, 2 * self.latent))
        self.decoder = nn.Sequential(nn.Linear(self.latent, self.hidden), nn.Tanh(), nn.Linear(self.hidden, n_items))

    def build(self, data) -> None:
        self.n_items, self.steps = data.n_items, 0
        self._layers(data.n_items)
        for m in self.modules():
            if isinstance(m, nn.Linear):  # Xavier weights, small normal biases (reference implementation)
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=0.001)
        self.drop = nn.Dropout(self.input_dropout)
        h = data.history
        self.register_buffer("_indptr", torch.as_tensor(h.indptr), persistent=False)
        self.register_buffer("_items", torch.as_tensor(h.items), persistent=False)
        self.register_buffer("_train_users", torch.as_tensor(np.flatnonzero(h.lengths() > 0)), persistent=False)

    def rows(self, users: torch.Tensor) -> torch.Tensor:
        """Dense binary interaction rows ``(B, n_items)`` built on the device from the CSR history."""
        starts, ends = self._indptr[users], self._indptr[users + 1]
        lengths = ends - starts
        row = torch.repeat_interleave(torch.arange(len(users), device=users.device), lengths)
        offset = torch.arange(int(lengths.sum()), device=users.device) - torch.repeat_interleave(
            torch.cumsum(lengths, 0) - lengths, lengths)
        x = torch.zeros(len(users), self.n_items, device=users.device)
        x[row, self._items[starts[row] + offset]] = 1.0
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, kl)``; ``kl`` is the batch-mean KL divergence."""
        h = self.encoder(self.drop(F.normalize(x, dim=1)))
        mu, logvar = h[:, : self.latent], h[:, self.latent :]
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        kl = 0.5 * (logvar.exp() + mu.square() - 1 - logvar).sum(1).mean()
        return self.decoder(z), kl

    def batches(self, data):
        for idx in minibatches(len(self._train_users), self.batch_size, self.device):
            yield self.rows(self._train_users[idx])

    def loss(self, x):
        logits, kl = self(x)
        nll = -(F.log_softmax(logits, 1) * x).sum(1).mean()
        beta = min(self.anneal_cap, self.steps / max(self.total_anneal_steps, 1))
        self.steps += 1
        return nll + beta * kl

    def score(self, users, history):
        m = history.matrix(users).tocoo()
        x = torch.zeros(len(users), self.n_items, device=self.device)
        x[torch.as_tensor(m.row, device=self.device).long(), torch.as_tensor(m.col, device=self.device).long()] = 1.0
        return self(x)[0]


class MultDAE(MultVAE):
    """MultDAE (Liang et al., WWW 2018): denoising autoencoder over a user's item row with a multinomial likelihood.

    ``n_items -> hidden -> latent -> hidden -> n_items`` with tanh, trained on
    dropout-corrupted inputs; L2 regularization ``reg`` on the weights.
    """

    space: ClassVar[dict[str, list]] = {"input_dropout": [0.3, 0.5], "reg": [0.0, 0.001, 0.01], "lr": [1e-3, 3e-3]}

    def __init__(self, hidden: int = 600, latent: int = 200, input_dropout: float = 0.5, reg: float = 0.01,
                 **kwargs) -> None:
        super().__init__(hidden, latent, input_dropout, **kwargs)
        self.reg = reg

    def _layers(self, n_items: int) -> None:
        self.encoder = nn.Sequential(nn.Linear(n_items, self.hidden), nn.Tanh(), nn.Linear(self.hidden, self.latent),
                                     nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(self.latent, self.hidden), nn.Tanh(), nn.Linear(self.hidden, n_items))

    def forward(self, x):
        return self.decoder(self.encoder(self.drop(F.normalize(x, dim=1)))), None

    def loss(self, x):
        logits, _ = self(x)
        l2 = sum(m.weight.square().sum() for m in self.modules() if isinstance(m, nn.Linear))
        # Reference code: mean NLL + 2 * (lam / 2) * sum ||W||^2.
        return -(F.log_softmax(logits, 1) * x).sum(1).mean() + self.reg * l2
