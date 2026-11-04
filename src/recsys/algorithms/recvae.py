"""RecVAE (Shenbin et al., WSDM 2020): Mult-VAE with a dense encoder, composite prior and alternating updates."""

from __future__ import annotations

import math
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms.multvae import MultVAE


def log_normal(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Elementwise log-density of ``N(mu, exp(logvar))`` at ``z``."""
    return -0.5 * (logvar + math.log(2 * math.pi) + (z - mu).square() / logvar.exp())


class Encoder(nn.Module):
    """Densely connected encoder: each layer sees the sum of all earlier outputs; swish then layer norm."""

    def __init__(self, n_items: int, hidden: int, latent: int, depth: int = 5) -> None:
        super().__init__()
        self.fc = nn.ModuleList([nn.Linear(n_items if d == 0 else hidden, hidden) for d in range(depth)])
        self.ln = nn.ModuleList([nn.LayerNorm(hidden, eps=0.1) for _ in range(depth)])
        self.out = nn.Linear(hidden, 2 * latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = 0
        for fc, ln in zip(self.fc, self.ln, strict=True):
            h = ln(F.silu(fc(x) + skip))
            skip, x = skip + h, h
        return self.out(x).chunk(2, dim=1)


class RecVAE(MultVAE):
    """RecVAE (Shenbin et al., WSDM 2020): Mult-VAE with a dense encoder, composite prior and alternating updates.

    The encoder is a five-layer densely connected MLP (swish, layer norm) on the
    L2-normalized row; the decoder is a single linear layer. The KL term is
    taken against a composite prior, a ``3/20, 3/4, 1/10`` mixture of
    ``N(0, I)``, the posterior of a frozen copy of the encoder from the previous
    epoch, and a wide ``N(0, e^10 I)``. Each user's KL weight is ``gamma`` times
    their number of interactions. One epoch makes ``enc_epochs`` passes that
    update only the encoder (input dropout on), refreshes the prior's encoder
    copy, then ``dec_epochs`` passes that update only the decoder (no dropout).
    The KL is a single-sample Monte-Carlo estimate, as in the official code.

    Deviations from the paper:

    * One Adam optimizer holds both parts; a pass leaves the other part's
      gradients unset so Adam skips it, which equals the official two optimizers.
    * Linear layers use Xavier init (as ``MultVAE``) rather than PyTorch's default.
    """

    space: ClassVar[dict[str, list]] = {"gamma": [0.003, 0.005, 0.01], "lr": [5e-4, 1e-3]}
    mixture = (3 / 20, 3 / 4, 1 / 10)

    def __init__(self, hidden: int = 600, latent: int = 200, input_dropout: float = 0.5, gamma: float = 0.005,
                 enc_epochs: int = 3, dec_epochs: int = 1, *, epochs: int = 50, batch_size: int = 500,
                 lr: float = 5e-4, weight_decay: float = 0.0, patience: int | None = 10, eval_every: int = 1,
                 device: str = "auto") -> None:
        super().__init__(hidden, latent, input_dropout, epochs=epochs, batch_size=batch_size, lr=lr,
                         weight_decay=weight_decay, patience=patience, eval_every=eval_every, device=device)
        self.gamma = gamma
        self.enc_epochs = enc_epochs
        self.dec_epochs = dec_epochs

    def _layers(self, n_items: int) -> None:
        self.encoder = Encoder(n_items, self.hidden, self.latent)
        self.prior_encoder = Encoder(n_items, self.hidden, self.latent).requires_grad_(False)
        self.decoder = nn.Linear(self.latent, n_items)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.decoder(self.encoder(F.normalize(x, dim=1))[0]), None

    def batches(self, data):
        for _ in range(self.enc_epochs):
            for x in super().batches(data):
                yield x, True
        self.prior_encoder.load_state_dict(self.encoder.state_dict())
        for _ in range(self.dec_epochs):
            for x in super().batches(data):
                yield x, False

    def prior(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Elementwise log-density of the composite prior at ``z``."""
        with torch.no_grad():
            mu, logvar = self.prior_encoder(F.normalize(x, dim=1))
        zero = torch.zeros_like(z)
        parts = [log_normal(z, zero, zero), log_normal(z, mu, logvar), log_normal(z, zero, zero + 10)]
        return torch.logsumexp(torch.stack([p + math.log(w) for p, w in zip(parts, self.mixture, strict=True)]), 0)

    def loss(self, batch):
        x, encoder_step = batch
        with torch.set_grad_enabled(encoder_step):
            h = F.normalize(x, dim=1)
            mu, logvar = self.encoder(self.drop(h) if encoder_step else h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        weight, bias = self.decoder.weight, self.decoder.bias
        if encoder_step:
            weight, bias = weight.detach(), bias.detach()
        nll = -(F.log_softmax(F.linear(z, weight, bias), 1) * x).sum(1)
        kl = (log_normal(z, mu, logvar) - self.prior(x, z)).sum(1)
        return (nll + self.gamma * x.sum(1) * kl).mean()
