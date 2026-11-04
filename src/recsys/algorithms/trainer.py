"""``TorchRecommender``: the built-in training loop for neural algorithms."""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from recsys.algorithms.base import Recommender
from recsys.device import resolve_device

if TYPE_CHECKING:  # pragma: no cover
    from recsys import TrainData

LOGGER = logging.getLogger(__name__)


def minibatches(n: int, batch_size: int, device: torch.device) -> Iterator[torch.Tensor]:
    """Yield shuffled index tensors covering ``range(n)``."""
    perm = torch.randperm(n, device=device)
    for start in range(0, n, batch_size):
        yield perm[start : start + batch_size]


class TorchRecommender(Recommender, nn.Module):
    """A ``Recommender`` trained by a minimal Adam loop with early stopping.

    Subclasses implement :meth:`build` (create layers and device-resident
    training tensors — register them with ``register_buffer(...,
    persistent=False)`` so they follow ``.to(device)``), :meth:`batches` and
    :meth:`loss`. After every ``eval_every`` epochs the loop calls
    ``data.validate``; it stops after ``patience`` evaluations without
    improvement and restores the best weights. ``losses`` and ``val_metrics``
    record the per-epoch training loss and validation metric.
    """

    def __init__(
        self,
        *,
        epochs: int = 50,
        batch_size: int = 1024,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int | None = 5,
        eval_every: int = 1,
        device: str = "auto",
    ) -> None:
        nn.Module.__init__(self)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.eval_every = eval_every
        self.device = resolve_device(device)

    @abstractmethod
    def build(self, data: TrainData) -> None:
        """Create parameters and training tensors from ``data``."""

    @abstractmethod
    def batches(self, data: TrainData) -> Iterator[Any]:
        """Yield one epoch of training batches, already on :attr:`device`."""

    @abstractmethod
    def loss(self, batch: Any) -> torch.Tensor:
        """Return the scalar training loss for ``batch``."""

    def fit(self, data: TrainData) -> None:
        self.build(data)
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, stale = -float("inf"), None, 0
        self.losses, self.val_metrics = [], []
        for epoch in range(1, self.epochs + 1):
            self.train()
            start, total, steps = time.perf_counter(), torch.zeros((), device=self.device), 0
            for batch in self.batches(data):
                loss = self.loss(batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.detach()
                steps += 1
            self.losses.append(total.item() / max(steps, 1))
            message = f"epoch {epoch}: loss {self.losses[-1]:.4f} ({time.perf_counter() - start:.1f}s)"
            if data.validate is not None and epoch % self.eval_every == 0:
                self.eval()
                metric = data.validate(self)
                self.val_metrics.append(metric)
                message += f", val {metric:.4f}"
                if metric > best:
                    best, stale = metric, 0
                    best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                else:
                    stale += 1
            LOGGER.info("%s %s", type(self).__name__, message)
            if self.patience is not None and stale >= self.patience:
                break
        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()
