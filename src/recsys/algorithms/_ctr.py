"""Shared base for CTR models: field embeddings, device-resident training tensors, BCE loss and predict."""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms.trainer import TorchRecommender, minibatches


class Dice(nn.Module):
    """Dice activation (Zhou et al., KDD 2018): a PReLU whose rectifier point adapts to the batch statistics."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(width, affine=False, eps=1e-9)
        self.alpha = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        p = torch.sigmoid(self.bn(x))
        return (p * x + (1 - p) * self.alpha * x).reshape(shape)


def mlp(width: int, hidden: tuple[int, ...], dropout: float = 0.0, out: int | None = 1,
        act: Callable[[int], nn.Module] = lambda _: nn.ReLU(), batch_norm: bool = False) -> nn.Sequential:
    """``hidden`` fully connected layers (Linear [-> BatchNorm] -> act -> Dropout), then an optional output layer."""
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(width, h), *([nn.BatchNorm1d(h)] if batch_norm else []), act(h), nn.Dropout(dropout)]
        width = h
    if out is not None:
        layers.append(nn.Linear(width, out))
    return nn.Sequential(*layers)


def fits_on_device(nbytes: int, device: torch.device) -> bool:
    """Whether ``nbytes`` of training data may live on ``device``.

    CUDA: at most half of the currently free memory. MPS shares RAM with the
    host: at most 40% of physical RAM. CPU: always.
    """
    if device.type == "cuda":
        return nbytes < 0.5 * torch.cuda.mem_get_info(device)[0]
    if device.type == "mps":
        return nbytes < 0.4 * os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return True


class CTRModel(TorchRecommender):
    """Base for CTR networks over categorical fields; subclasses implement :meth:`network` and :meth:`forward`.

    One embedding table holds every field (``dim`` columns, offsets per field);
    :meth:`embed` returns ``(B, n_fields, dim)``. Set ``first_order = True`` for
    a per-feature weight table plus bias (:meth:`linear`). Models whose
    ``requires`` contains ``"behavior"`` also receive the ``history`` column,
    encoded in the ``item`` field's id space and embedded with the item table
    (:meth:`behavior`). ``embedding_reg`` is an L2 penalty on the embedding
    tables added to every batch loss (FuxiCTR's ``embedding_regularizer``).

    Memory rule: the training matrix (field ids plus offsets, and the history)
    is built column by column as int32, never as an int64 copy, and no
    reference to the polars frame is kept. It lives on the training device
    when :func:`fits_on_device` allows, otherwise in CPU memory with each
    batch gathered there and copied to the device (``non_blocking``).
    """

    tasks = frozenset({"ctr"})
    first_order = False
    init_std = 1e-2

    def __init__(self, dim: int = 16, embedding_reg: float = 1e-5, *, epochs: int = 20, batch_size: int = 4096,
                 lr: float = 1e-3, patience: int | None = 2, **kwargs) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, **kwargs)
        self.dim = dim
        self.embedding_reg = embedding_reg

    @property
    def uses_history(self) -> bool:
        return "behavior" in self.requires

    def network(self, data) -> None:
        """Create the layers that sit on top of the embeddings."""

    def build(self, data) -> None:
        self.field_names = list(data.fields)
        self.n_fields = len(self.field_names)
        sizes = np.array([data.fields[f] for f in self.field_names], dtype=np.int64)
        if sizes.sum() >= 2**31:
            raise ValueError("total vocabulary exceeds int32 ids")
        self.offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32)
        self.emb = nn.Embedding(int(sizes.sum()), self.dim)
        nn.init.normal_(self.emb.weight, std=self.init_std)
        if self.first_order:
            self.weight = nn.Embedding(int(sizes.sum()), 1)
            nn.init.zeros_(self.weight.weight)
            self.bias = nn.Parameter(torch.zeros(()))
        if self.uses_history:
            self.item_field = self.field_names.index("item")
            self.item_offset = int(self.offsets[self.item_field])
        self.network(data)
        x, h = self._encode(data.train)
        y = torch.from_numpy(data.train["label"].to_numpy().astype(np.float32))
        nbytes = sum(t.numel() * t.element_size() for t in (x, h, y) if t is not None)
        self.data_device = self.device if fits_on_device(nbytes, self.device) else torch.device("cpu")
        self._x, self._y = x.to(self.data_device), y.to(self.data_device)
        self._h = None if h is None else h.to(self.data_device)

    def _encode(self, rows) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Field ids plus offsets as an ``(n, n_fields)`` int32 CPU tensor, and the int32 history (or ``None``)."""
        x = np.empty((rows.height, self.n_fields), dtype=np.int32)
        for j, f in enumerate(self.field_names):
            x[:, j] = rows[f].to_numpy()
            x[:, j] += self.offsets[j]
        h = torch.from_numpy(np.require(rows["history"].to_numpy(), np.int32, "W")) if self.uses_history else None
        return torch.from_numpy(x), h

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        """First-order logit: bias + sum of per-feature weights."""
        return self.bias + self.weight(x).sum((1, 2))

    def behavior(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """History ids ``(B, L)`` (-1 = padding) -> item embeddings ``(B, L, dim)`` and a validity mask."""
        mask = h >= 0
        return self.emb(torch.where(mask, h + self.item_offset, self.item_offset)), mask

    def reg(self) -> torch.Tensor:
        if not self.embedding_reg:
            return torch.zeros((), device=self.device)
        tables = [self.emb.weight] + ([self.weight.weight] if self.first_order else [])
        return self.embedding_reg * sum(t.square().sum() for t in tables)

    def _to_device(self, t: torch.Tensor | None) -> torch.Tensor | None:
        return None if t is None else t.to(self.device, non_blocking=True)

    def batches(self, data):
        for idx in minibatches(len(self._y), self.batch_size, self.data_device):
            yield tuple(self._to_device(None if t is None else t[idx]) for t in (self._x, self._h, self._y))

    def loss(self, batch):
        x, h, y = batch
        return F.binary_cross_entropy_with_logits(self(x, h), y) + self.reg()

    @torch.no_grad()
    def predict(self, rows):
        x, h = self._encode(rows)
        chunks = [self(self._to_device(x[i:i + self.batch_size]),
                       self._to_device(None if h is None else h[i:i + self.batch_size]))
                  for i in range(0, len(x), self.batch_size)]
        return torch.sigmoid(torch.cat(chunks)).float().cpu().numpy()
