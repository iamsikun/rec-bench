"""Shared pieces of the two-tower retrieval models (DSSM, YouTubeDNN, TwoTowerLogQ).

Every model here produces a user vector and an item vector; :meth:`TwoTower.recommend`
returns the exact maximum-inner-product top-k over the item tower, skipping items
already in the user's history.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms.trainer import TorchRecommender, minibatches

if TYPE_CHECKING:  # pragma: no cover
    from recsys import History


def embedding(n: int, dim: int, std: float = 0.01) -> nn.Embedding:
    emb = nn.Embedding(n, dim)
    nn.init.normal_(emb.weight, std=std)
    return emb


def mlp(dims: Sequence[int]) -> nn.Sequential:
    """Linear layers with ReLU in between (none after the last)."""
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


class FeatureEmbedding(nn.Module):
    """Concatenated embeddings of categorical side features, looked up by entity id (value 0 = missing)."""

    def __init__(self, features: dict[str, np.ndarray], dim: int) -> None:
        super().__init__()
        names = sorted(features)
        self.out_dim = dim * len(names)
        self.embs = nn.ModuleList(embedding(int(np.max(features[n])) + 1, dim) for n in names)
        table = np.stack([np.asarray(features[n], dtype=np.int64) for n in names], 1) if names else np.zeros((1, 0))
        self.register_buffer("table", torch.as_tensor(table, dtype=torch.long), persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if not self.embs:
            return torch.zeros((len(ids), 0), device=ids.device)
        return torch.cat([emb(self.table[ids, j]) for j, emb in enumerate(self.embs)], -1)


def mean_pool(emb: nn.Embedding, seq: torch.Tensor) -> torch.Tensor:
    """Mean embedding of the non-padding (``>= 0``) ids of each row; zeros for empty rows."""
    mask = (seq >= 0).unsqueeze(-1).float()
    return (emb(seq.clamp(min=0)) * mask).sum(1) / mask.sum(1).clamp(min=1.0)


def in_batch_softmax(u: torch.Tensor, v: torch.Tensor, items: torch.Tensor, temperature: float,
                     log_q: torch.Tensor | None = None) -> torch.Tensor:
    """Softmax over the batch's items; row i's positive is column i. Repeats of the positive are masked."""
    logits = u @ v.T / temperature
    if log_q is not None:
        logits = logits - log_q[None, :]
    repeats = (items[:, None] == items[None, :]).fill_diagonal_(False)
    logits = logits.masked_fill(repeats, -1e4)  # finite: -inf takes a slow path on MPS
    return F.cross_entropy(logits, torch.arange(len(items), device=items.device))


def top_k_unseen(scores: torch.Tensor, users: np.ndarray, history: History, k: int) -> torch.Tensor:
    """Top-``k`` item ids per row of ``scores`` excluding ``history`` items; ``-1`` pads short lists."""
    seen = history.matrix(np.asarray(users)).tocoo()
    rows = torch.as_tensor(seen.row, dtype=torch.long, device=scores.device)
    cols = torch.as_tensor(seen.col, dtype=torch.long, device=scores.device)
    scores = scores.index_put((rows, cols), torch.tensor(-torch.inf, device=scores.device))
    vals, ids = scores.topk(min(k, scores.shape[1]), dim=1)
    ids = ids.masked_fill(vals == -torch.inf, -1)
    return F.pad(ids, (0, k - ids.shape[1]), value=-1)


class TwoTower(TorchRecommender):
    """Base class: subclasses define :meth:`user_vectors` and :meth:`item_vectors`."""

    tasks = frozenset({"ranking"})

    @abstractmethod
    def user_vectors(self, users: torch.Tensor, history: History | None) -> torch.Tensor:
        """``(len(users), dim)`` user-tower outputs given each user's known ``history``."""

    @abstractmethod
    def item_vectors(self, items: torch.Tensor | None = None) -> torch.Tensor:
        """``(len(items), dim)`` item-tower outputs (all items if ``None``)."""

    @torch.no_grad()
    def dense_scores(self, users: np.ndarray, history: History) -> torch.Tensor:
        """Inner products with every item (for inspection; the evaluator uses :meth:`recommend`)."""
        u = torch.as_tensor(np.asarray(users), dtype=torch.long, device=self.device)
        return self.user_vectors(u, history) @ self.item_vectors().T

    def recommend(self, users, history, k):
        return top_k_unseen(self.dense_scores(users, history), users, history, k)


class HistoryTwoTower(TwoTower):
    """User tower = MLP(mean of the last ``max_len`` history-item embeddings ++ user side features).

    Training examples are every train interaction after a user's first: the
    preceding ``max_len`` items (and the user's features) predict the item.
    Inductive: no user-id embedding, so unseen users are scored from history.
    """

    inductive = True
    requires = frozenset({"timestamps"})  # the user tower reads the most recent items
    normalize = False
    dim: int
    hidden: tuple[int, ...]
    max_len: int

    def build_user_tower(self, data) -> None:
        self.n_items = data.n_items
        self.hist_emb = embedding(data.n_items, self.dim)
        self.user_feat = FeatureEmbedding(data.user_features, self.dim)
        self.user_tower = mlp([self.dim + self.user_feat.out_dim, *self.hidden, self.dim])
        h = data.history
        users = np.repeat(np.arange(h.n_users), h.lengths())
        pos = np.arange(len(h.items))
        keep = pos > h.indptr[users]
        for name, arr in (("_flat", h.items), ("_ex_users", users[keep]), ("_starts", h.indptr[users][keep]),
                          ("_ends", pos[keep])):
            self.register_buffer(name, torch.as_tensor(arr, dtype=torch.long), persistent=False)

    def encode_user(self, users: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        u = self.user_tower(torch.cat([mean_pool(self.hist_emb, seq), self.user_feat(users)], -1))
        return F.normalize(u, dim=-1) if self.normalize else u

    def user_vectors(self, users, history):
        seq = torch.as_tensor(history.sequences(users.cpu().numpy(), self.max_len), device=self.device)
        return self.encode_user(users, seq)

    def batches(self, data):
        offsets = torch.arange(self.max_len, device=self.device) - self.max_len
        for idx in minibatches(len(self._ends), self.batch_size, self.device):
            starts, ends = self._starts[idx], self._ends[idx]
            pos = ends[:, None] + offsets
            seq = torch.where(pos >= starts[:, None], self._flat[pos.clamp(min=0)], -1)
            yield self._ex_users[idx], seq, self._flat[ends]
