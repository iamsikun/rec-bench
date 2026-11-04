"""Small generated dataset for tests and smoke runs (no download)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from recsys.data.base import Dataset, Interactions


@dataclass(frozen=True)
class Synthetic(Dataset):
    """Latent-factor interactions with a learnable signal.

    ``events``: user, item, timestamp, rating (1-5). ``users``: user, segment.
    ``items``: item, category. Users drift through item space over time so
    order carries signal for sequential models.
    """

    n_users: int = 300
    n_items: int = 200
    events_per_user: int = 30
    seed: int = 0
    dirname = "synthetic"

    def files_present(self, root: Path) -> bool:
        return True

    def download(self, root: Path) -> None:  # pragma: no cover - nothing to fetch
        pass

    def read(self, root: Path) -> Interactions:
        rng = np.random.default_rng(self.seed)
        dim = 8
        user_f = rng.normal(size=(self.n_users, dim))
        item_f = rng.normal(size=(self.n_items, dim))
        popularity = 0.3 * np.log1p(rng.zipf(1.5, self.n_items).clip(max=1000))
        state = user_f.copy()
        steps = []
        for t in range(self.events_per_user):  # vectorized over users; Gumbel-max sampling
            logits = 0.4 * state @ item_f.T + popularity
            item = np.argmax(logits + rng.gumbel(size=logits.shape), axis=1)
            rating = np.clip(np.round(3 + (item_f[item] * user_f).sum(1) / 2), 1, 5).astype(np.int64)
            steps.append(pl.DataFrame({"user": np.arange(self.n_users), "item": item,
                                       "timestamp": 1_600_000_000 + 3600 * t, "rating": rating}))
            state = 0.7 * state + 0.3 * item_f[item]
        events = pl.concat(steps).with_columns(pl.col("timestamp").cast(pl.Int64)).sort(["user", "timestamp"])
        users = pl.DataFrame({"user": np.arange(self.n_users), "segment": (user_f[:, 0] > 0).astype(np.int64)})
        items = pl.DataFrame({"item": np.arange(self.n_items), "category": np.argmax(item_f[:, :4], axis=1)})
        return Interactions(events=events, users=users, items=items)
