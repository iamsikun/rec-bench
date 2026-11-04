"""Preprocessing steps and splits.

Every step is a frozen dataclass so that its ``repr`` is stable and feeds the
benchmark version hash. Steps map an events DataFrame to an events DataFrame;
splits map it to ``(train, val, test)``. A split may add a boolean ``target``
column to val/test: rows with ``target=False`` are fold-in history, visible
when evaluating that split but never trained on or scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

# ---------------------------------------------------------------- preprocessing


@dataclass(frozen=True)
class MinValue:
    """Keep rows with ``column >= threshold`` (e.g. ratings >= 4 as implicit positives)."""

    column: str
    threshold: float

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col(self.column) >= self.threshold)


@dataclass(frozen=True)
class Binarize:
    """Add a binary ``label`` column: ``column >= threshold``."""

    column: str
    threshold: float

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns((pl.col(self.column) >= self.threshold).cast(pl.Float32).alias("label"))


@dataclass(frozen=True)
class Dedup:
    """Keep only the earliest interaction of each (user, item) pair."""

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.sort("timestamp", maintain_order=True).unique(["user", "item"], keep="first", maintain_order=True)


@dataclass(frozen=True)
class MinCount:
    """Keep rows whose ``column`` value occurs at least ``k`` times (one pass, not iterative)."""

    column: str
    k: int

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.len().over(self.column) >= self.k)


@dataclass(frozen=True)
class LogSquareBucket:
    """Discretize numeric columns as BARS/FuxiCTR do for Criteo: ``x > 2 -> floor(ln(x)^2)``, else ``x``.

    Missing values are first filled with ``fill`` (BARS uses 0).
    """

    columns: tuple[str, ...]
    fill: int = 0

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        def bucket(c: str) -> pl.Expr:
            x = pl.col(c).fill_null(self.fill).cast(pl.Float64)
            return pl.when(x > 2).then((x.log() ** 2).floor()).otherwise(x).cast(pl.Int32).alias(c)

        return df.with_columns([bucket(c) for c in self.columns])


@dataclass(frozen=True)
class DateParts:
    """Add ``hour``, ``weekday`` (0 = Sunday) and ``weekend`` columns from Unix-second ``timestamp`` (UTC)."""

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        t = pl.from_epoch("timestamp", time_unit="s")
        weekday = (t.dt.weekday() % 7).cast(pl.Int8)
        return df.with_columns(t.dt.hour().cast(pl.Int8).alias("hour"), weekday.alias("weekday"),
                               weekday.is_in([0, 6]).cast(pl.Int8).alias("weekend"))


@dataclass(frozen=True)
class KCore:
    """Iteratively drop users and items with fewer than ``k`` interactions."""

    k: int

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        while True:
            n = df.height
            df = df.filter((pl.len().over("user") >= self.k) & (pl.len().over("item") >= self.k))
            if df.height == n:
                return df


# ----------------------------------------------------------------------- splits


def _partition(df: pl.DataFrame, role: pl.Expr) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = df.with_columns(role.alias("_role"))
    return tuple(df.filter(pl.col("_role") == r).drop("_role") for r in range(3))  # type: ignore[return-value]


@dataclass(frozen=True)
class LeaveLastOut:
    """Per user, in time order: last event -> test, second to last -> val, rest -> train.

    Users with fewer than three events stay entirely in train. Equal
    timestamps keep the source file's order (stable sort).
    """

    def __call__(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        df = df.sort(["user", "timestamp"], maintain_order=True)
        n = pl.len().over("user")
        from_end = n - 1 - pl.int_range(pl.len()).over("user")
        role = pl.when(n < 3).then(0).when(from_end == 0).then(2).when(from_end == 1).then(1).otherwise(0)
        return _partition(df, role)


@dataclass(frozen=True)
class TemporalSplit:
    """Global time cutoffs: the last ``test`` fraction of events -> test, the ``val`` before it -> val."""

    val: float = 0.1
    test: float = 0.1

    def __call__(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        ts = df["timestamp"]
        t_val = ts.quantile(1 - self.val - self.test, "nearest")
        t_test = ts.quantile(1 - self.test, "nearest")
        role = pl.when(pl.col("timestamp") >= t_test).then(2).when(pl.col("timestamp") >= t_val).then(1).otherwise(0)
        return _partition(df, role)


@dataclass(frozen=True)
class RandomSplit:
    """Row-level random split with a pinned seed."""

    val: float = 0.1
    test: float = 0.1
    seed: int = 0

    def __call__(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        u = np.random.default_rng(self.seed).random(df.height)
        role = np.where(u < self.test, 2, np.where(u < self.test + self.val, 1, 0))
        return _partition(df, pl.Series(role))


@dataclass(frozen=True)
class HeldOutUsers:
    """Strong generalization (Liang et al., WWW 2018): evaluate on users never seen in training.

    ``n_val`` / ``n_test`` random users are held out. Their events on items
    that occur in train are split per user at random: a ``1 - fold_in``
    fraction (``int(n * (1 - fold_in))`` events, only for users with at least
    ``min_events``) becomes the target, the rest is fold-in history.
    """

    n_val: int = 10_000
    n_test: int = 10_000
    fold_in: float = 0.8
    min_events: int = 5
    seed: int = 98765
    holds_out_users = True

    def __call__(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        rng = np.random.default_rng(self.seed)
        users = rng.permutation(df["user"].unique().sort().to_numpy())
        test_users, val_users = users[: self.n_test], users[self.n_test : self.n_test + self.n_val]
        held = pl.col("user").is_in(np.concatenate([test_users, val_users]))
        train = df.filter(~held)
        rest = df.filter(held & pl.col("item").is_in(train["item"].unique().implode()))
        rest = rest.with_columns(pl.Series("_key", rng.random(rest.height))).sort(["user", "_key"])
        n = pl.len().over("user")
        n_target = pl.when(n >= self.min_events).then((n * (1 - self.fold_in)).floor()).otherwise(0)
        rest = rest.with_columns((pl.int_range(pl.len()).over("user") < n_target).alias("target")).drop("_key")
        return (
            train,
            rest.filter(pl.col("user").is_in(val_users)),
            rest.filter(pl.col("user").is_in(test_users)),
        )


@dataclass(frozen=True)
class Predefined:
    """Use the split published with the dataset (an events column ``split`` in {train, val, test}).

    If the source publishes no validation rows, a ``val`` fraction of train
    rows is drawn at random (pinned seed) for model selection.
    """

    val: float = 0.1
    seed: int = 0

    def __call__(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        if not (df["split"] == "val").any():
            carve = np.random.default_rng(self.seed).random(df.height) < self.val
            df = df.with_columns(
                pl.when((pl.col("split") == "train") & pl.Series(carve)).then(pl.lit("val"))
                .otherwise(pl.col("split")).alias("split")
            )
        return tuple(df.filter(pl.col("split") == s).drop("split") for s in ("train", "val", "test"))  # type: ignore[return-value]
