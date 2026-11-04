"""Benchmark definition and the data bundles handed to algorithms."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
import scipy.sparse as sp

from recsys import metrics as M
from recsys.data.base import Dataset, data_root

if TYPE_CHECKING:  # pragma: no cover
    from recsys.algorithms.base import Recommender

LOGGER = logging.getLogger(__name__)

Task = Literal["ranking", "ctr"]
Split = Literal["val", "test"]


@dataclass
class History:
    """Per-user item ids (and optional ``times``) in time order, CSR-style (``items[indptr[u]:indptr[u+1]]``)."""

    indptr: np.ndarray
    items: np.ndarray
    n_items: int
    times: np.ndarray | None = None

    @classmethod
    def from_frame(cls, df: pl.DataFrame, n_users: int, n_items: int) -> History:
        df = df.sort(["user", "timestamp"], maintain_order=True)
        counts = np.bincount(df["user"].to_numpy(), minlength=n_users)
        indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return cls(indptr, df["item"].to_numpy().astype(np.int64), n_items, df["timestamp"].to_numpy().astype(np.int64))

    @property
    def n_users(self) -> int:
        return len(self.indptr) - 1

    def lengths(self, users: np.ndarray | None = None) -> np.ndarray:
        lengths = np.diff(self.indptr)
        return lengths if users is None else lengths[users]

    def matrix(self, users: np.ndarray | None = None) -> sp.csr_matrix:
        """Binary ``(len(users), n_items)`` interaction matrix (all users if ``None``)."""
        m = sp.csr_matrix(
            (np.ones(len(self.items), dtype=np.float32), self.items, self.indptr),
            shape=(self.n_users, self.n_items),
        )
        return m if users is None else m[users]

    def sequences(self, users: np.ndarray, max_len: int) -> np.ndarray:
        """Most recent ``max_len`` items per user, right-aligned and left-padded with ``-1``."""
        return _right_aligned(self.items, self.indptr[users], self.indptr[users + 1], max_len)

    def time_sequences(self, users: np.ndarray, max_len: int) -> np.ndarray:
        """Timestamps matching :meth:`sequences`, left-padded with ``-1``."""
        return _right_aligned(self.times, self.indptr[users], self.indptr[users + 1], max_len)


def _right_aligned(values: np.ndarray, starts: np.ndarray, ends: np.ndarray, max_len: int) -> np.ndarray:
    """``values[max(start, end - max_len):end]`` per row, right-aligned and padded with ``-1``."""
    if len(values) == 0:
        return np.full((len(starts), max_len), -1, dtype=np.int64)
    lens = np.minimum(ends - starts, max_len)
    offsets = np.arange(max_len)[None, :] - (max_len - lens)[:, None]
    idx = (ends - lens)[:, None] + offsets
    return np.where(offsets >= 0, values[np.clip(idx, 0, max(len(values) - 1, 0))], -1)


@dataclass
class TrainData:
    """Everything an algorithm may use to fit.

    ``train`` is encoded. Ranking: ``user``, ``item``, ``timestamp`` sorted by
    user then time, and ``history`` holds the same interactions as a
    :class:`History`. CTR: one int column per field (id 0 = unseen in train),
    ``label``, and with behavior benchmarks a ``history`` array column of the
    user's earlier positive items (``-1`` padded). ``fields`` maps every
    categorical input (including side features) to its cardinality.
    ``user_features`` / ``item_features`` (ranking) map a side-feature name to
    an array indexed by user / item id (0 = missing). ``validate`` returns the
    benchmark's model-selection metric on the validation split.
    """

    task: Task
    train: pl.DataFrame
    n_users: int
    n_items: int
    fields: dict[str, int]
    history: History | None
    seed: int
    validate: Callable[[Recommender], float] | None = None
    user_features: dict[str, np.ndarray] = field(default_factory=dict)
    item_features: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class BenchmarkData:
    """A prepared benchmark: encoded splits plus helpers for evaluation.

    For ranking, val/test carry a boolean ``target`` column; non-target rows
    are fold-in history (see :mod:`recsys.protocol`).
    """

    benchmark: Benchmark
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame
    n_users: int
    n_items: int
    fields: dict[str, int]
    user_features: dict[str, np.ndarray] = field(default_factory=dict)
    item_features: dict[str, np.ndarray] = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    def _history(self, key: str, frames: list[pl.DataFrame]) -> History:
        if key not in self._cache:
            self._cache[key] = History.from_frame(
                pl.concat([f.select("user", "item", "timestamp") for f in frames]), self.n_users, self.n_items
            )
        return self._cache[key]

    def known(self, split: Split) -> History:
        """Interactions visible when scoring ``split``: train, earlier splits, and the split's fold-in rows."""
        own = getattr(self, split).filter(~pl.col("target"))
        frames = [self.train, own] if split == "val" else [self.train, self.val, own]
        return self._history(f"known-{split}", frames)

    def targets(self, split: Split) -> History:
        return self._history(f"targets-{split}", [getattr(self, split).filter(pl.col("target"))])

    def train_data(self, seed: int, validate: Callable[[Recommender], float] | None = None) -> TrainData:
        ranking = self.benchmark.task == "ranking"
        history = self._history("train", [self.train]) if ranking else None
        return TrainData(
            self.benchmark.task, self.train, self.n_users, self.n_items, self.fields, history, seed, validate,
            self.user_features, self.item_features,
        )

    @property
    def stats(self) -> tuple[int, int, int, int, int]:
        if self.benchmark.task == "ranking":
            sizes = [self.train.height, *(getattr(self, s)["target"].sum() for s in ("val", "test"))]
        else:
            sizes = [self.train.height, self.val.height, self.test.height]
        return (self.n_users, self.n_items, *map(int, sizes))


@dataclass(frozen=True)
class Benchmark:
    """A pinned evaluation protocol: dataset x preprocessing x split x task x metrics.

    The first metric selects models (early stopping) and must be higher-is-better.

    * Ranking: ``user_fields`` / ``item_fields`` name side-feature columns of
      the dataset's ``users`` / ``items`` tables offered to algorithms.
    * CTR: preprocessing must produce a ``label`` column; ``fields`` lists the
      categorical inputs (from events or side tables); ``history > 0`` adds
      each row's last ``history`` positive items of the same user strictly
      before the row's timestamp (from train for train/val rows, from
      train + val for test rows). Field values seen fewer than ``min_count``
      times in train map to the unseen id 0.

    ``stats`` pins ``(n_users, n_items, n_train, n_val_targets, n_test_targets)``
    (CTR: field cardinalities of user/item and row counts); ``prepare`` checks it.
    ``description``, ``leakage`` and ``reference`` document the protocol.
    """

    name: str
    dataset: Dataset
    task: Task
    split: Callable[[pl.DataFrame], tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]]
    metrics: tuple[str, ...]
    preprocess: tuple[Callable[[pl.DataFrame], pl.DataFrame], ...] = ()
    fields: tuple[str, ...] = ()
    user_fields: tuple[str, ...] = ()
    item_fields: tuple[str, ...] = ()
    history: int = 0
    min_count: int = 1
    stats: tuple[int, ...] | None = field(default=None, repr=False, compare=False)
    description: str = field(default="", repr=False, compare=False)
    leakage: str = field(default="", repr=False, compare=False)
    reference: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        for m in self.metrics:
            if self.task == "ctr":
                if m not in M.CTR:
                    raise ValueError(f"{self.name}: unknown CTR metric {m!r}")
            else:
                M.parse(m)
        if self.history and not {"user", "item"} <= set(self.fields):
            raise ValueError(f"{self.name}: behavior history needs 'user' and 'item' fields")

    @property
    def version(self) -> str:
        """Hash of the full definition and the dataset's pinned checksums."""
        blob = repr(self) + repr(self.dataset.pins())
        return hashlib.sha256(blob.encode()).hexdigest()[:10]

    @property
    def provides(self) -> frozenset[str]:
        """Inputs and conditions this benchmark offers; algorithms declare what they ``require``."""
        out = set()
        if self.dataset.has_time:
            out.add("timestamps")
        if self.user_fields:
            out.add("user_features")
        if self.item_fields:
            out.add("item_features")
        if self.history:
            out.add("behavior")
        if getattr(self.split, "holds_out_users", False):
            out.add("cold_users")
        return frozenset(out)

    def prepare(self) -> BenchmarkData:
        data = _prepare(self)
        if self.stats is not None and data.stats != tuple(self.stats):
            raise RuntimeError(f"{self.name}: prepared stats {data.stats} differ from pinned {self.stats}")
        return data


# ------------------------------------------------------------------ preparation


@functools.lru_cache(maxsize=2)
def _prepare(b: Benchmark) -> BenchmarkData:
    cache = data_root() / "prepared" / f"{b.name}-{b.version}"
    if (cache / "meta.json").exists():
        return _load(b, cache)
    LOGGER.info("Preparing benchmark %s (%s)", b.name, b.version)
    raw = b.dataset.load()
    data = _prepare_ranking(b, raw) if b.task == "ranking" else _prepare_ctr(b, raw)
    _save(data, cache)
    LOGGER.info("%s: stats %s", b.name, data.stats)
    return data


def _vocab_ids(column: pl.Series, vocab: pl.Series, start: int) -> pl.Series:
    ids = pl.int_range(start, start + len(vocab), eager=True)
    return column.replace_strict(vocab, ids, default=0, return_dtype=pl.Int64).fill_null(0)


def _side_features(table: Any, key: str, vocab: pl.Series, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Encode side-feature columns (1..V, 0 = missing) into arrays indexed by encoded id."""
    if not names:
        return {}
    if table is None or not set(names) <= set(table.columns):
        raise ValueError(f"side features {names} not found in the dataset's {key} table")
    table = table.unique(key).filter(pl.col(key).is_in(vocab.implode()))
    rows = _vocab_ids(table[key], vocab, 0).to_numpy()
    out = {}
    for name in names:
        values = table[name].drop_nulls().unique().sort()
        arr = np.zeros(len(vocab), dtype=np.int64)
        arr[rows] = _vocab_ids(table[name], values, 1).to_numpy()
        out[name] = arr
    return out


def _events(b: Benchmark, raw: Any) -> pl.DataFrame:
    """Preprocessed events; the loader's reference is dropped so large frames can be freed early."""
    df, raw.events = raw.events, None
    for step in b.preprocess:
        df = step(df)
    return df


def _prepare_ranking(b: Benchmark, raw: Any) -> BenchmarkData:
    df = _events(b, raw)
    df = df.select("user", "item", "timestamp", *(["split"] if "split" in df.columns else []))
    train, val, test = b.split(df)
    del df
    val, test = (s if "target" in s.columns else s.with_columns(pl.lit(True).alias("target")) for s in (val, test))
    everything = pl.concat([train.select("user", "item"), val.select("user", "item"), test.select("user", "item")])
    users, items = everything["user"].unique().sort(), everything["item"].unique().sort()

    def encode(s: pl.DataFrame) -> pl.DataFrame:
        s = s.with_columns(_vocab_ids(s["user"], users, 0).alias("user"), _vocab_ids(s["item"], items, 0).alias("item"))
        return s.sort(["user", "timestamp"], maintain_order=True)

    user_features = _side_features(raw.users, "user", users, b.user_fields)
    item_features = _side_features(raw.items, "item", items, b.item_fields)
    fields = {"user": len(users), "item": len(items)}
    fields |= {k: int(v.max()) + 1 for k, v in (user_features | item_features).items()}
    return BenchmarkData(
        b, encode(train), encode(val), encode(test), len(users), len(items), fields, user_features, item_features
    )


def _prepare_ctr(b: Benchmark, raw: Any) -> BenchmarkData:
    df = _events(b, raw)
    for key, table in (("user", raw.users), ("item", raw.items)):
        wanted = [f for f in b.fields if f not in df.columns and table is not None and f in table.columns]
        if wanted:
            df = df.join(table.select(key, *wanted).unique(key), on=key, how="left")
    missing = [f for f in b.fields if f not in df.columns]
    if missing:
        raise ValueError(f"{b.name}: fields {missing} not found in events or side tables")
    keep = [*b.fields, "label", *(["timestamp"] if b.history else [])]
    needed = [*dict.fromkeys([*keep, *(c for c in ("user", "timestamp", "split") if c in df.columns)])]
    splits = [s.select(keep) for s in b.split(df.select(needed))]
    del df
    fields: dict[str, int] = {}
    for f in b.fields:
        counts = splits[0][f].drop_nulls().value_counts()
        vocab = counts.filter(pl.col("count") >= b.min_count)[f].sort()
        splits = [s.with_columns(_vocab_ids(s[f], vocab, 1).alias(f)) for s in splits]
        fields[f] = len(vocab) + 1
    splits = [s.with_columns(pl.col("label").cast(pl.Float32)) for s in splits]
    if b.history:
        train, val, test = splits
        splits = [
            _with_behavior(train, train, b.history),
            _with_behavior(val, train, b.history),
            _with_behavior(test, pl.concat([train, val]), b.history),
        ]
    train, val, test = splits
    return BenchmarkData(b, train, val, test, fields.get("user", 0), fields.get("item", 0), fields)


def _with_behavior(rows: pl.DataFrame, source: pl.DataFrame, length: int) -> pl.DataFrame:
    """Add ``history``: each row's user's last ``length`` positive items in ``source`` strictly before it."""
    src = source.filter((pl.col("label") > 0.5) & (pl.col("user") > 0)).sort(["user", "timestamp"], maintain_order=True)
    key = lambda u, t: (u.astype(np.int64) << 32) | t.astype(np.int64)  # noqa: E731
    src_key = key(src["user"].to_numpy(), src["timestamp"].to_numpy())
    users, ts = rows["user"].to_numpy(), rows["timestamp"].to_numpy()
    ends = np.searchsorted(src_key, key(users, ts), side="left")
    starts = np.searchsorted(src_key, key(users, np.zeros_like(ts)), side="left")
    hist = _right_aligned(src["item"].to_numpy(), starts, ends, length).astype(np.int32)
    return rows.drop("timestamp").with_columns(pl.Series("history", hist))


# ---------------------------------------------------------------- on-disk cache


def _save(data: BenchmarkData, path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        getattr(data, split).write_parquet(path / f"{split}.parquet")
    for kind in ("user_features", "item_features"):
        features = getattr(data, kind)
        if features:
            pl.DataFrame(features).write_parquet(path / f"{kind}.parquet")
    meta = {"n_users": data.n_users, "n_items": data.n_items, "fields": data.fields}
    (path / "meta.json").write_text(json.dumps(meta))  # written last: marks the cache complete


def _load(b: Benchmark, path) -> BenchmarkData:
    meta = json.loads((path / "meta.json").read_text())
    frames = [pl.read_parquet(path / f"{s}.parquet") for s in ("train", "val", "test")]
    features = []
    for kind in ("user_features", "item_features"):
        p = path / f"{kind}.parquet"
        features.append({c: s.to_numpy() for c, s in pl.read_parquet(p).to_dict().items()} if p.exists() else {})
    return BenchmarkData(b, *frames, meta["n_users"], meta["n_items"], meta["fields"], *features)
