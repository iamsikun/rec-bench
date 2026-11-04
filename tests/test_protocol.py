"""Benchmark definition and protocol validity (criteria B1-B8, B10, B11)."""

import dataclasses
import inspect
import os
import subprocess
import sys

import numpy as np
import polars as pl
import pytest
import torch
from helpers import Constant, FrameDataset, Spy, ranking_benchmark

import recsys
from recsys import BENCHMARKS, Recommender
from recsys.benchmarks.base import _prepare
from recsys.data import Synthetic
from recsys.evaluate import evaluate, evaluate_ranking
from recsys.protocol import (
    Binarize,
    Dedup,
    HeldOutUsers,
    KCore,
    LeaveLastOut,
    Predefined,
    RandomSplit,
    TemporalSplit,
)


def test_b1_benchmarks_are_frozen_and_not_overridable():
    b = BENCHMARKS["synthetic-ranking"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.metrics = ("hr@1",)
    for fn in (recsys.run, recsys.run_all):
        assert not {"split", "metrics", "preprocess", "fields"} & set(inspect.signature(fn).parameters)


def test_b2_version_changes_with_every_field():
    b = BENCHMARKS["synthetic-ranking"]
    variants = [
        dataclasses.replace(b, preprocess=(Dedup(), KCore(4))),
        dataclasses.replace(b, metrics=("ndcg@10",)),
        dataclasses.replace(b, split=TemporalSplit()),
        dataclasses.replace(b, dataset=Synthetic(seed=1)),
        dataclasses.replace(b, user_fields=()),
        dataclasses.replace(b, name="other"),
    ]
    versions = {v.version for v in variants} | {b.version}
    assert len(versions) == len(variants) + 1
    assert dataclasses.replace(b, description="docs only").version == b.version


def test_b2_version_is_stable_across_processes():
    code = "import recsys; print(recsys.BENCHMARKS['ml-1m-loo'].version)"
    env = {**os.environ, "PYTHONHASHSEED": "12345"}  # a different hash seed than this process
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert out.stdout.strip() == BENCHMARKS["ml-1m-loo"].version


def test_b2_version_includes_dataset_pins():
    b = BENCHMARKS["ml-1m-loo"]

    class Repinned(type(b.dataset)):
        def pins(self):
            return ("different",)

    assert dataclasses.replace(b, dataset=Repinned("1m")).version != b.version


def test_b3_pinned_stats_are_enforced():
    b = BENCHMARKS["synthetic-ranking"]
    assert b.prepare().stats == tuple(b.stats)
    drifted = dataclasses.replace(b, name="drifted", dataset=Synthetic(seed=7))
    with pytest.raises(RuntimeError, match="differ from pinned"):
        drifted.prepare()


def test_b3_every_builtin_benchmark_pins_stats():
    assert [n for n, b in BENCHMARKS.items() if b.stats is None] == []


def test_b4_test_split_is_unreachable_during_fit():
    data = BENCHMARKS["synthetic-ranking"].prepare()
    spy = Spy()
    spy.fit(data.train_data(0, lambda m: 0.0))
    test = set(zip(data.test["user"].to_list(), data.test["item"].to_list()))
    val = set(zip(data.val["user"].to_list(), data.val["item"].to_list()))
    assert spy.seen.isdisjoint(test) and spy.seen.isdisjoint(val)


class Oracle(Recommender):
    """Puts a fixed set of (user, item) pairs on top."""

    tasks = frozenset({"ranking"})

    def __init__(self, pairs, n_items):
        self.pairs, self.n_items = pairs, n_items

    def fit(self, data):
        pass

    def score(self, users, history):
        s = torch.rand(len(users), history.n_items)
        rows = {u: r for r, u in enumerate(users)}
        for u, i in self.pairs:
            if u in rows:
                s[rows[u], i] = 10.0
        return s


def test_b4_validation_uses_val_targets_only():
    data = BENCHMARKS["synthetic-ranking"].prepare()
    oracle = Oracle(set(zip(data.test["user"].to_list(), data.test["item"].to_list())), data.n_items)
    assert evaluate(oracle, data, "test")["recall@10"] == 1.0
    assert evaluate(oracle, data, "val")["recall@10"] < 0.5


def _invariants(b, train, val, test):
    if isinstance(b.split, LeaveLastOut):
        for frame in (val, test):
            assert frame["user"].n_unique() == frame.height
        cols = ["user", "item", "timestamp"]
        last = pl.concat([f.select(cols) for f in (train, val, test)]).group_by("user").agg(pl.col("timestamp").max())
        joined = test.join(last, on="user", suffix="_max")
        assert (joined["timestamp"] == joined["timestamp_max"]).all()
        tv = val.join(train.group_by("user").agg(pl.col("timestamp").max()), on="user", suffix="_train")
        assert (tv["timestamp"] >= tv["timestamp_train"]).all()
    elif isinstance(b.split, TemporalSplit):
        assert train["timestamp"].max() <= val["timestamp"].min() <= val["timestamp"].max() <= test["timestamp"].min()
    elif isinstance(b.split, HeldOutUsers):
        users = [set(f["user"].to_list()) for f in (train, val, test)]
        assert not (users[0] & users[1] or users[0] & users[2] or users[1] & users[2])


@pytest.mark.parametrize("name", [n for n in BENCHMARKS if n.startswith("synthetic")])
def test_b5_split_invariants_fixture(name):
    b = BENCHMARKS[name]
    d = b.prepare()
    _invariants(b, d.train, d.val, d.test)
    if b.task == "ctr":
        assert d.train.height + d.val.height + d.test.height == Synthetic().load().events.height


def test_b5_random_and_temporal_partition_rows():
    df = pl.DataFrame({"user": np.arange(1000) % 7, "item": np.arange(1000), "timestamp": np.arange(1000)})
    for split in (TemporalSplit(0.1, 0.1), RandomSplit(0.1, 0.1)):
        parts = split(df)
        assert sorted(pl.concat(parts)["item"].to_list()) == list(range(1000))
        _invariants(ranking_benchmark(split=split), *parts)


def test_b5_predefined_split():
    df = pl.DataFrame({"user": [0, 0, 1, 1], "item": [1, 2, 3, 4], "timestamp": [0] * 4,
                       "split": ["train", "test", "train", "val"]})
    train, val, test = Predefined()(df)
    assert (train.height, val.height, test.height) == (2, 1, 1)


@pytest.mark.network
@pytest.mark.parametrize("name", [n for n in BENCHMARKS if not n.startswith("synthetic")])
def test_b5_split_invariants_real(name):
    b = BENCHMARKS[name]
    d = b.prepare()
    if "timestamp" in d.train.columns:
        _invariants(b, d.train, d.val.filter(pl.col("target")), d.test.filter(pl.col("target")))


def test_b6_ctr_value_only_in_test_maps_to_unseen():
    events = pl.DataFrame({
        "user": [1, 1, 2, 2, 3], "item": [1, 2, 1, 2, 99], "timestamp": [1, 2, 3, 4, 5], "rating": [5, 1, 5, 1, 5],
    })
    ds = FrameDataset(tag="b6", frames={"events": events})
    b = recsys.Benchmark("b6", ds, "ctr", TemporalSplit(0.2, 0.2), ("auc",), (Binarize("rating", 4),),
                         fields=("user", "item"))
    d = b.prepare()
    assert d.test["item"].to_list() == [2, 0] and d.test["user"].to_list() == [0, 0]  # only user 1 is in train


def test_b6_popularity_counts_known_interactions_only():
    data = BENCHMARKS["synthetic-ranking"].prepare()
    assert len(data.known("val").items) == data.train.height
    assert len(data.known("test").items) == data.train.height + data.val.height


def test_b6_b7_every_benchmark_documents_leakage_and_a_temporal_one_exists():
    assert all(b.leakage for b in BENCHMARKS.values())
    temporal = [b for b in BENCHMARKS.values() if b.task == "ranking" and isinstance(b.split, TemporalSplit)]
    assert temporal


def test_b8_scorer_sees_full_catalog_rows_in_bounded_batches():
    data = BENCHMARKS["synthetic-ranking"].prepare()
    spy = Spy()
    spy.fit(data.train_data(0))
    evaluate_ranking(spy, data.known("test"), data.targets("test"), ("hr@10",), batch_size=64)
    assert max(spy.batches) <= 64 and sum(spy.batches) == data.targets("test").lengths().astype(bool).sum()


def test_b10_constant_scores_earn_random_expectation():
    ds = Synthetic(n_users=4000, n_items=600, events_per_user=4, seed=3)
    b = ranking_benchmark(ds, metrics=("hr@100", "hr@10"))
    data = b.prepare()
    known, targets = data.known("test"), data.targets("test")
    result = evaluate_ranking(Constant(), known, targets, b.metrics)
    users = result["user"].to_numpy()
    for k in (100, 10):
        expected = np.mean(k / (data.n_items - known.lengths(users)))
        assert result[f"hr@{k}"].mean() == pytest.approx(expected, rel=0.1)


def test_b11_oracle_and_history_mask():
    data = BENCHMARKS["synthetic-ranking"].prepare()
    oracle = Oracle(set(zip(data.test["user"].to_list(), data.test["item"].to_list())), data.n_items)
    assert evaluate_ranking(oracle, data.known("test"), data.targets("test"), ("hr@1",))["hr@1"].mean() == 1.0

    known = data.known("test")

    class HistoryFirst(Recommender):
        tasks = frozenset({"ranking"})
        sign = 1.0

        def fit(self, data):
            pass

        def score(self, users, history):
            base = torch.as_tensor(np.random.default_rng(1).random((len(users), history.n_items))).float()
            boost = torch.as_tensor(history.matrix(users).toarray()) * 100 * self.sign
            return base + boost

    low = type("HistoryLast", (HistoryFirst,), {"sign": -1.0})
    a = evaluate_ranking(HistoryFirst(), known, data.targets("test"), ("ndcg@10",))
    b = evaluate_ranking(low(), known, data.targets("test"), ("ndcg@10",))
    assert a["ndcg@10"].to_list() == b["ndcg@10"].to_list()


def test_prepare_cache_roundtrip():
    b = BENCHMARKS["synthetic-ctr"]
    first = b.prepare()
    _prepare.cache_clear()
    second = b.prepare()
    assert first.train.equals(second.train) and first.fields == second.fields
