import numpy as np
import polars as pl
import pytest
import torch

from recsys import metrics as M
from recsys.benchmarks import BENCHMARKS, History
from recsys.protocol import Dedup, KCore, LeaveLastOut, RandomSplit, TemporalSplit


def test_ranking_metrics_hand_computed():
    hits = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    n_rel = torch.tensor([1.0, 2.0])
    assert M.hr(hits, n_rel, 3).tolist() == [1.0, 0.0]
    assert M.recall(hits, n_rel, 1).tolist() == [0.0, 0.0]
    assert M.mrr(hits, n_rel, 3).tolist() == pytest.approx([0.5, 0.0])
    assert M.ndcg(hits, n_rel, 3).tolist() == pytest.approx([1 / np.log2(3), 0.0])
    # Two relevant items both at the top: perfect NDCG.
    assert M.ndcg(torch.tensor([[1.0, 1.0, 0.0]]), torch.tensor([2.0]), 3).item() == pytest.approx(1.0)


def test_ctr_metrics():
    y = np.array([0, 0, 1, 1])
    assert M.auc(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert M.logloss(y, np.array([0.5] * 4)) == pytest.approx(np.log(2))


def _events():
    return pl.DataFrame({
        "user": [1, 1, 1, 1, 2, 2, 3],
        "item": [10, 11, 12, 10, 10, 11, 12],
        "timestamp": [1, 2, 3, 4, 1, 2, 1],
    })


def test_dedup_keeps_first_interaction():
    out = Dedup()(_events())
    assert out.filter(pl.col("user") == 1)["timestamp"].to_list() == [1, 2, 3]


def test_kcore_is_iterative():
    out = KCore(2)(_events())
    assert set(out["user"].to_list()) == {1, 2}
    assert set(out["item"].to_list()) == {10, 11}


def test_leave_last_out():
    train, val, test = LeaveLastOut()(Dedup()(_events()))
    assert test.select("user", "item").rows() == [(1, 12)]
    assert val.select("user", "item").rows() == [(1, 11)]
    assert train.height == 4  # user 1 keeps one event; users 2 and 3 have < 3 events and stay whole


def test_splits_partition_rows():
    df = pl.DataFrame({"user": np.arange(1000) % 7, "item": np.arange(1000), "timestamp": np.arange(1000)})
    for split in (TemporalSplit(0.1, 0.1), RandomSplit(0.1, 0.1)):
        train, val, test = split(df)
        assert train.height + val.height + test.height == 1000
        assert 50 < test.height < 150
    train, _, test = TemporalSplit(0.1, 0.1)(df)
    assert train["timestamp"].max() < test["timestamp"].min()


def test_history_sequences_right_aligned():
    df = pl.DataFrame({"user": [0, 0, 0, 2], "item": [5, 6, 7, 8], "timestamp": [3, 1, 2, 1]})
    h = History.from_frame(df, n_users=3, n_items=10)
    seqs = h.sequences(np.array([0, 1, 2]), 2)
    assert seqs.tolist() == [[7, 5], [-1, -1], [-1, 8]]  # user 0 in time order: 6, 7, 5
    assert h.matrix(np.array([0])).toarray()[0, [5, 6, 7]].tolist() == [1, 1, 1]


def test_benchmark_version_is_stable_and_specific():
    a, b = BENCHMARKS["ml-1m-loo"], BENCHMARKS["ml-1m-temporal"]
    assert a.version == BENCHMARKS["ml-1m-loo"].version
    assert a.version != b.version


def test_ctr_fields_encoded_from_train():
    data = BENCHMARKS["synthetic-ctr"].prepare()
    for name, size in data.fields.items():
        assert data.train[name].min() >= 1  # 0 is reserved for values unseen in train
        assert data.test[name].max() < size
