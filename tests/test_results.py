"""Reproducibility and results (criteria E1-E6)."""

import hashlib
import logging
import shutil
import subprocess

import numpy as np
import polars as pl
import pytest
from scipy import stats

import recsys
from recsys import BENCHMARKS, store
from recsys.benchmarks.base import _prepare
from recsys.data import data_root

CPU = {"epochs": 3, "device": "cpu"}


def test_e1_same_seed_same_result_different_seed_different_result():
    a = recsys.run("bpr", "synthetic-ranking", seed=0, params=CPU, results_dir=None)
    b = recsys.run("bpr", "synthetic-ranking", seed=0, params=CPU, results_dir=None)
    c = recsys.run("bpr", "synthetic-ranking", seed=1, params=CPU, results_dir=None)
    metrics = BENCHMARKS["synthetic-ranking"].metrics
    assert [a[m] for m in metrics] == [b[m] for m in metrics]
    assert [a[m] for m in metrics] != [c[m] for m in metrics]


def test_e2_split_does_not_depend_on_run_seed():
    digests = []
    for seed in (0, 1):
        recsys.harness.seed_everything(seed)
        _prepare.cache_clear()
        shutil.rmtree(data_root() / "prepared", ignore_errors=True)  # force a fresh split
        test = BENCHMARKS["synthetic-ctr"].prepare().test
        digests.append(hashlib.sha256(test.write_ndjson().encode()).hexdigest())
    assert digests[0] == digests[1]


def test_e3_result_rows_carry_full_provenance(tmp_path):
    recsys.run("popularity", "synthetic-ranking", results_dir=tmp_path)
    row = pl.read_parquet(tmp_path / "synthetic-ranking.parquet").row(0, named=True)
    for key in ("benchmark", "version", "algo", "params", "params_hash", "seed", "fit_s", "eval_s",
                "peak_rss_mb", "peak_device_mb", "device", "timestamp", "code_sha", "dirty", "versions", "run_id"):
        assert row[key] is not None, key
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert row["code_sha"] == head and "torch" in row["versions"]


def test_e4_report_never_merges_versions(tmp_path, caplog):
    rec = recsys.run("popularity", "synthetic-ranking", results_dir=tmp_path)
    store.append(tmp_path, {**rec, "version": "other", "run_id": "x"})
    with caplog.at_level(logging.WARNING):
        table = store.summary(tmp_path)["synthetic-ranking"]
    assert table.height == 2 and "versions" in caplog.text


def test_e5_paired_test_matches_scipy(tmp_path):
    for seed in (0, 1):
        for algo in ("ease", "popularity"):
            recsys.run(algo, "synthetic-ranking", seed=seed, results_dir=tmp_path, per_user=True)
    result = store.compare(tmp_path, "synthetic-ranking", "ease", "popularity")
    assert result["unit"] == "user"
    df = store.load(tmp_path)
    detail = {}
    for algo in ("ease", "popularity"):
        ids = df.filter(pl.col("algo") == algo).sort("seed")["run_id"]
        detail[algo] = np.concatenate([
            pl.read_parquet(tmp_path / "detail" / "synthetic-ranking" / f"{i}.parquet")["ndcg@10"].to_numpy()
            for i in ids
        ])
    assert result["t_test_p"] == pytest.approx(stats.ttest_rel(detail["ease"], detail["popularity"]).pvalue)


def test_e6_per_user_detail_matches_aggregate(tmp_path):
    rec = recsys.run("ease", "synthetic-ranking", results_dir=tmp_path, per_user=True)
    detail = pl.read_parquet(tmp_path / "detail" / "synthetic-ranking" / f"{rec['run_id']}.parquet")
    data = BENCHMARKS["synthetic-ranking"].prepare()
    assert detail.height == int((data.targets("test").lengths() > 0).sum())
    assert detail["ndcg@10"].mean() == pytest.approx(rec["ndcg@10"])
