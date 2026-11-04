"""Efficiency sanity bounds on real data (criteria F1-F3); run with ``-m perf`` on a GPU machine."""

import os
import resource
import shutil
import sys
import time

import pytest

import recsys
from recsys import BENCHMARKS
from recsys.benchmarks.base import _prepare
from recsys.data import data_root
from recsys.evaluate import evaluate_ranking

pytestmark = pytest.mark.perf
CACHE = data_root()  # the real cache: raw downloads are reused, prepared splits are rebuilt


def _rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def _cold_prepare_seconds(name: str) -> float:
    b = BENCHMARKS[name]
    shutil.rmtree(CACHE / "prepared" / f"{b.name}-{b.version}", ignore_errors=True)
    _prepare.cache_clear()
    start = time.perf_counter()
    b.prepare()
    return time.perf_counter() - start


@pytest.mark.parametrize(("name", "bound"), [("ml-1m-loo", 10), ("ml-20m-strong", 90)])
def test_f2_prepare_bounds(name, bound):
    assert _cold_prepare_seconds(name) <= bound


@pytest.mark.parametrize(("algo", "bench", "params", "bound"), [
    ("popularity", "ml-1m-loo", {}, 20),
    ("ease", "ml-1m-loo", {}, 30),
    ("ease", "ml-20m-strong", {}, 600),
    ("sasrec", "ml-1m-loo", {"epochs": 1}, 60),
    ("deepfm", "ml-1m-ctr", {"epochs": 1}, 60),
])
def test_f2_pipeline_bounds(algo, bench, params, bound):
    BENCHMARKS[bench].prepare()
    record = recsys.run(algo, bench, params=params, results_dir=None)
    assert record["fit_s"] + record["eval_s"] <= bound


def test_f4_warm_prepare_is_cheap():
    cold = _cold_prepare_seconds("ml-1m-loo")
    _prepare.cache_clear()
    start = time.perf_counter()
    BENCHMARKS["ml-1m-loo"].prepare()
    assert time.perf_counter() - start <= 0.2 * cold


def test_f1_f3_evaluation_memory_and_scaling():
    data = BENCHMARKS["ml-20m-loo"].prepare()
    model = recsys.ALGORITHMS["popularity"]()
    model.fit(data.train_data(0))
    known, targets = data.known("test"), data.targets("test")
    before = _rss_mb()
    timings = []
    for fraction in (0.1, 1.0):
        n = int(targets.n_users * fraction)
        sub = recsys.History(targets.indptr.copy(), targets.items, targets.n_items)
        sub.indptr[n + 1:] = sub.indptr[n]  # keep only the first n users' targets
        start = time.perf_counter()
        evaluate_ranking(model, known, sub, BENCHMARKS["ml-20m-loo"].metrics)
        timings.append(time.perf_counter() - start)
    assert _rss_mb() - before <= 4096
    assert timings[1] <= 12 * timings[0]


@pytest.mark.skipif(not os.environ.get("RECSYS_PERF_GPU"), reason="set RECSYS_PERF_GPU=1 to profile device copies")
def test_f5_training_tensors_stay_on_device():
    import torch

    data = BENCHMARKS["ml-1m-loo"].prepare()
    model = recsys.ALGORITHMS["sasrec"](epochs=1)
    model.build(data.train_data(0))
    model.to(model.device)
    with torch.profiler.profile() as prof:
        for batch in model.batches(None):
            model.loss(batch).backward()
    copies = [e for e in prof.key_averages() if e.key == "aten::_to_copy"]
    assert sum(e.count for e in copies) == 0
