"""Run algorithms on benchmarks and record the results."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import itertools
import json
import logging
import random
import resource
import subprocess
import sys
import time
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from recsys import store
from recsys.algorithms import ALGORITHMS, Recommender
from recsys.algorithms.base import incompatibility
from recsys.benchmarks import BENCHMARKS, Benchmark
from recsys.evaluate import evaluate, evaluate_detailed, summarize

LOGGER = logging.getLogger(__name__)

AlgoSpec = str | type[Recommender]
BenchSpec = str | Benchmark


def _algo(spec: AlgoSpec) -> tuple[str, type[Recommender]]:
    if isinstance(spec, str):
        return spec, ALGORITHMS[spec]
    name = next((k for k, v in ALGORITHMS.items() if v is spec), spec.__name__.lower())
    return name, spec


def _accepted(cls: type) -> set[str]:
    """Constructor keyword names, following ``**kwargs`` up the class hierarchy."""
    names: set[str] = set()
    for klass in cls.__mro__:
        if "__init__" not in vars(klass):
            continue
        sig = inspect.signature(klass.__init__).parameters.values()
        names |= {p.name for p in sig if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)} - {"self"}
        if not any(p.kind is p.VAR_KEYWORD for p in sig):
            break
    return names


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=3, cwd=Path(__file__).parent)
        return out.stdout.strip() if out.returncode == 0 else None
    except OSError:
        return None


def _versions() -> str:
    names = {"recsys": "rec-benchmark", "torch": "torch", "numpy": "numpy", "scipy": "scipy", "polars": "polars"}
    out = {"python": sys.version.split()[0]}
    for label, dist in names.items():
        try:
            out[label] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[label] = "unknown"
    return json.dumps(out, sort_keys=True)


def _peak_device_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return 0.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(
    algo: AlgoSpec,
    benchmark: BenchSpec,
    seed: int = 0,
    params: dict[str, Any] | None = None,
    results_dir: str | Path | None = "results",
    per_user: bool = False,
) -> dict[str, Any]:
    """Fit ``algo`` on ``benchmark``, evaluate on its test split, and append the result to ``results_dir``.

    ``per_user=True`` also stores per-user metrics (ranking) or per-row
    predictions (CTR) next to the results file.
    """
    name, cls = _algo(algo)
    bench = BENCHMARKS[benchmark] if isinstance(benchmark, str) else benchmark
    reason = incompatibility(cls, bench)
    if reason:
        raise ValueError(f"{name} cannot run on {bench.name}: {reason}")
    params = dict(params or {})
    seed_everything(seed)
    data = bench.prepare()
    model = cls(**params)
    select = bench.metrics[0]

    def validate(m: Recommender) -> float:
        return evaluate(m, data, "val")[select]

    start = time.perf_counter()
    model.fit(data.train_data(seed, validate))
    fit_s = time.perf_counter() - start
    start = time.perf_counter()
    detail = evaluate_detailed(model, data, "test")
    metrics = summarize(detail, bench.metrics)
    eval_s = time.perf_counter() - start
    device = torch.device(str(getattr(model, "device", "cpu")))
    params_json = json.dumps(params, sort_keys=True, default=str)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {
        "run_id": uuid.uuid4().hex[:12],
        "benchmark": bench.name,
        "version": bench.version,
        "algo": name,
        "params": params_json,
        "params_hash": hashlib.sha256(params_json.encode()).hexdigest()[:10],
        "seed": seed,
        **metrics,
        "fit_s": fit_s,
        "eval_s": eval_s,
        "peak_rss_mb": rss / 2**20 if sys.platform == "darwin" else rss / 2**10,
        "peak_device_mb": _peak_device_mb(device),
        "device": str(device),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_sha": _git("rev-parse", "HEAD") or "unknown",
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "versions": _versions(),
    }
    LOGGER.info("%s on %s (seed %d): %s", name, bench.name, seed, " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    if results_dir is not None:
        store.append(results_dir, record)
        if per_user:
            store.save_detail(results_dir, record, detail)
    return record


def run_all(
    algos: AlgoSpec | Iterable[AlgoSpec],
    benchmarks: BenchSpec | Iterable[BenchSpec] = "all",
    seeds: Iterable[int] = (0,),
    params: dict[str, Any] | None = None,
    results_dir: str | Path | None = "results",
    per_user: bool = False,
) -> pl.DataFrame:
    """Run every algorithm on every compatible benchmark for every seed.

    ``benchmarks="all"`` means every registered benchmark except the synthetic
    ones. Incompatible pairs are skipped with a logged reason. Each algorithm
    receives only the ``params`` its constructor accepts.
    """
    algos = [algos] if isinstance(algos, (str, type)) else list(algos)
    params = dict(params or {})
    accepted = {spec: _accepted(_algo(spec)[1]) for spec in algos}
    unused = set(params) - set().union(*accepted.values())
    if unused:
        raise ValueError(f"No selected algorithm accepts parameters {sorted(unused)}")
    if benchmarks == "all":
        benchmarks = [b for n, b in BENCHMARKS.items() if not n.startswith("synthetic")]
    elif isinstance(benchmarks, (str, Benchmark)):
        benchmarks = [benchmarks]
    benches = [BENCHMARKS[b] if isinstance(b, str) else b for b in benchmarks]
    records = []
    for spec in algos:
        name, cls = _algo(spec)
        own = {k: v for k, v in params.items() if k in accepted[spec]}
        for bench in benches:
            reason = incompatibility(cls, bench)
            if reason:
                LOGGER.info("skip %s on %s: %s", name, bench.name, reason)
                continue
            for seed in seeds:
                records.append(run(spec, bench, seed, own, results_dir, per_user))
    return pl.DataFrame(records, infer_schema_length=None) if records else pl.DataFrame()


def tune(
    algo: AlgoSpec,
    benchmark: BenchSpec,
    budget: int = 20,
    seed: int = 0,
    results_dir: str | Path | None = "results",
) -> dict[str, Any]:
    """Random search with a fixed ``budget`` over the algorithm's ``space``, selected on validation.

    The best configuration is then run once (``run``) and its result row
    records the search space and budget, so every algorithm can be given the
    same tuning budget.
    """
    name, cls = _algo(algo)
    bench = BENCHMARKS[benchmark] if isinstance(benchmark, str) else benchmark
    space = getattr(cls, "space", {})
    if not space:
        raise ValueError(f"{name} declares no search space")
    rng = random.Random(seed)
    grid = [dict(zip(space, values)) for values in itertools.product(*space.values())]
    trials = rng.sample(grid, min(budget, len(grid)))
    data = bench.prepare()
    select = bench.metrics[0]
    best, best_params = -float("inf"), {}
    for params in trials:
        seed_everything(seed)
        model = cls(**params)
        model.fit(data.train_data(seed, lambda m: evaluate(m, data, "val")[select]))
        metric = evaluate(model, data, "val")[select]
        LOGGER.info("tune %s %s: val %s=%.4f", name, params, select, metric)
        if metric > best:
            best, best_params = metric, params
    record = run(algo, bench, seed, best_params, results_dir=None)
    record |= {"search_space": json.dumps(space, sort_keys=True, default=str), "budget": len(trials)}
    if results_dir is not None:
        store.append(results_dir, record)
    return record
