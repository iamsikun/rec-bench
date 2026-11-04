"""Append-only parquet result store (one file per benchmark), summaries and significance tests."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

LOGGER = logging.getLogger(__name__)

_KEYS = ["benchmark", "version", "algo", "params_hash"]
_META = {
    *_KEYS, "run_id", "params", "seed", "fit_s", "eval_s", "peak_rss_mb", "peak_device_mb", "device",
    "timestamp", "code_sha", "dirty", "versions", "search_space", "budget",
}


def append(results_dir: str | Path, record: dict[str, Any]) -> None:
    path = Path(results_dir) / f"{record['benchmark']}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pl.DataFrame([record])
    if path.exists():
        row = pl.concat([pl.read_parquet(path), row], how="diagonal_relaxed")
    row.write_parquet(path)


def _detail_path(results_dir: str | Path, benchmark: str, run_id: str) -> Path:
    return Path(results_dir) / "detail" / benchmark / f"{run_id}.parquet"


def save_detail(results_dir: str | Path, record: dict[str, Any], detail: pl.DataFrame) -> None:
    path = _detail_path(results_dir, record["benchmark"], record["run_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    detail.write_parquet(path)


def load(results_dir: str | Path, benchmark: str | None = None) -> pl.DataFrame:
    paths = sorted(Path(results_dir).glob(f"{benchmark or '*'}.parquet"))
    return pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed") if paths else pl.DataFrame()


def metric_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META and df[c].dtype.is_numeric() and df[c].null_count() < df.height]


def summary(results_dir: str | Path = "results", benchmark: str | None = None) -> dict[str, pl.DataFrame]:
    """Per benchmark: mean and std of every metric over seeds, grouped by (version, algo, params).

    Rows from different benchmark versions are never averaged together; a
    warning is logged when one benchmark name has results under several versions.
    """
    out = {}
    for name, df in load(results_dir, benchmark).group_by("benchmark", maintain_order=True):
        if df["version"].n_unique() > 1:
            LOGGER.warning("%s has results under %d versions; they are reported separately",
                           name[0], df["version"].n_unique())
        metrics = metric_columns(df)
        out[name[0]] = (
            df.group_by(_KEYS[1:], maintain_order=True)
            .agg(
                pl.len().alias("seeds"),
                *[pl.col(m).mean() for m in metrics],
                *[pl.col(m).std().alias(f"{m} std") for m in metrics],
                pl.col("fit_s").mean(),
            )
            .sort(metrics[0], descending=True)
        )
    return out


def compare(results_dir: str | Path, benchmark: str, a: str, b: str, metric: str | None = None) -> dict[str, Any]:
    """Paired comparison of algorithms ``a`` and ``b`` on the latest benchmark version.

    Uses per-user (or, for CTR, per-seed) values when per-user detail exists
    for every shared seed, otherwise per-seed means. Returns the mean
    difference and paired t-test and Wilcoxon p-values.
    """
    df = load(results_dir, benchmark)
    df = df.filter(pl.col("version") == df.sort("timestamp")["version"][-1])
    metric = metric or metric_columns(df)[0]
    runs = {
        algo: df.filter(pl.col("algo") == algo).sort("timestamp").unique("seed", keep="last").sort("seed")
        for algo in (a, b)
    }
    seeds = sorted(set(runs[a]["seed"]) & set(runs[b]["seed"]))
    if not seeds:
        raise ValueError(f"{a} and {b} share no seeds on {benchmark}")
    rid = {algo: dict(zip(r["seed"], r["run_id"])) for algo, r in runs.items()}
    paths = {algo: [_detail_path(results_dir, benchmark, rid[algo][s]) for s in seeds] for algo in (a, b)}
    detail = all(p.exists() for ps in paths.values() for p in ps)
    if detail and metric in pl.read_parquet(paths[a][0]).columns:
        joined = [
            pl.read_parquet(pa).join(pl.read_parquet(pb), on="user", suffix="_b")
            for pa, pb in zip(paths[a], paths[b])
        ]
        x = np.concatenate([j[metric].to_numpy() for j in joined])
        y = np.concatenate([j[f"{metric}_b"].to_numpy() for j in joined])
        unit = "user"
    else:
        x = runs[a].filter(pl.col("seed").is_in(seeds))[metric].to_numpy()
        y = runs[b].filter(pl.col("seed").is_in(seeds))[metric].to_numpy()
        unit = "seed"
    diff = x - y
    t_p = float(stats.ttest_rel(x, y).pvalue) if len(diff) > 1 and diff.std() > 0 else float("nan")
    w_p = float(stats.wilcoxon(diff).pvalue) if np.count_nonzero(diff) > 0 else 1.0
    return {"metric": metric, "unit": unit, "n": len(diff), "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "diff": float(diff.mean()), "t_test_p": t_p, "wilcoxon_p": w_p}
