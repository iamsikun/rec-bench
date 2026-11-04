"""Evaluators: full-catalog ranking and pointwise CTR."""

from __future__ import annotations

import numpy as np
import polars as pl
import torch

from recsys import metrics as M
from recsys.algorithms.base import Recommender, implements
from recsys.benchmarks.base import BenchmarkData, History, Split
from recsys.device import resolve_device


def evaluate(algo: Recommender, data: BenchmarkData, split: Split) -> dict[str, float]:
    """The benchmark's pinned metrics on ``split`` (means over users / rows)."""
    return summarize(evaluate_detailed(algo, data, split), data.benchmark.metrics)


def evaluate_detailed(algo: Recommender, data: BenchmarkData, split: Split) -> pl.DataFrame:
    """Ranking: one row per evaluated user with each metric. CTR: one row per example with label and prediction."""
    if data.benchmark.task == "ranking":
        return evaluate_ranking(algo, data.known(split), data.targets(split), data.benchmark.metrics)
    rows = getattr(data, split)
    return pl.DataFrame({"label": rows["label"], "prediction": predict_ctr(algo, rows)})


def summarize(detail: pl.DataFrame, metrics: tuple[str, ...]) -> dict[str, float]:
    if "prediction" in detail.columns:
        y, p = detail["label"].to_numpy(), detail["prediction"].to_numpy()
        return {m: M.CTR[m](y, p) for m in metrics}
    return {m: float(detail[m].mean()) for m in metrics}


@torch.no_grad()
def evaluate_ranking(
    algo: Recommender,
    known: History,
    targets: History,
    metrics: tuple[str, ...],
    batch_size: int = 1024,
) -> pl.DataFrame:
    """Rank the full catalog for every user with targets; ``known`` items are excluded.

    Score-based algorithms get exact ties broken uniformly at random (fixed
    seed) so constant scores earn the random-ranking expectation, not a bias
    toward low item ids. Algorithms that only implement ``recommend`` are
    scored on the ids they return; repeated ids count once.
    """
    specs = [M.parse(m) for m in metrics]
    k_max = max(k for _, k in specs)
    users = np.flatnonzero(targets.lengths() > 0)
    device = resolve_device()
    generator = torch.Generator(device=device).manual_seed(0)
    use_score = implements(algo, "score")
    columns: list[list[np.ndarray]] = [[] for _ in specs]
    for start in range(0, len(users), batch_size):
        batch = users[start : start + batch_size]
        rel = targets.matrix(batch).tocoo()
        relevant = torch.zeros((len(batch), targets.n_items), dtype=torch.float32, device=device)
        relevant[torch.as_tensor(rel.row, device=device), torch.as_tensor(rel.col, device=device)] = 1.0
        if use_score:
            scores = torch.as_tensor(algo.score(batch, known)).float().to(device)
            seen = known.matrix(batch).tocoo()
            scores[torch.as_tensor(seen.row, device=device), torch.as_tensor(seen.col, device=device)] = -torch.inf
            perm = torch.rand(scores.shape, device=device, generator=generator).argsort(dim=1)
            top = perm.gather(1, scores.gather(1, perm).topk(k_max, dim=1).indices)
            hits = relevant.gather(1, top)
        else:
            ids = torch.as_tensor(algo.recommend(batch, known, k_max)).long().to(device)[:, :k_max]
            hits = torch.zeros((len(batch), k_max), device=device)
            for j in range(ids.shape[1]):
                col = ids[:, j : j + 1]
                valid = (col >= 0) & (col < targets.n_items)
                safe = col.clamp(0, targets.n_items - 1)
                hits[:, j : j + 1] = relevant.gather(1, safe) * valid
                relevant.scatter_(1, safe, relevant.gather(1, safe) * ~valid)  # consume hits
        n_rel = torch.as_tensor(targets.lengths(batch), device=device, dtype=torch.float32)
        for j, (name, k) in enumerate(specs):
            columns[j].append(M.RANKING[name](hits, n_rel, k).cpu().numpy())
    out = {"user": users}
    out |= {m: np.concatenate(columns[j]) if columns[j] else np.zeros(0) for j, m in enumerate(metrics)}
    return pl.DataFrame(out)


@torch.no_grad()
def predict_ctr(algo: Recommender, rows: pl.DataFrame, batch_size: int = 65536) -> np.ndarray:
    return np.concatenate([
        np.asarray(algo.predict(rows.slice(i, batch_size)), dtype=np.float64).reshape(-1)
        for i in range(0, rows.height, batch_size)
    ])
