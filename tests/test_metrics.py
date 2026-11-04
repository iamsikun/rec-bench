"""Metric definitions against independent references (criterion B9)."""

import numpy as np
import pytest
import torch
from sklearn.metrics import log_loss, roc_auc_score

from recsys import metrics as M


def _reference(ranked: np.ndarray, relevant: set, k: int) -> dict[str, float]:
    """Plain-Python definitions, one user at a time."""
    top = list(ranked[:k])
    hits = [1.0 if i in relevant else 0.0 for i in top]
    dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
    idcg = sum(1 / np.log2(r + 2) for r in range(min(len(relevant), k)))
    first = next((r for r, h in enumerate(hits) if h), None)
    return {
        "hr": float(any(hits)),
        "recall": sum(hits) / len(relevant),
        "nrecall": sum(hits) / min(k, len(relevant)),
        "precision": sum(hits) / k,
        "ndcg": dcg / idcg,
        "mrr": 0.0 if first is None else 1 / (first + 1),
    }


@pytest.mark.parametrize("name", list(M.RANKING))
def test_ranking_metric_matches_reference(name):
    rng = np.random.default_rng(0)
    for _ in range(60):
        n_items = int(rng.integers(5, 60))
        k = int(rng.integers(1, min(20, n_items)))
        ranked = rng.permutation(n_items)
        relevant = set(rng.choice(n_items, size=int(rng.integers(1, 8)), replace=False).tolist())
        hits = torch.tensor([[1.0 if i in relevant else 0.0 for i in ranked[:k]]])
        got = M.RANKING[name](hits, torch.tensor([float(len(relevant))]), k).item()
        assert got == pytest.approx(_reference(ranked, relevant, k)[name], abs=1e-6)


def test_ctr_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    for _ in range(60):
        n = int(rng.integers(10, 500))
        y = rng.integers(0, 2, n)
        if y.min() == y.max():
            continue
        p = rng.random(n).round(int(rng.integers(1, 4)))  # rounding creates ties
        p = np.clip(p, 1e-6, 1 - 1e-6)
        assert M.auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-6)
        assert M.logloss(y, p) == pytest.approx(log_loss(y, p), abs=1e-6)


def test_metric_names_parse():
    assert M.parse("ndcg@10") == ("ndcg", 10)
    with pytest.raises(ValueError):
        M.parse("ndcg")
