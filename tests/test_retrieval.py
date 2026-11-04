"""Two-tower retrieval models: end-to-end runs and the ``recommend`` (MIPS top-k) path."""

import numpy as np
import pytest
import torch

from recsys import BENCHMARKS, History, Recommender, run
from recsys.algorithms._towers import top_k_unseen
from recsys.algorithms.dssm import DSSM
from recsys.algorithms.sbcnn import TwoTowerLogQ
from recsys.algorithms.youtubednn import YouTubeDNN
from recsys.evaluate import evaluate_ranking

CLASSES = [DSSM, YouTubeDNN, TwoTowerLogQ]
PARAMS = {"epochs": 3, "device": "cpu"}


@pytest.mark.parametrize("cls", CLASSES)
def test_runs_on_synthetic(cls):
    record = run(cls, "synthetic-ranking", seed=0, params=PARAMS, results_dir=None)
    assert record["recall@20"] > 0.1  # random ranking over 200 items gives ~0.1


class _Dense(Recommender):
    """Wraps a fitted model's dense inner products as a score-based recommender."""

    tasks = frozenset({"ranking"})

    def __init__(self, model):
        self.model = model

    def fit(self, data):  # pragma: no cover
        pass

    def score(self, users, history):
        u = torch.as_tensor(users, dtype=torch.long)
        with torch.no_grad():
            return self.model.user_vectors(u, history) @ self.model.item_vectors().T


@pytest.mark.parametrize("cls", CLASSES)
def test_recommend_path(cls):
    data = BENCHMARKS["synthetic-ranking"].prepare()
    model = cls(epochs=1, device="cpu")
    model.fit(data.train_data(0))
    known, targets = data.known("test"), data.targets("test")
    users = np.arange(data.n_users)
    ids = model.recommend(users, known, 20)
    assert isinstance(ids, torch.Tensor) and ids.dtype == torch.long and ids.shape == (data.n_users, 20)
    seen = known.matrix(users).toarray().astype(bool)
    assert not seen[np.arange(len(users))[:, None], ids.numpy()].any()
    assert all(len(set(row)) == 20 for row in ids.tolist())
    # More than the catalog: valid ids first, -1 padding after, still no history items.
    long = model.recommend(users[:5], known, data.n_items + 3).numpy()
    for row, s in zip(long, seen[:5]):
        valid = row[row >= 0]
        assert len(valid) == data.n_items - s.sum() and not s[valid].any() and (row[len(valid):] == -1).all()
    # Same metrics as the equivalent dense-score model through the evaluator.
    metrics = data.benchmark.metrics
    a = evaluate_ranking(model, known, targets, metrics)
    b = evaluate_ranking(_Dense(model), known, targets, metrics)
    for m in metrics:
        assert np.allclose(a[m].to_numpy(), b[m].to_numpy(), atol=1e-6), m


def test_top_k_unseen_matches_masked_argsort():

    history = History(np.array([0, 2, 2, 3]), np.array([1, 3, 0]), 5)
    scores = torch.tensor([[5.0, 4, 3, 2, 1], [1, 2, 3, 4, 5], [0, 1, 2, 3, 4]])
    out = top_k_unseen(scores, np.arange(3), history, 4)
    assert out.tolist() == [[0, 2, 4, -1], [4, 3, 2, 1], [4, 3, 2, 1]]
