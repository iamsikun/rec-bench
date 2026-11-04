"""Classical ranking baselines: they beat Random on synthetic data and match reference implementations."""

from types import SimpleNamespace

import numpy as np
import pytest

from recsys import BENCHMARKS, History, run
from recsys.algorithms.ials import IALS
from recsys.algorithms.itemknn import ItemKNN
from recsys.algorithms.puresvd import PureSVD
from recsys.algorithms.random import Random
from recsys.algorithms.rp3beta import RP3beta
from recsys.algorithms.sknn import SKNN
from recsys.algorithms.slim import SLIM
from recsys.algorithms.userknn import UserKNN

# Small-data settings: the paper defaults overfit the 300 x 200 synthetic matrix.
CASES = {
    ItemKNN: {}, UserKNN: {}, RP3beta: {}, SKNN: {}, SLIM: {},
    PureSVD: {"factors": 4}, IALS: {"dim": 8, "reg": 0.03, "epochs": 8},
}
BENCHES = ["synthetic-ranking", "synthetic-heldout"]


@pytest.fixture(scope="module")
def random_recall():
    return {b: run(Random, b, params={"device": "cpu"}, results_dir=None)["recall@20"] for b in BENCHES}


@pytest.mark.parametrize("bench", BENCHES)
@pytest.mark.parametrize("cls", list(CASES), ids=lambda c: c.__name__)
def test_beats_random(cls, bench, random_recall):
    assert cls.inductive
    record = run(cls, bench, params={**CASES[cls], "device": "cpu"}, results_dir=None)
    assert record["recall@20"] > random_recall[bench] + 0.03


def _tiny(n_users=40, n_items=12, seed=0):
    X = (np.random.default_rng(seed).random((n_users, n_items)) < 0.3).astype(np.float32)
    indptr = np.concatenate([[0], np.cumsum(X.sum(1))]).astype(np.int64)
    return X, History(indptr, np.nonzero(X)[1].astype(np.int64), n_items)


def test_itemknn_matches_numpy_reference():
    k, shrink = 3, 2.0
    for seed in range(100):  # pick data with no ties at the top-k boundary, so the neighbour set is unique
        X, history = _tiny(seed=seed)
        G = X.T @ X
        norms = np.sqrt(np.diag(G))
        S = G / (np.outer(norms, norms) + shrink + 1e-6)
        np.fill_diagonal(S, 0.0)
        ranked = -np.sort(-S, axis=0)
        if np.all(ranked[k - 1] > ranked[k] + 1e-5):
            break
    keep = S >= ranked[k - 1][None, :]
    expected = X @ np.where(keep, S, 0.0)
    model = ItemKNN(k=k, shrink=shrink, device="cpu")
    model.fit(SimpleNamespace(history=history))
    users = np.arange(X.shape[0])
    np.testing.assert_allclose(model.score(users, history).numpy(), expected, rtol=1e-5, atol=1e-6)


def test_puresvd_scores_are_x_v_vt():
    X, history = _tiny()
    factors = 4
    model = PureSVD(factors=factors, device="cpu")
    model.fit(SimpleNamespace(history=history))
    V = np.linalg.svd(X.astype(np.float64))[2][:factors].T
    scores = model.score(np.arange(X.shape[0]), history).numpy()
    np.testing.assert_allclose(scores, X @ V @ V.T, atol=1e-4)


def test_ials_objective_decreases():
    data = BENCHMARKS["synthetic-ranking"].prepare().train_data(0)
    model = IALS(dim=8, epochs=6, device="cpu")
    model.fit(data)
    losses = np.array(model.losses)
    assert np.all(np.diff(losses) <= 1e-4 * losses[:-1])
    assert losses[-1] < losses[0]


def test_ials_cg_close_to_exact():
    data = BENCHMARKS["synthetic-ranking"].prepare().train_data(0)
    exact, cg = IALS(dim=8, epochs=4, device="cpu"), IALS(dim=8, epochs=4, cg_steps=8, device="cpu")
    exact.fit(data)
    cg.fit(data)
    assert cg.losses[-1] == pytest.approx(exact.losses[-1], rel=1e-3)
