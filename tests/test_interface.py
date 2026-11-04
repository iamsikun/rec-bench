"""Algorithm interface and extensibility (criteria C1-C7, F1, F4)."""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from helpers import Constant

import recsys
from recsys import ALGORITHMS, BENCHMARKS, Recommender
from recsys.algorithms.base import incompatibility
from recsys.harness import run_all

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = [b for n, b in BENCHMARKS.items() if n.startswith("synthetic")]


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _code_lines(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#"))


class TwoTower(Recommender):
    """Retrieval: exposes embeddings and answers ``recommend`` with a top-k inner product."""

    tasks = frozenset({"ranking"})

    def fit(self, data):
        g = torch.Generator().manual_seed(0)
        self.u, self.i = torch.randn(data.n_users, 8, generator=g), torch.randn(data.n_items, 8, generator=g)

    def recommend(self, users, history, k):
        scores = self.u[torch.as_tensor(users)] @ self.i.T
        seen = history.matrix(users).tocoo()
        scores[seen.row, seen.col] = -torch.inf
        return scores.topk(k, dim=1).indices


class ToyGenerative(Recommender):
    """Generative: decodes ``k`` ids per user (repeats and -1 allowed), no score matrix."""

    tasks = frozenset({"ranking"})
    inductive = True

    def fit(self, data):
        self.n = data.n_items

    def recommend(self, users, history, k):
        last = history.sequences(users, 1)[:, 0]
        ids = (last[:, None] + np.arange(k)[None, :]) % self.n
        ids[:, -1] = -1
        ids[:, -2] = ids[:, 0]  # a repeat counts once
        return ids


@pytest.mark.parametrize("algo", [Constant, TwoTower, ToyGenerative])
def test_c1_dense_retrieval_and_generative_algorithms_evaluate(algo):
    record = recsys.run(algo, "synthetic-ranking", results_dir=None)
    assert 0.0 <= record["ndcg@10"] <= 1.0


def test_c1_repeated_ids_count_once():
    data = BENCHMARKS["synthetic-ranking"].prepare()

    class Repeat(Recommender):
        tasks = frozenset({"ranking"})

        def fit(self, data):
            pass

        def recommend(self, users, history, k):
            target = data.targets("test")
            first = target.items[target.indptr[users]]
            return np.repeat(first[:, None], k, axis=1)

    r = recsys.evaluate(Repeat(), data, "test")
    assert r["recall@20"] == pytest.approx(1.0) and r["mrr@10"] == pytest.approx(1.0)


def test_c2_minimal_algorithm_is_short_and_runs_everywhere():
    path = ROOT / "examples" / "minimal_algo.py"
    assert _code_lines(path) <= 15
    algo = _load(path).RecentPopularity
    out = run_all(algo, SYNTHETIC, results_dir=None)
    expected = {b.name for b in SYNTHETIC if incompatibility(algo, b) is None}
    assert set(out["benchmark"]) == expected and len(expected) >= 2


def test_c3_minimal_torch_algorithm_gets_loop_early_stopping_and_best_weights():
    path = ROOT / "examples" / "minimal_torch_algo.py"
    assert _code_lines(path) <= 40
    cls = _load(path).TinyMF
    snapshots = []

    class Recording(cls):
        def fit(self, data):
            validate = data.validate

            def spy(model):
                metric = validate(model)
                snapshots.append((metric, {k: v.clone() for k, v in model.state_dict().items()}))
                return metric

            data.validate = spy
            super().fit(data)

    data = BENCHMARKS["synthetic-ranking"].prepare()
    model = Recording(epochs=60, lr=0.05, patience=1, device="cpu")
    model.fit(data.train_data(0, lambda m: recsys.evaluate(m, data, "val")["ndcg@10"]))
    assert 0 < len(snapshots) < 60  # validation ran and early stopping fired
    best = max(snapshots, key=lambda s: s[0])[1]
    assert all(torch.equal(v, best[k]) for k, v in model.state_dict().items())


def test_c4_run_all_resolves_compatibility_without_errors():
    algos = ["popularity", "deepfm", "ease", "bpr"]
    out = run_all(algos, SYNTHETIC, params={"epochs": 1, "device": "cpu"}, results_dir=None)
    expected = {(a, b.name) for a in algos for b in SYNTHETIC if incompatibility(ALGORITHMS[a], b) is None}
    assert set(zip(out["algo"], out["benchmark"])) == expected
    assert ("bpr", "synthetic-heldout") not in expected and ("deepfm", "synthetic-ranking") not in expected


def test_c4_requirements_are_checked():
    class NeedsBehavior(Recommender):
        tasks = frozenset({"ctr"})
        requires = frozenset({"behavior"})

        def fit(self, data):
            pass

    assert incompatibility(NeedsBehavior, BENCHMARKS["synthetic-ctr"]) is None
    assert "behavior" in incompatibility(NeedsBehavior, BENCHMARKS["kuairec-ctr"])


def test_c5_algorithms_use_only_the_public_data_api():
    pattern = re.compile(r"recsys\.benchmarks|data\._|history\._")
    offenders = [p.name for p in (ROOT / "src/recsys/algorithms").glob("*.py") if pattern.search(p.read_text())]
    assert offenders == []


def test_c6_every_algorithm_module_is_registered_and_unregistered_classes_run():
    modules = {p.stem for p in (ROOT / "src/recsys/algorithms").glob("*.py")
               if not p.stem.startswith("_") and p.stem not in {"base", "trainer", "__init__"}}
    registered = {cls.__module__.rsplit(".", 1)[-1] for cls in ALGORITHMS.values()}
    assert modules == registered
    assert recsys.run(Constant, "synthetic-ranking", results_dir=None)["algo"] == "constant"


def test_c7_params_are_logged_and_tuning_records_its_space(tmp_path):
    a = recsys.run("ease", "synthetic-ranking", params={"reg": 10.0}, results_dir=None)
    b = recsys.run("ease", "synthetic-ranking", params={"reg": 20.0}, results_dir=None)
    assert a["params_hash"] != b["params_hash"] and json.loads(a["params"]) == {"reg": 10.0}
    tuned = recsys.tune("ease", "synthetic-ranking", budget=2, results_dir=tmp_path)
    assert json.loads(tuned["search_space"]) and tuned["budget"] == 2


def test_c7_every_algorithm_declares_a_search_space():
    assert [n for n, cls in ALGORITHMS.items() if not cls.space] == []


def test_f4_prepare_once_per_benchmark(monkeypatch):
    from recsys.benchmarks.base import _prepare
    from recsys.data import Synthetic

    calls = []
    original = Synthetic.read
    monkeypatch.setattr(Synthetic, "read", lambda self, root: calls.append(1) or original(self, root))
    _prepare.cache_clear()
    run_all("popularity", "synthetic-ranking", seeds=[0, 1, 2], results_dir=None)
    assert len(calls) <= 1  # 0 when the on-disk prepared cache is warm
