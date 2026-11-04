"""Every registered algorithm on every compatible synthetic benchmark, plus correctness checks (D3-D5)."""

import math

import numpy as np
import polars as pl
import pytest
import torch

import recsys
from recsys import ALGORITHMS, BENCHMARKS, TorchRecommender
from recsys.algorithms import EASE, Popularity, SASRec
from recsys.algorithms.base import incompatibility
from recsys.benchmarks import Benchmark, History
from recsys.data import Synthetic
from recsys.evaluate import evaluate_ranking
from recsys.protocol import Binarize, Dedup, LeaveLastOut, RandomSplit

SYNTHETIC = [n for n in BENCHMARKS if n.startswith("synthetic")]
CASES = [(a, b) for a, cls in ALGORITHMS.items() for b in SYNTHETIC if incompatibility(cls, BENCHMARKS[b]) is None]


# Small model sizes for algorithms whose defaults are too heavy for the offline suite.
_GENERATIVE = {"dim": 32, "ff": 64, "n_heads": 2, "n_layers": 1, "max_len": 10, "codes": 16, "rq_steps": 100}
_PREFIX = {"epochs": 1, "max_len": 20}  # one training example per prefix: many more steps per epoch
SMALL = {"tiger": _GENERATIVE, "letter": _GENERATIVE, "liger": _GENERATIVE, "fmlprec": _PREFIX, "bsarec": _PREFIX}


def _params(name: str, epochs: int = 3) -> dict:
    base = {"epochs": epochs, "device": "cpu"} if issubclass(ALGORITHMS[name], TorchRecommender) else {}
    return {**base, **SMALL.get(name, {})}


@pytest.mark.parametrize(("name", "bench"), CASES)
def test_algorithm_on_synthetic(name, bench, tmp_path):
    record = recsys.run(name, bench, seed=0, params=_params(name), results_dir=tmp_path)
    if BENCHMARKS[bench].task == "ranking":  # quality is checked by D4 and each family's tests
        assert all(0.0 <= record[m] <= 1.0 for m in BENCHMARKS[bench].metrics)
    else:
        assert 0.5 < record["auc"] <= 1.0 and math.isfinite(record["logloss"])
    assert (tmp_path / f"{bench}.parquet").exists()


def test_every_algorithm_runs_somewhere():
    assert {a for a, _ in CASES} == set(ALGORITHMS)


# ------------------------------------------------------------------ D3: exactness


def test_d3_ease_matches_numpy_closed_form():
    data = BENCHMARKS["synthetic-ranking"].prepare()
    model = EASE(reg=50.0, device="cpu")
    model.fit(data.train_data(0))
    X = data.train_data(0).history.matrix().toarray().astype(np.float64)
    P = np.linalg.inv(X.T @ X + 50.0 * np.eye(X.shape[1]))
    B = P / -np.diag(P)
    np.fill_diagonal(B, 0.0)
    assert np.abs(model.B.cpu().numpy() - B).max() < 1e-5 * max(1.0, np.abs(B).max())


def test_d3_popularity_scores_equal_counts():
    data = BENCHMARKS["synthetic-ranking"].prepare()
    model = Popularity(device="cpu")
    model.fit(data.train_data(0))
    known = data.known("test")
    scores = model.score(np.array([0]), known).cpu().numpy()[0]
    assert np.array_equal(scores, np.bincount(known.items, minlength=data.n_items))


# --------------------------------------------------------------- D4: they can learn

TINY = Synthetic(n_users=24, n_items=30, events_per_user=10, seed=5)
TINY_RANKING = Benchmark("tiny-ranking", TINY, "ranking", LeaveLastOut(), ("hr@10",), (Dedup(),),
                         user_fields=("segment",), item_fields=("category",))
TINY_CTR = Benchmark("tiny-ctr", TINY, "ctr", RandomSplit(0.05, 0.05), ("auc",), (Binarize("rating", 4),),
                     fields=("user", "item", "segment", "category"), history=10)
TORCH = [n for n, cls in ALGORITHMS.items() if issubclass(cls, TorchRecommender)]
# Settings that let each model overfit the tiny fixture quickly (not their benchmark defaults).
OVERFIT = {
    "bert4rec": {"lr": 5e-3, "epochs": 150}, "multvae": {"lr": 1e-3, "epochs": 300}, "tiger": _GENERATIVE,
    "letter": {**_GENERATIVE, "epochs": 30}, "liger": {**_GENERATIVE, "epochs": 30},
    "simgcl": {"epochs": 50}, "xsimgcl": {"epochs": 50}, "recvae": {"epochs": 40, "hidden": 64, "latent": 32},
    "tisasrec": {"epochs": 30}, "mamba4rec": {"epochs": 30}, "fmlprec": {"epochs": 30}, "bsarec": {"epochs": 60},
    "cl4srec": {"epochs": 60}, "duorec": {"epochs": 60, "dropout": 0.1},
}


@pytest.mark.parametrize("name", TORCH)
def test_d4_neural_models_learn_and_overfit(name):
    cls = ALGORITHMS[name]
    bench = TINY_CTR if "ctr" in cls.tasks else TINY_RANKING
    if incompatibility(cls, bench) is not None:
        pytest.skip(incompatibility(cls, bench))
    recsys.harness.seed_everything(0)
    data = bench.prepare()
    params = {"epochs": 60 if bench.task == "ctr" else 150, "batch_size": 32, "patience": None, "device": "cpu",
              "lr": 1e-2}
    model = cls(**{**params, **OVERFIT.get(name, {})})
    model.fit(data.train_data(0))
    assert model.losses[-1] <= 0.7 * model.losses[0]
    if name == "lr":
        return  # linear: it learns (loss above) but cannot memorize
    if bench.task == "ctr":
        assert recsys.metrics.auc(data.train["label"].to_numpy(), model.predict(data.train)) >= 0.95
    else:
        ordered = data.train.with_row_index("_i").sort(["user", "timestamp", "_i"])
        is_last = pl.int_range(pl.len()).over("user") == pl.len().over("user") - 1
        known = History.from_frame(ordered.filter(~is_last), data.n_users, data.n_items)
        target = History.from_frame(ordered.filter(is_last), data.n_users, data.n_items)
        result = evaluate_ranking(model, known, target, ("hr@10",))  # recall a training item from the rest
        assert result["hr@10"].mean() >= 0.9


# ------------------------------------------------------------------- D5: causality


def test_d5_sasrec_is_causal_and_padding_invariant():
    data = BENCHMARKS["synthetic-ranking"].prepare()
    model = SASRec(epochs=1, max_len=16, device="cpu")
    model.fit(data.train_data(0))
    seq = torch.randint(1, data.n_items + 1, (4, 12))
    changed = seq.clone()
    changed[:, 8:] = torch.randint(1, data.n_items + 1, (4, 4))
    with torch.no_grad():
        assert torch.allclose(model.encode(seq)[:, :8], model.encode(changed)[:, :8], atol=1e-5)
        short = torch.cat([torch.zeros(4, 2, dtype=torch.long), seq[:, 2:]], 1)
        longer = torch.cat([torch.zeros(4, 4, dtype=torch.long), short], 1)
        assert torch.allclose(model.encode(short)[:, -1], model.encode(longer)[:, -1], atol=1e-5)


def test_ctr_fields_encoded_from_train():
    data = BENCHMARKS["synthetic-ctr"].prepare()
    for name, size in data.fields.items():
        assert data.train[name].min() >= 1
        assert data.test[name].max() < size
    assert isinstance(data.test, pl.DataFrame) and "history" in data.test.columns
