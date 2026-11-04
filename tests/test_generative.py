"""Generative recommenders (TIGER, HSTU): accuracy, valid decoding, unique semantic IDs, causality."""

import numpy as np
import pytest
import torch

from recsys import BENCHMARKS, evaluate, run
from recsys.algorithms.hstu import HSTU
from recsys.algorithms.tiger import TIGER

# Small, CPU-friendly settings (~2 s fit); the defaults are sized for real data.
TIGER_PARAMS = {"dim": 32, "ff": 64, "n_heads": 2, "n_layers": 1, "max_len": 10, "codes": 16, "rq_steps": 100,
                "epochs": 3, "batch_size": 256, "lr": 1e-2, "dropout": 0.0, "patience": None, "device": "cpu"}


@pytest.fixture(scope="module")
def tiger():
    data = BENCHMARKS["synthetic-ranking"].prepare()
    torch.manual_seed(0)
    model = TIGER(**TIGER_PARAMS)
    model.fit(data.train_data(seed=0))
    return model, data


def test_tiger_beats_random(tiger):
    model, data = tiger
    assert evaluate(model, data, "test")["recall@20"] > 0.1  # random ranking over 200 items gives ~0.1


def test_hstu_beats_random():
    record = run(HSTU, "synthetic-ranking", seed=0, params={"max_len": 30, "epochs": 12, "device": "cpu"},
                 results_dir=None)
    assert record["recall@20"] > 0.1  # random ranking over 200 items gives ~0.1


def test_tiger_semantic_ids_unique(tiger):
    model, data = tiger
    assert model.sids.shape == (data.n_items, model.levels + 1)
    assert len(np.unique(model.sids, axis=0)) == data.n_items


def test_tiger_recommend_valid(tiger):
    model, data = tiger
    known = data.known("test")
    users = np.arange(data.n_users)
    ids = torch.as_tensor(model.recommend(users, known, 20)).cpu().numpy()
    assert ids.shape == (len(users), 20)
    assert ((ids == -1) | ((ids >= 0) & (ids < data.n_items))).all()
    seen = known.matrix(users).toarray() > 0
    for u, row in zip(users, ids):
        real = row[row >= 0]
        assert len(set(real)) == len(real)
        assert not seen[u, real].any()
    assert (ids >= 0).all()  # pruned beams always fill the list (enough unseen items here)


def test_hstu_is_causal():
    data = BENCHMARKS["synthetic-ranking"].prepare()
    model = HSTU(max_len=12, device="cpu")
    model.build(data.train_data(seed=0))
    model.eval()
    seq = torch.randint(1, data.n_items + 1, (4, 12))
    seq[:, :3] = 0  # left padding
    changed = seq.clone()
    changed[:, 8:] = torch.randint(1, data.n_items + 1, (4, 4))
    with torch.no_grad():
        a, b = model.encode(seq), model.encode(changed)
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-6)
    assert not torch.allclose(a[:, 8:], b[:, 8:])
