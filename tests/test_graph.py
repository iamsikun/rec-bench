"""Graph CF and autoencoder methods: SimGCL, XSimGCL, GF-CF and RecVAE learn and match their definitions."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from recsys import BENCHMARKS, History, run
from recsys.algorithms.gfcf import GFCF
from recsys.algorithms.recvae import RecVAE
from recsys.algorithms.simgcl import SimGCL, XSimGCL

FAST = {"device": "cpu", "patience": None}
# Random ranking over 200 items gives recall@20 ~ 0.1. The synthetic data rewards popularity, which the
# contrastive (uniformity) term of SimGCL/XSimGCL works against; small noise and a high lr let BPR win quickly.
GCL = {"epochs": 10, "lr": 1e-2, "eps": 0.01}
CASES = [
    (SimGCL, "synthetic-ranking", GCL),
    (XSimGCL, "synthetic-ranking", GCL),
    (GFCF, "synthetic-ranking", {}),
    (GFCF, "synthetic-heldout", {}),
    (RecVAE, "synthetic-ranking", {"epochs": 3, "batch_size": 64}),
    (RecVAE, "synthetic-heldout", {"epochs": 3, "batch_size": 64}),
]


@pytest.mark.parametrize(("cls", "bench", "params"), CASES, ids=lambda x: getattr(x, "__name__", x))
def test_learns_on_synthetic(cls, bench, params):
    params = params | FAST if params else {"device": "cpu"}
    assert run(cls, bench, seed=0, params=params, results_dir=None)["recall@20"] > 0.14


def test_gfcf_matches_numpy_closed_form():
    X = (np.random.default_rng(0).random((40, 12)) < 0.3).astype(np.float64)
    X[np.arange(40), np.arange(40) % 12] = 1.0  # no empty rows or columns
    history = History(np.concatenate([[0], np.cumsum(X.sum(1))]).astype(np.int64), np.nonzero(X)[1], 12)
    alpha, k = 0.3, 4
    du, di = X.sum(1) ** -0.5, X.sum(0) ** -0.5
    R = du[:, None] * X * di[None, :]
    V = np.linalg.svd(R)[2][:k].T
    expected = X @ (R.T @ R + alpha * (di[:, None] * V) @ (V.T / di[None, :]))
    model = GFCF(alpha=alpha, k=k, device="cpu")
    model.fit(SimpleNamespace(history=history))
    np.testing.assert_allclose(model.score(np.arange(40), history).numpy(), expected, atol=1e-4)


@torch.no_grad()
def test_simgcl_noise_is_sign_aligned_with_norm_eps_and_scoring_is_clean():
    torch.manual_seed(0)
    data = BENCHMARKS["synthetic-ranking"].prepare().train_data(0)
    model = SimGCL(eps=0.1, device="cpu")
    model.build(data)
    clean, noise = model.layers(False)[0], model.layers(True)[0] - model.layers(False)[0]
    assert torch.allclose(noise.norm(dim=1), torch.full((len(noise),), 0.1), atol=1e-5)
    assert (noise * clean >= 0).all()
    model.eval()
    users = np.arange(5)
    first = model.score(users, data.history)
    model.train().eval()  # drops the score cache
    assert torch.equal(first, model.score(users, data.history))


def test_recvae_alternates_encoder_and_decoder_updates():
    torch.manual_seed(0)
    data = BENCHMARKS["synthetic-ranking"].prepare().train_data(0)
    model = RecVAE(hidden=16, latent=8, batch_size=100, device="cpu")
    model.build(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def snapshot(module):
        return [p.detach().clone() for p in module.parameters()]

    def same(a, b):
        return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))

    enc, dec, phases = snapshot(model.encoder), snapshot(model.decoder), []
    for x, encoder_step in model.batches(data):
        if not encoder_step and (not phases or phases[-1]):  # first decoder pass: the encoder phase just ended
            assert same(dec, snapshot(model.decoder)) and not same(enc, snapshot(model.encoder))
            assert same(snapshot(model.prior_encoder), snapshot(model.encoder))
            enc, dec = snapshot(model.encoder), snapshot(model.decoder)
        phases.append(encoder_step)
        loss = model.loss((x, encoder_step))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert phases == [True] * 9 + [False] * 3  # 300 users, 3 encoder passes then 1 decoder pass
    assert same(enc, snapshot(model.encoder)) and not same(dec, snapshot(model.decoder))
