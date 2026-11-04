"""Tier-1 neural ranking methods: they learn on synthetic data, overfit tiny data, and respect causality/padding."""

import numpy as np
import polars as pl
import pytest
import torch

from recsys import History, TrainData, run
from recsys.algorithms.bert4rec import BERT4Rec
from recsys.algorithms.gru4rec import GRU4Rec, left_align
from recsys.algorithms.gsasrec import GSASRec
from recsys.algorithms.multvae import MultDAE, MultVAE
from recsys.algorithms.neumf import NeuMF
from recsys.algorithms.srgnn import SRGNN

FAST = {"device": "cpu", "patience": None}
SYNTHETIC = [
    (NeuMF, {"epochs": 3}),
    (MultVAE, {"epochs": 10, "batch_size": 32}),
    (MultDAE, {"epochs": 10, "batch_size": 32}),
    (GRU4Rec, {"epochs": 3}),
    (SRGNN, {"epochs": 2}),
    (BERT4Rec, {"epochs": 5, "max_len": 30, "eval_every": 5}),
    (GSASRec, {"epochs": 10, "lr": 1e-2}),  # gBCE with many negatives plateaus at popularity early on at 1e-3
]


@pytest.mark.parametrize(("cls", "params"), SYNTHETIC, ids=lambda x: getattr(x, "__name__", ""))
def test_learns_on_synthetic(cls, params):
    record = run(cls, "synthetic-ranking", seed=0, params=params | FAST, results_dir=None)
    assert record["recall@20"] > 0.1  # random ranking over 200 items gives ~0.1


@pytest.mark.parametrize("cls", [MultVAE, GRU4Rec])
def test_inductive_on_heldout_users(cls):
    params = {"epochs": 10, "batch_size": 32} if cls is MultVAE else {"epochs": 3}
    record = run(cls, "synthetic-heldout", seed=0, params=params | FAST, results_dir=None)
    assert record["recall@20"] > 0.1


def _tiny(n_users: int = 16, n_items: int = 100, length: int = 8) -> TrainData:
    """Each user walks a deterministic cycle through the catalog: easy to memorize."""
    rows = [(u, (3 * u + 2 * t) % n_items, t) for u in range(n_users) for t in range(length)]
    train = pl.DataFrame(rows, schema=["user", "item", "timestamp"], orient="row")
    history = History.from_frame(train, n_users, n_items)
    return TrainData("ranking", train, n_users, n_items, {"user": n_users, "item": n_items}, history, 0)


OVERFIT = [  # (class, params, learning rate)
    (NeuMF, {}, 1e-2), (MultVAE, {"input_dropout": 0.0}, 1e-3), (MultDAE, {"input_dropout": 0.0, "reg": 0.0}, 1e-3),
    (GRU4Rec, {"dropout": 0.0}, 1e-2), (GRU4Rec, {"dropout": 0.0, "loss": "sampled", "n_samples": 8}, 1e-2),
    (SRGNN, {"weight_decay": 0.0}, 1e-2), (BERT4Rec, {"dropout": 0.0, "max_len": 10}, 3e-3),
    (GSASRec, {"dropout": 0.0}, 1e-2),
]


@pytest.mark.parametrize(("cls", "params", "lr"), OVERFIT, ids=lambda x: getattr(x, "__name__", ""))
def test_loss_decreases_on_tiny_data(cls, params, lr):
    torch.manual_seed(0)
    data = _tiny()
    model = cls(**params, batch_size=4, device="cpu")
    model.build(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = []
    for _ in range(30):
        losses = []
        for batch in model.batches(data):
            loss = model.loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epochs.append(np.mean(losses))
    assert epochs[-1] <= 0.7 * epochs[0], epochs


def _fitted(cls, **params):
    torch.manual_seed(0)
    model = cls(**params, device="cpu")
    model.build(_tiny())
    return model.eval()


@torch.no_grad()
def test_gru4rec_and_gsasrec_are_causal():
    seq = torch.tensor([[0, 0, 4, 9, 2, 7, 5, 11]])
    changed = seq.clone()
    changed[0, 5:] = torch.tensor([13, 1, 3])
    gru = _fitted(GRU4Rec, max_len=8)
    a, b = gru.encode(left_align(seq)[0]), gru.encode(left_align(changed)[0])
    assert torch.allclose(a[:, :3], b[:, :3]) and not torch.allclose(a[:, 3:], b[:, 3:])
    sas = _fitted(GSASRec, max_len=8)
    a, b = sas.encode(seq), sas.encode(changed)
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-6) and not torch.allclose(a[:, 5:], b[:, 5:])


@torch.no_grad()
def test_srgnn_trains_on_prefixes_matching_inference():
    data = _tiny()
    model = _fitted(SRGNN, max_len=8)
    seq, target = model.prefixes(torch.arange(len(model._ends)))
    user = np.searchsorted(data.history.indptr, model._starts.numpy(), side="right") - 1
    t = (model._ends - model._starts).numpy()  # prefix length
    for i in [0, 5, 30, len(t) - 1]:
        items = data.history.items[data.history.indptr[user[i]] :][: t[i] + 1]
        assert target[i].item() == items[-1]  # the target is the next item, never inside the prefix
        prefix = History(np.array([0, t[i]]), items[:-1], data.n_items)
        expected = model.score(np.array([0]), prefix)
        assert torch.allclose(model.encode(seq[i : i + 1]) @ model.item_emb.weight[1:].T, expected, atol=1e-5)


@pytest.mark.parametrize("cls", [GRU4Rec, SRGNN, BERT4Rec, GSASRec])
@torch.no_grad()
def test_left_padding_does_not_change_scores(cls):
    model = _fitted(cls, max_len=12)
    history = _tiny().history  # 8 items per user: padded by 4 at max_len 12, by 1 at max_len 9
    users = np.arange(4)
    long = model.score(users, history)
    model.max_len = 9
    assert torch.allclose(long, model.score(users, history), atol=1e-5)
