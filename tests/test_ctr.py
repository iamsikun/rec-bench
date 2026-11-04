"""CTR models: end-to-end on synthetic-ctr, overfitting a tiny subset, and behavior-history use (offline, CPU)."""

import dataclasses
import inspect
import math

import numpy as np
import polars as pl
import pytest
import torch

from recsys import BENCHMARKS, metrics, run
from recsys.algorithms.autoint import AutoInt
from recsys.algorithms.bst import BST
from recsys.algorithms.dcn import DCN
from recsys.algorithms.dcnv2 import DCNv2
from recsys.algorithms.deepfm import DeepFM
from recsys.algorithms.dien import DIEN
from recsys.algorithms.din import DIN
from recsys.algorithms.eta import ETA
from recsys.algorithms.finalmlp import FinalMLP
from recsys.algorithms.finalnet import FinalNet
from recsys.algorithms.fm import FM
from recsys.algorithms.gdcn import GDCN
from recsys.algorithms.lr import LR
from recsys.algorithms.masknet import MaskNet
from recsys.algorithms.sim import SIM
from recsys.algorithms.widedeep import WideDeep
from recsys.algorithms.xdeepfm import XDeepFM

SMALL = {"hidden": (64, 64)}
CASES = {
    "lr": (LR, {}),
    "fm": (FM, {}),
    "deepfm": (DeepFM, SMALL),
    "widedeep": (WideDeep, SMALL),
    "dcn": (DCN, SMALL),
    "dcnv2": (DCNv2, SMALL),
    "dcnv2-stacked-lowrank": (DCNv2, {**SMALL, "structure": "stacked", "low_rank": 8}),
    "xdeepfm": (XDeepFM, {**SMALL, "cin": (32, 32)}),
    "autoint": (AutoInt, {}),
    "autoint+": (AutoInt, {**SMALL, "dnn": True}),
    "finalmlp": (FinalMLP, {"mlp1": (64, 64), "mlp2": (64,), "fs_hidden": (64,), "heads": 4}),
    "din": (DIN, {}),
    "din-dice": (DIN, {"dice": True}),
    "dien": (DIEN, {}),
    "gdcn": (GDCN, SMALL),
    "gdcn-stacked": (GDCN, {**SMALL, "structure": "stacked"}),
    "finalnet": (FinalNet, {}),
    "finalnet-1b": (FinalNet, {"blocks": 1}),
    "masknet": (MaskNet, {}),
    "masknet-parallel": (MaskNet, {"structure": "parallel", "blocks": 2}),
    "bst": (BST, SMALL),
    "sim": (SIM, {**SMALL, "top_k": 5, "short_len": 3}),
    "eta": (ETA, {**SMALL, "top_k": 5, "short_len": 3}),
}


@pytest.fixture(scope="module")
def data():
    return BENCHMARKS["synthetic-ctr"].prepare()


@pytest.mark.parametrize("name", CASES)
def test_runs_on_synthetic(name):
    cls, params = CASES[name]
    record = run(cls, "synthetic-ctr", seed=0, results_dir=None,
                 params={"epochs": 3, "batch_size": 512, "device": "cpu", **params})
    assert 0.5 < record["auc"] <= 1.0
    assert math.isfinite(record["logloss"])


@pytest.mark.parametrize("name", CASES)
def test_overfits_tiny_subset(name, data):
    cls, params = CASES[name]
    torch.manual_seed(0)
    tiny = data.train.head(256)
    params = {**params, "epochs": 60, "batch_size": 64, "lr": 5e-2 if cls is LR else 1e-2,
              "embedding_reg": 0.0, "device": "cpu"}
    if "dropout" in inspect.signature(cls).parameters:
        params["dropout"] = 0.0
    model = cls(**params)
    model.fit(dataclasses.replace(data.train_data(0), train=tiny))
    assert metrics.auc(tiny["label"].to_numpy(), model.predict(tiny)) >= 0.95


@pytest.mark.parametrize("cls", [DIN, DIEN, BST, SIM, ETA])
def test_reads_history(cls, data):
    torch.manual_seed(0)
    model = cls(epochs=2, batch_size=512, device="cpu")
    model.fit(data.train_data(0))
    rows = data.test.head(200)
    blank = rows.with_columns(pl.Series("history", np.full((rows.height, rows["history"].dtype.size), -1,
                                                           dtype=np.int32)))
    assert np.abs(model.predict(rows) - model.predict(blank)).max() > 1e-4


def test_finalnet_ensemble_is_the_mean_block_probability():
    z = torch.randn(2, 100) * 5
    p = torch.sigmoid(FinalNet.ensemble(list(z)))
    assert torch.allclose(p, torch.sigmoid(z).mean(0), atol=1e-6)


def test_host_staging_when_data_does_not_fit(data, monkeypatch):
    import recsys.algorithms._ctr as ctr

    monkeypatch.setattr(ctr, "fits_on_device", lambda nbytes, device: False)
    model = DIN(epochs=1, batch_size=512, device="cpu")
    model.fit(data.train_data(0))
    assert model._x.dtype == torch.int32 and model._h.dtype == torch.int32
    assert model.predict(data.test.head(50)).shape == (50,)
