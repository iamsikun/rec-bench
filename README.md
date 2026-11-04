# rec-bench

[![PyPI version](https://img.shields.io/pypi/v/rec-benchmark.svg)](https://pypi.org/project/rec-benchmark/)
[![Python versions](https://img.shields.io/pypi/pyversions/rec-benchmark.svg)](https://pypi.org/project/rec-benchmark/)
[![License](https://img.shields.io/pypi/l/rec-benchmark.svg)](https://github.com/iamsikun/rec-bench/blob/main/LICENSE)

A standardized benchmark for recommendation algorithms. It provides:

- **Datasets** that download the original public data and load it into one standard format.
- **Benchmarks** that are pinned and versioned: dataset × preprocessing × split × task × metrics.
- **One `Recommender` interface**, so a new algorithm runs on every compatible benchmark with a few lines of code.

Ranking metrics score the full catalog, and every run is written to a parquet results log.

## Install

```bash
pip install rec-benchmark    # or: uv add rec-benchmark; the import package and CLI are `recsys`
```

From a clone:

```bash
uv sync
export RECSYS_DATA=~/data/recsys   # optional; where datasets are downloaded (default ~/.cache/recsys)
```

## Run

```bash
uv run recsys list benchmarks
uv run recsys run --algo popularity,ease,sasrec --benchmark ml-1m-loo --seeds 0,1,2
uv run recsys run --algo deepfm --benchmark ml-1m-ctr -p epochs=5 -p dim=32
uv run recsys report
```

The same from Python:

```python
import recsys

recsys.run("sasrec", "ml-1m-loo", seed=0, params={"max_len": 200})
recsys.run_all(["ease", "lightgcn"], benchmarks="all", seeds=[0, 1, 2])
recsys.summary("results")["ml-1m-loo"]
```

## Bring your own algorithm

A complete classical algorithm in 14 lines ([examples/minimal_algo.py](examples/minimal_algo.py)); `run_all` runs it on every compatible benchmark:

```python
import numpy as np
import torch

import recsys


class RecentPopularity(recsys.Recommender):
    tasks = frozenset({"ranking"})
    inductive = True

    def fit(self, data, window=10):
        self.window = window

    def score(self, users, history):
        recent = history.sequences(np.arange(history.n_users), self.window).ravel()
        counts = np.bincount(recent[recent >= 0], minlength=history.n_items)
        return torch.tensor(counts, dtype=torch.float32).repeat(len(users), 1)


if __name__ == "__main__":
    recsys.run_all(RecentPopularity, seeds=[0])
```

A neural model in 30 lines ([examples/minimal_torch_algo.py](examples/minimal_torch_algo.py)). The built-in loop handles the device (GPU first), the optimizer, validation, early stopping, best-weight restore and seeding:

```python
import torch
import torch.nn.functional as F
from torch import nn

import recsys
from recsys.algorithms.trainer import minibatches


class TinyMF(recsys.TorchRecommender):
    tasks = frozenset({"ranking"})

    def __init__(self, dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, data):
        self.users = nn.Embedding(data.n_users, self.dim)
        self.items = nn.Embedding(data.n_items, self.dim)
        nn.init.normal_(self.users.weight, std=0.1)
        nn.init.normal_(self.items.weight, std=0.1)
        self.register_buffer("u", torch.tensor(data.train["user"].to_numpy()), persistent=False)
        self.register_buffer("i", torch.tensor(data.train["item"].to_numpy()), persistent=False)

    def batches(self, data):
        for idx in minibatches(len(self.u), self.batch_size, self.device):
            yield self.u[idx], self.i[idx], torch.randint(data.n_items, (len(idx),), device=self.device)

    def loss(self, batch):
        u, i, j = batch
        return -F.logsigmoid((self.users(u) * (self.items(i) - self.items(j))).sum(1)).mean()

    def score(self, users, history):
        return self.users.weight[torch.as_tensor(users, device=self.device)] @ self.items.weight.T


if __name__ == "__main__":
    recsys.run_all(TinyMF, seeds=[0])
```

Retrieval and generative models implement `recommend(users, history, k)` (top-k item ids) instead of `score`. To include an algorithm in the CLI and the test suite, add it to `ALGORITHMS` in `src/recsys/algorithms/__init__.py`.

## Algorithms

Every Tier-1 method of [docs/methods.md](docs/methods.md):

| Key | Method | Family | Task |
|---|---|---|---|
| `random` | Random | classical | ranking |
| `popularity` | Popularity | classical | ranking, ctr |
| `itemknn` | ItemKNN | classical | ranking |
| `userknn` | UserKNN | classical | ranking |
| `slim` | SLIM | classical | ranking |
| `rp3beta` | RP3beta | classical | ranking |
| `puresvd` | PureSVD | classical | ranking |
| `ials` | iALS / WMF | classical | ranking |
| `sknn` | Session kNN | classical | ranking |
| `ease` | EASE | classical | ranking |
| `bpr` | BPR-MF | neural CF / graph / autoencoder | ranking |
| `lightgcn` | LightGCN | neural CF / graph / autoencoder | ranking |
| `neumf` | NeuMF | neural CF / graph / autoencoder | ranking |
| `multvae` | Mult-VAE | neural CF / graph / autoencoder | ranking |
| `multdae` | Mult-DAE | neural CF / graph / autoencoder | ranking |
| `gru4rec` | GRU4Rec | sequential | ranking |
| `srgnn` | SR-GNN | sequential | ranking |
| `sasrec` | SASRec (full-softmax CE) | sequential | ranking |
| `gsasrec` | gSASRec | sequential | ranking |
| `bert4rec` | BERT4Rec | sequential | ranking |
| `dssm` | DSSM | two-tower retrieval | ranking |
| `youtubednn` | YouTube DNN | two-tower retrieval | ranking |
| `twotower_logq` | Two-tower with logQ correction | two-tower retrieval | ranking |
| `tiger` | TIGER (semantic-id generative) | generative | ranking |
| `hstu` | HSTU (generative sequential transducer) | generative | ranking |
| `lr` | Logistic regression | CTR feature interaction | ctr |
| `fm` | FM | CTR feature interaction | ctr |
| `widedeep` | Wide & Deep | CTR feature interaction | ctr |
| `deepfm` | DeepFM | CTR feature interaction | ctr |
| `dcn` | DCN | CTR feature interaction | ctr |
| `dcnv2` | DCN-V2 | CTR feature interaction | ctr |
| `xdeepfm` | xDeepFM | CTR feature interaction | ctr |
| `autoint` | AutoInt / AutoInt+ | CTR feature interaction | ctr |
| `finalmlp` | FinalMLP | CTR feature interaction | ctr |
| `din` | DIN | CTR user behavior | ctr |
| `dien` | DIEN | CTR user behavior | ctr |

## Benchmarks

`recsys list benchmarks -v` prints each benchmark's full protocol: dataset, preprocessing, split, leakage notes, pinned statistics and the matching literature reference.

| Name | Task | Protocol |
|---|---|---|
| `synthetic-ranking` | ranking | Generated sequences, leave-last-out, with side features. |
| `synthetic-heldout` | ranking | Generated data, strong generalization (held-out users with fold-in). |
| `synthetic-ctr` | ctr | Generated ratings >= 4 as clicks, with behavior history. |
| `ml-100k-loo` | ranking | MovieLens-100K, all ratings as implicit feedback. |
| `ml-1m-loo` | ranking | MovieLens-1M, all ratings as implicit feedback. |
| `ml-20m-loo` | ranking | MovieLens-20M, all ratings as implicit feedback. |
| `amazon-games-loo` | ranking | Amazon Reviews 2023, Video_Games. |
| `amazon14-beauty-loo` | ranking | Amazon Reviews 2014 5-core Beauty. |
| `amazon14-sports-loo` | ranking | Amazon Reviews 2014 5-core Sports_and_Outdoors. |
| `amazon14-toys-loo` | ranking | Amazon Reviews 2014 5-core Toys_and_Games. |
| `gowalla-lightgcn` | ranking | gowalla 10-core, LightGCN release train/test; 5% of train held out for validation. |
| `yelp2018-lightgcn` | ranking | yelp2018 10-core, LightGCN release train/test; 5% of train held out for validation. |
| `amazonbook-lightgcn` | ranking | amazon-book 10-core, LightGCN release train/test; 5% of train held out for validation. |
| `ml-1m-temporal` | ranking | MovieLens-1M ratings >= 4, last 10% of time as test, previous 10% as val. |
| `ml-20m-strong` | ranking | MovieLens-20M ratings >= 4, users with >= 5 ratings, 10k val + 10k test held-out users, 80% fold-in. |
| `ml-1m-ctr` | ctr | MovieLens-1M, rating >= 4 as positive, random 80/10/10 split, 50-item behavior history. |
| `kuairand-ctr` | ctr | KuaiRand-Pure standard logs, is_click label, temporal 80/10/10 split, behavior history. |
| `avazu-x4` | ctr | Avazu, BARS Avazu_x4_001 split (8:1:1 random), hour/weekday/weekend from the hour field. |
| `criteo-x4` | ctr | Criteo, BARS Criteo_x4_001 split (8:1:1 random), numeric fields bucketed as floor(ln(x)^2). |
| `kuairec-ctr` | ctr | KuaiRec small matrix, watch_ratio >= 2 as positive, random 80/10/10 split. |

## Documentation

- [docs/dev.md](docs/dev.md): design, contracts, how to extend, roadmap
- [docs/acceptance.md](docs/acceptance.md): acceptance criteria and how each is checked
- [docs/methods.md](docs/methods.md): catalog of published methods and implementation status
- [docs/datasets.md](docs/datasets.md): datasets and acquisition

## Development

```bash
uv run pytest                 # offline suite on synthetic data
uv run pytest -m network      # download and check the real datasets
uv run pytest -m reference    # reproduce published numbers (GPU, hours)
uv run pytest -m perf         # time and memory bounds
```
