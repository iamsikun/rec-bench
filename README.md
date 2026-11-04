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

```python
import numpy as np
import recsys


class ItemKNN(recsys.Recommender):
    tasks = frozenset({"ranking"})

    def __init__(self, k: int = 100):
        self.k = k

    def fit(self, data):
        X = data.history.matrix()                      # users x items, scipy CSR
        sim = (X.T @ X).toarray()
        np.fill_diagonal(sim, 0)
        sim[sim < np.sort(sim, 0)[-min(self.k, len(sim))]] = 0  # keep top-k neighbours per item
        self.sim = sim

    def score(self, users, history):
        return history.matrix(users) @ self.sim        # evaluator masks already-seen items


recsys.run_all(ItemKNN, benchmarks="all", seeds=[0, 1, 2])
```

For neural models, subclass `recsys.TorchRecommender` and implement `build`, `batches`, and `loss`. The built-in loop handles the device, the optimizer, validation, early stopping, and restoring the best weights. To include an algorithm in the CLI and the test suite, add it to `ALGORITHMS` in `src/recsys/algorithms/__init__.py`.

## What's included

| Algorithms | Task |
|---|---|
| Popularity | ranking, ctr |
| EASE | ranking |
| BPR-MF | ranking |
| LightGCN | ranking |
| SASRec (full-softmax CE) | ranking |
| DeepFM | ctr |

**Benchmarks:**

- `ml-{100k,1m,20m}-loo`
- `amazon-games-loo`
- `ml-1m-temporal`
- `ml-1m-ctr`
- `kuairand-ctr`
- `kuairec-ctr`
- `synthetic-{ranking,ctr}`

Run `recsys list benchmarks` for their versions and descriptions.

## Documentation

- [docs/dev.md](docs/dev.md): design, contracts, how to extend, roadmap
- [docs/methods.md](docs/methods.md): catalog of published methods and implementation status
- [docs/datasets.md](docs/datasets.md): datasets and acquisition

## Development

```bash
uv run pytest
```

The tests use only the synthetic dataset. They need no network access and finish in seconds.
