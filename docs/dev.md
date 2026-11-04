# Developer guide

This is the design reference for `recsys`, a standardized benchmark for recommendation algorithms. The library has three parts:

1. **Datasets.** Classes that download the original public data and load it into one standard in-memory format.
2. **Benchmarks.** Frozen, versioned evaluation protocols (dataset × preprocessing × split × task × metrics).
3. **Algorithms.** One `Recommender` base class. Any algorithm that subclasses it runs on every compatible benchmark with one call.

Related documents:

- [methods.md](methods.md): the catalog of published methods, with what is implemented
- [datasets.md](datasets.md): the datasets and how they are acquired
- [acceptance.md](acceptance.md): the acceptance criteria the repository is held to, each with an automated check

## Principles

1. **Simple over clever.** Few concepts, few files, and no plugin magic. Registries are plain dicts. The core can be read in one sitting:
   - `data/base.py`
   - `protocol.py`
   - `benchmarks/base.py`
   - `evaluate.py`
   - `algorithms/base.py`
   - `algorithms/trainer.py`
   - `harness.py`
2. **Benchmarks are pinned.** A benchmark fixes everything that affects its numbers: preprocessing, the split and its seed, the candidate set, metrics, and the pinned dataset checksums. Nothing is configurable at run time; a new protocol gets a new name. `Benchmark.version` hashes all of it, and `prepare()` checks the pinned dataset statistics.
3. **Full-catalog ranking.** Ranking metrics score every item. The user's known items are masked, and exact ties are broken at random. Sampled-negative metrics are not offered, because they can reorder algorithms (Krichene & Rendle, KDD 2020).
4. **The test split is never seen while fitting.** Algorithms get the train data and a validation callback. Model selection and tuning use validation only.
5. **One interface, many paradigms.** An algorithm declares the tasks it supports, the inputs it requires, and whether it is inductive. The harness runs it on every compatible benchmark and skips the others with a logged reason.
6. **GPU first.** Device `auto` (the default everywhere) picks the CUDA GPU with the most free memory, then Apple MPS, and falls back to the CPU only when no accelerator exists (`recsys/device.py`). Closed-form models and the evaluator compute on the GPU too.
7. **Reproducible by default.** Every result row records:
   - the benchmark version, algorithm, params, and seed
   - the metrics
   - fit and evaluation time, peak memory, and the device
   - the code SHA and whether the tree was dirty
   - package versions
8. **No Lightning and no backward-compatibility shims.** Neural models use the built-in training loop. The project is pre-1.0: when a better design appears, it replaces the old one.

## Concepts

### Tasks

There are exactly two tasks. They are defined by what the evaluator asks the algorithm for.

| Task | The algorithm implements | Evaluation |
|---|---|---|
| `ranking` | `score(users, history) -> (len(users), n_items)` **or** `recommend(users, history, k) -> (len(users), k)` item ids | Full-catalog top-k. Known items are masked, and ties are broken at random. Metrics: `ndcg@k`, `recall@k` (hits / #relevant), `nrecall@k` (hits / min(k, #relevant), as in Liang et al. 2018), `hr@k`, `precision@k`, `mrr@k` |
| `ctr` | `predict(rows) -> P(label = 1)` | `auc` and `logloss` over held-out rows |

All of these use the same `ranking` task:

- **General recommenders** read `history` as a sparse matrix.
- **Sequential models** read it as time-ordered sequences.
- **Retrieval models** answer `recommend` with a top-k inner product.
- **Generative models** decode item ids in `recommend`. Repeated ids count once, and `-1` pads short lists.

What differs between them is the benchmark's split.

### Datasets (`recsys/data/`)

```python
MovieLens("1m").load()  # downloads on first call, returns Interactions(events, users, items)
```

- **`events`** has at least `user` and `item`, plus `timestamp` in integer Unix seconds, plus raw signal columns. Sources without times set `has_time = False` and use 0.
- **`users` and `items`** are optional side tables keyed by `user` / `item`.
- **Raw ids are kept.** The benchmark encodes them.
- **Pinned downloads.** Every downloaded artifact is pinned by SHA-256, or on the Hugging Face Hub by revision.
  - Downloads stream to a `.part` file and are renamed only when complete and verified.
  - Archives extract into a staging directory first.
  - Files are cached under `$RECSYS_DATA` (default `~/.cache/recsys`).
- **Version hashing.** Datasets are frozen dataclasses. Their `repr` and `pins()` feed `Benchmark.version`.

### Protocol (`recsys/protocol.py`)

Frozen dataclasses that act as pure functions on a polars DataFrame.

**Preprocessing steps:**

| Step | What it does |
|---|---|
| `MinValue` | keeps rows whose column is at least a threshold (e.g. implicit feedback) |
| `Binarize` | adds a CTR `label` column |
| `Dedup` | keeps the first interaction of each user–item pair |
| `KCore` | iterative k-core filter |
| `MinCount` | one-pass minimum count on one column |
| `LogSquareBucket` | the BARS discretization for numeric features |

**Splits:**

| Split | What it does |
|---|---|
| `LeaveLastOut` | per user, in time order: last event → test, second-to-last → val |
| `TemporalSplit` | global time cutoffs |
| `RandomSplit` | random rows, pinned seed |
| `HeldOutUsers` | Liang et al. strong generalization: val/test users are unseen in training, with a fold-in history and targets |
| `Predefined` | the split published with the dataset |

A split may add a boolean `target` column to val and test. Rows with `target=False` are fold-in history: visible when scoring that split, but never trained on or scored.

### Benchmarks (`recsys/benchmarks/`)

```python
Benchmark(
    name="ml-1m-loo",
    dataset=MovieLens("1m"),
    task="ranking",
    split=LeaveLastOut(),
    metrics=("ndcg@10", "recall@10", "ndcg@20", "recall@20", "mrr@10"),
    preprocess=(Dedup(), KCore(5)),
    user_fields=("gender", "age", "occupation"), item_fields=("genre",),
    stats=(6040, 3416, 987531, 6040, 6040),
    description=..., leakage=..., reference=...,
)
```

- **Model selection.** The first metric selects models and must be higher-is-better.
- **Ranking benchmarks.**
  - Users and items are encoded to `0..n-1`.
  - Validation is scored with `train` (plus the val fold-in) as history. Test is scored with `train ∪ val` (plus the test fold-in).
  - `user_fields` / `item_fields` offer side features as arrays indexed by encoded id.
- **CTR benchmarks.**
  - `fields` are categorical inputs whose vocabularies are built from train. Values seen fewer than `min_count` times map to 0.
  - `history > 0` adds each row's earlier positive items of the same user. These come from train for train and val rows, and from train + val for test rows.
- **What a benchmark provides.** `provides` lists `user_features`, `item_features`, `behavior`, and `cold_users`.
- **Caching.** `prepare()` is cached in memory and on disk under `$RECSYS_DATA/prepared/<name>-<version>/`.
- **Documentation.** `recsys list benchmarks -v` prints each benchmark's full protocol: dataset, preprocessing, split, leakage notes, candidates, metrics, pinned stats, and the matching literature protocol.

### Algorithms (`recsys/algorithms/`)

```python
class Recommender(ABC):
    tasks: ClassVar[frozenset[str]]        # {"ranking"}, {"ctr"}, or both
    requires: ClassVar[frozenset[str]]     # e.g. {"behavior"}, {"user_features", "item_features"}
    inductive: ClassVar[bool]              # scores unseen users from their history (held-out-user benchmarks)
    space: ClassVar[dict[str, list]]       # hyperparameter grid for recsys.tune
    def fit(self, data: TrainData) -> None: ...
    def score(self, users, history): ...         # ranking, dense
    def recommend(self, users, history, k): ...  # ranking, top-k ids
    def predict(self, rows): ...                 # ctr
```

`TrainData` carries:

- `train`: the encoded training frame
- `history`: the training interactions as a `History`, with `.matrix(users)` and `.sequences(users, max_len)`
- `n_users`, `n_items`, and `fields` (cardinalities)
- `user_features` and `item_features`
- `seed`
- `validate()`: the model-selection metric on validation

`TorchRecommender` (`algorithms/trainer.py`) is the built-in loop. A subclass implements three methods:

- `build(data)`: creates layers and device-resident training tensors, registered with `register_buffer(..., persistent=False)`
- `batches(data)`: yields batches that are already on the device
- `loss(batch)`

The loop handles everything else:

- choosing the device
- the Adam optimizer and epochs
- validation every `eval_every` epochs
- early stopping with `patience`, then restoring the best weights
- recording `losses` and `val_metrics`

### Harness, tuning and results

```python
recsys.run("sasrec", "ml-1m-loo", seed=0, params={"max_len": 200})
recsys.run_all(MyRecommender, benchmarks="all", seeds=[0, 1, 2])   # every compatible benchmark
recsys.tune("ease", "ml-1m-loo", budget=20)                        # same budget for every algorithm
recsys.summary("results")                                          # mean ± std, per benchmark version
```

```bash
recsys list {datasets,benchmarks,algorithms} [-v]
recsys run --algo ease,sasrec --benchmark ml-1m-loo --seeds 0,1,2 -p max_len=200 [--save-per-user]
recsys report [--benchmark ml-1m-loo] [--compare ease,sasrec]
```

- **Results store.** Results are appended to `results/<benchmark>.parquet`.
- **Per-user detail.** `--save-per-user` stores per-user metrics (ranking) or per-row predictions (CTR) under `results/detail/`.
- **Comparisons.** `report --compare` runs a paired t-test and a Wilcoxon test over users, or over seeds when no detail is stored.
- **Versions.** The report never averages across benchmark versions.

## Layout

```
src/recsys/
  __init__.py            public API
  device.py              GPU-first device choice
  data/                  base.py, download.py, one module per source (+ synthetic.py)
  protocol.py            preprocessing steps and splits
  benchmarks/            base.py (Benchmark, BenchmarkData, TrainData, History), catalog.py
  metrics.py             ranking and CTR metric functions
  evaluate.py            full-catalog ranking and CTR evaluators
  algorithms/            base.py, trainer.py, one module per method, _shared helpers
  harness.py             run / run_all / tune
  store.py               parquet results, summaries, paired tests
  cli.py
examples/                the minimal classical and neural algorithms shown in the README
tests/                   offline suite (default) plus -m network / -m reference / -m perf
```

## Extending

### Add a dataset

1. Create a `@dataclass(frozen=True)` subclass of `Dataset` in `src/recsys/data/<source>.py` with `dirname`.
2. Implement `pins()`, `files_present(root)`, `download(root)`, and `read(root) -> Interactions`.
   - In `download`, use `fetch(url, root, sha256)` for archives and `download(url, dest, sha256)` for single files.
   - In `read`, rename the id columns to `user` / `item` and convert the time column to `timestamp` in Unix seconds.
3. Add it to `DATASETS` in `data/__init__.py` and to the table in [datasets.md](datasets.md).
4. Build a benchmark on it (next section) and run every algorithm: `recsys.run_all(list(recsys.ALGORITHMS), "<benchmark>")`.

If the source requires accepting a license manually, `download` raises an error saying where to get the files and where to put them.

### Add a benchmark

1. Add a `Benchmark(...)` to `benchmarks/catalog.py` with `description`, `leakage`, and `reference`.
2. Run `uv run python -c "import recsys; print(recsys.BENCHMARKS['<name>'].prepare().stats)"` and pin the printed tuple as `stats=`.
3. Never edit an existing definition once results exist; add a new name instead.
4. Run everything compatible: `recsys.run_all(list(recsys.ALGORITHMS), "<name>")`.

### Add a classical algorithm

1. Subclass `Recommender` in `src/recsys/algorithms/<name>.py`. Set `tasks`, `inductive`, and `space`, and keep heavy math on `resolve_device(device)`.
2. Implement `fit`, plus `score` for ranking or `predict` for CTR.
3. Add it to `ALGORITHMS` and mark it **Implemented** in [methods.md](methods.md). The test suite then runs it on every compatible synthetic benchmark.

[examples/minimal_algo.py](../examples/minimal_algo.py) is a complete 14-line example. Run it everywhere with `recsys.run_all(MyAlgo, seeds=[0, 1, 2])`.

### Add a neural algorithm

1. Subclass `TorchRecommender` and implement `build`, `batches`, and `loss`, plus `score` or `predict`.
2. Keep training tensors on the device and initialize embeddings small (std ≈ 0.02 to 0.1).
3. Make paper defaults the constructor defaults.

[examples/minimal_torch_algo.py](../examples/minimal_torch_algo.py) is a complete 30-line BPR-MF. The loop gives it validation, early stopping, best-weight restore, and seeding. Run it everywhere with `recsys.run_all(TinyMF, seeds=[0, 1, 2])`.

### Add a generative or retrieval algorithm

1. Implement `recommend(users, history, k)` instead of `score`. It returns a `(len(users), k)` id tensor on the device, best first.
   - Exclude history items.
   - Pad with `-1` when fewer than k valid ids are decoded.
2. Constrain decoding to valid items (TIGER uses a semantic-id prefix trie).
3. The evaluator scores the returned ids directly, and repeats count once.

Run it with `recsys.run_all(MyGenerative, seeds=[0, 1, 2])`.

## Development

- `uv sync`, then `uv run pytest`. The tests are offline, use synthetic data, and run in parallel.
- `uv run pytest -m network` downloads and checks real datasets.
- `-m reference` reproduces published numbers from `tests/reference.toml`.
- `-m perf` checks the time and memory bounds.
- `results/` is append-only and git-ignored. Never edit it by hand.

## Roadmap

1. **Throughput parity** with RecBole and FuxiCTR: mixed precision, `torch.compile`, and sampled softmax for million-item catalogs (acceptance F6, F7).
2. **Tier-2 methods** from [methods.md](methods.md), in its suggested order.
3. **More protocols:** session datasets with session splits, cold-start item splits, and beyond-accuracy metrics as new benchmarks.

## Non-goals

- Configurable metrics or splits on an existing benchmark.
- More than one training framework.
- Shipping dataset files in the repository.
