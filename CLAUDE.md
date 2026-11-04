# CLAUDE.md

Guidance for Claude Code in this repository: a standardized benchmark for recommendation algorithms. [docs/dev.md](docs/dev.md) is the design reference and [docs/acceptance.md](docs/acceptance.md) the acceptance criteria every change must keep passing. Read them before changing core abstractions.

## Commands

- `uv sync` installs dependencies. Re-run it after editing `pyproject.toml`. Always use `uv run ...`.
- `uv run pytest` runs the offline suite: synthetic data only, no network (enforced by `tests/conftest.py`), parallel via xdist, and it must stay under 60 s (acceptance G4). Use `-n0` to debug a single file.
- `uv run pytest -m network | reference | perf` run the opt-in gates: real downloads, published-number reproduction (`tests/reference.toml`), and time/memory bounds.
- Long GPU experiments run on the user's RTX 4090 workstation: `ssh workstation`, repo at `~/research/rec-bench` tracking the pushed branch (commit + push here, `git pull` there; `uv` is at `~/.local/bin/uv`). Use `nohup ... > logs/<name>.log &` there and keep memory-heavy jobs (Criteo/Avazu, ~7 GB frames) from running concurrently with the parallel test suite.
- `uv run recsys list {datasets,benchmarks,algorithms}`
- `uv run recsys run --algo ease,sasrec --benchmark ml-1m-loo --seeds 0,1,2 [-p key=json_value]`
- `uv run recsys report`
- Datasets download on first use into `$RECSYS_DATA` (default `~/.cache/recsys`). For a quick real-data check, use `ml-100k-loo` (5 MB).

## Rules

- **Keep it simple.** Few files and few concepts. Registries are plain dicts (`DATASETS`, `BENCHMARKS`, `ALGORITHMS`). Don't add plugin machinery, YAML configs, or framework layers.
- **No backward-compatibility shims.** Replace designs outright; this is pre-1.0.
- **No Lightning.** Neural models subclass `TorchRecommender` (`src/recsys/algorithms/trainer.py`) and implement `build` / `batches` / `loss`.
- **Benchmarks are pinned.** Never change an existing `Benchmark` in `benchmarks/catalog.py` once results exist; add a new name instead. `Benchmark.version` hashes the full definition. The first metric selects models and must be higher-is-better.
- **Ranking is full-catalog**, with known items masked by the evaluator. Don't add sampled-negative metrics.
- `results/` is append-only and git-ignored. Never edit it by hand.

## Architecture in one screen

- `data/`:
  - `Dataset` frozen dataclasses with `files_present` / `download` / `read`
  - `load()` returns `Interactions(events, users, items)` with raw ids; `events` has `user`, `item`, `timestamp` plus raw signals
- `protocol.py`: preprocessing steps (`MinValue`, `Binarize`, `Dedup`, `KCore`) and splits (`LeaveLastOut`, `TemporalSplit`, `RandomSplit`), all frozen dataclasses.
- `benchmarks/base.py`:
  - `Benchmark.prepare()` returns `BenchmarkData`, cached in memory
    - ranking: users and items encoded to `0..n-1`
    - CTR: field vocabularies built from train, with `0` for unseen values
  - `TrainData` is what `fit` receives
  - `History` is per-user time-ordered items in CSR layout, with `.matrix()` and `.sequences()`
- `evaluate.py`: `evaluate(algo, data, split)`.
  - Ranking calls `algo.score(users, history)`, then masks known items and takes the top-k on the scores' device.
  - CTR calls `algo.predict(rows)`.
- `algorithms/`:
  - `Recommender` declares `tasks` ⊆ `{"ranking", "ctr"}`
  - each algorithm is registered in `ALGORITHMS` in `algorithms/__init__.py`
  - `tests/test_algorithms.py` runs every registered algorithm automatically
- `harness.py`: `run` / `run_all` seed, fit with validation-based early stopping, evaluate on test, and append to `results/<benchmark>.parquet` via `store.py`.

## Adding things

- **Algorithm:** add a module in `src/recsys/algorithms/`, then an entry in `ALGORITHMS`. Mark it as implemented in [docs/methods.md](docs/methods.md).
- **Dataset:** add a module in `src/recsys/data/`, then an entry in `DATASETS`, then a row in [docs/datasets.md](docs/datasets.md).
- **Benchmark:** add an entry in `benchmarks/catalog.py`.

## Platform notes

- On macOS, device `auto` resolves to MPS. `torch.sparse.mm` with COO tensors works on MPS; CSR does not.
- On Windows, torch comes from the CUDA index pinned in `pyproject.toml`. Don't add that marker for other platforms.
