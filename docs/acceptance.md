# rec-bench: Acceptance Criteria

Adopted in full on 2026-11-03. `tests/` automates each check (offline by default; `-m network`, `-m reference`, `-m perf` for the ones that need real data). The verification status is tracked at the end of this file.

These criteria come from the owner's seven goals, not from the current code. Each one gives a statement, a one-line reason, a verification procedure with a pass threshold, and a priority:

- **Must:** blocks acceptance.
- **Should:** expected before calling v1 credible.
- **Deferred:** a target to record now and enforce later.

Conventions:

- Commands run from the repo root with `uv run`.
- "Fixture" means a small offline dataset, such as `Synthetic` or a ~1k-row file under `tests/`.
- "Network test" means a test marked `@pytest.mark.network`, which is skipped by default and run with `uv run pytest -m network`.
- Where a check needs a test that doesn't exist yet, the procedure says what the test must assert. Writing that test is part of meeting the criterion.

---

## A. Datasets and acquisition

**A1 (Must): No data in the repo.** No dataset file is tracked by git or shipped in the wheel. Only test fixtures under `tests/` of 1 MB or less are allowed.
- *Why:* This is owner goal 6. It also avoids licensing problems and repo bloat.
- *Verify:*
  - `git ls-files | grep -Ev '^tests/' | grep -Ei '\.(csv|dat|tsv|parquet|zip|gz|json|jsonl|npz|pkl)$'` prints nothing.
  - `uv build && unzip -l dist/*.whl` lists no data files.
  - `datasets/` and `results/` are in `.gitignore`.

**A2 (Must): Download from the original source.** Every `Dataset` subclass's `load()` downloads from the original publisher on first call, with no manual step:
- GroupLens for MovieLens
- Zenodo for KuaiRec and KuaiRand
- `McAuley-Lab` on Hugging Face for Amazon 2023
- the official hosts for Criteo, Avazu, and others

Datasets whose license requires manual acceptance are the exception. For those, `download()` must raise an error that gives the URL and the target path.
- *Why:* This is owner goal 6. A third-party mirror can silently change the data.
- *Verify:* A network test runs `Dataset(...).load()` into an empty temp `$RECSYS_DATA` for the smallest variant of each class, and every one passes. Run `grep -rn "http" src/recsys/data/` and review each host against the original-source list in `docs/datasets.md`.

**A3 (Must): Downloads are pinned.** Every downloaded artifact has a pinned expected SHA-256, or a pinned Hugging Face revision hash for HF datasets. A mismatch raises an error and leaves nothing in the cache.
- *Why:* If an upstream file changes silently, a pinned benchmark measures different data under the same version.
- *Verify:*
  - A unit test monkeypatches the downloader to serve corrupted bytes. It asserts that an error is raised and that no file remains in the cache, not even `.part` files.
  - `grep -c sha256` in each dataset module is at least 1, or the module passes `revision=` to the HF snapshot.

**A4 (Must): Downloads are atomic and safe to resume.** An interrupted download never leaves a file that a later `load()` treats as valid.
- *Why:* A truncated archive that looks valid gives wrong numbers with no error.
- *Verify:* A unit test kills the transfer after N bytes using a local `http.server` that sends a short body, then calls `load()` again. The second call must either re-download or raise. It must never parse a partial file.

**A5 (Must): Cache location and offline reuse.** The cache root is `$RECSYS_DATA`, defaulting to a user cache directory and never the CWD. Once data is cached, `load()` makes no network calls.
- *Why:* Runs must be reproducible on clusters and offline, and they must not depend on the working directory.
- *Verify:* After a network test populates the cache, run `load()` again with sockets blocked (for example, `pytest-socket --disable-socket`, or monkeypatch `socket.socket` to raise). It succeeds. Running from `/tmp` with `RECSYS_DATA` set reads from that path.

**A6 (Must): One canonical in-memory format.** Every dataset's `load()` returns the same container type, with:
- an `events` frame containing columns `user` and `item`
- `timestamp` in one documented unit (seconds) wherever the source has time
- optional `users` and `items` side tables keyed by `user` / `item`
- *Why:* Benchmarks and splits must not contain dataset-specific code.
- *Verify:* Run a parametrized test over all registered datasets (fixture-backed, with real sources as a network test). It asserts the column names and dtypes, `timestamp` in seconds (median within 1e8–3e9), and side-table keys that are a subset of the event ids.

**A7 (Must): Coverage of the most-used datasets.** The registry includes at least:
- MovieLens 100K, 1M, and 20M
- one Amazon reviews family. Amazon 2014/2018 Beauty, Sports, and Toys are the standard for sequential and generative models such as SASRec and TIGER, so the family should be 2014/2018 or should document how it maps to them.
- Gowalla or Yelp2018, the LightGCN graph-CF standard
- Criteo and Avazu, the BARS/FuxiCTR CTR standard
- KuaiRand or KuaiRec

Should-level additions are Steam, LastFM, Yoochoose or Diginetica for session-based work, and MIND.
- *Why:* This is owner goal 2. Results are only comparable with the literature on the datasets the literature uses.
- *Verify:* `uv run recsys list datasets` contains each Must item.

**A8 (Should): Adding a dataset is a single subclass.** A new dataset needs one subclass that implements download and read, plus one registry line. No core file changes.
- *Why:* This is owner goal 2.
- *Verify:* A test defines a toy `Dataset` subclass of 40 lines or fewer (excluding the docstring) that reads a CSV fixture, registers it, and builds a benchmark on it. `git diff --stat` for the change touches only the new file and the registry line.

---

## B. Benchmark definition and protocol validity

**B1 (Must): Benchmarks are fully pinned and not configurable at run time.** A benchmark fixes the dataset variant, preprocessing, split (including its seed), candidate set, metrics with their cutoffs, and the model-selection metric. Nothing in the CLI or API can override them.
- *Why:* If a protocol can be tweaked, two results under the same benchmark name may not be comparable.
- *Verify:* Inspect `recsys run --help` and the signature of `run()`. No parameter changes the split, metrics, or filtering. A test that tries to mutate a benchmark instance raises (`FrozenInstanceError` or equivalent).

**B2 (Must): Content-addressed versioning.** `Benchmark.version` is a deterministic hash of the full definition. Changing any field changes it, and so does changing the pinned data checksum from A3.
- *Why:* Results from different protocols must never be mixed silently.
- *Verify:* A unit test creates a benchmark, then copies it with each field changed in turn (`KCore(5)` to `KCore(10)`, metric list, split, dataset variant). Each copy has a different version. Identical definitions in two separate Python processes have equal versions, so the hash doesn't depend on `PYTHONHASHSEED`.

**B3 (Must): Pinned dataset statistics.** Each benchmark stores the expected `#users`, `#items`, and `#train/#val/#test` interactions after preprocessing. `prepare()` asserts that they match exactly.
- *Why:* This catches upstream data drift, preprocessing regressions, and nondeterministic filtering.
- *Verify:* A network test runs `prepare()` on every built-in benchmark and every assertion passes. A unit test perturbs the fixture by one row and checks that the assertion fires.

**B4 (Must): The test split is never visible during training or model selection.** `fit()` receives only train data and a validation callback. Early stopping and hyperparameter selection use validation only.
- *Why:* Dacrema et al. (2019) and Sun et al. (2020) found test-set tuning in many published comparisons, which inflates results.
- *Verify:* A spy algorithm records every (user, item) and every row it can reach during `fit()`, including the full `TrainData` object graph. The test asserts an empty intersection with the test split. A second test asserts that the validation callback's labels come from the val split only.

**B5 (Must): Split invariants hold for every built-in benchmark.**
- **LeaveLastOut:** each user's test item has the maximum timestamp for that user, and val has the second-highest. Ties are broken deterministically and documented.
- **Temporal:** `max(train.ts) ≤ min(val.ts) ≤ max(val.ts) ≤ min(test.ts)`.
- **Random:** train, val, and test are disjoint and their union is the full set.
- **CTR:** no row appears in more than one split.
- *Why:* A wrong split invalidates every number on the benchmark.
- *Verify:* Parametrized `test_split_invariants` over all benchmarks: fixture-backed by default, real data as a network test.

**B6 (Must): No preprocessing leakage across the split.** Statistics that a model or benchmark derives from data come from train only, or from train ∪ val when evaluating on test. This includes CTR vocabularies, popularity counts, normalization statistics, and the item universe used for negative sampling. The ordering of k-core filtering and splitting is documented for each benchmark.
- *Why:* Ji et al. (2023) showed that global information leaking through temporal and leave-one-out splits changes algorithm rankings.
- *Verify:*
  - Unit test: a CTR field value that appears only in test maps to the OOV id 0.
  - Unit test: Popularity's counts sum to `#train` interactions at validation time and to `#train + #val` at test time.
  - The doc for each benchmark has a "Leakage notes" line.

**B7 (Should): A leak-free temporal benchmark exists.** At least one ranking benchmark uses a global temporal cutoff (`TemporalSplit`) next to the leave-last-out variant on the same dataset.
- *Why:* Leave-last-out lets a model train on interactions that happen after the test interactions of other users (Ji et al., 2023). Reporting both shows how large the effect is.
- *Verify:* `recsys list benchmarks` shows a `*-temporal` ranking benchmark, and it passes B5.

**B8 (Must): Ranking metrics use the full catalog by default.** Ranking metrics score every item and mask the user's known history. Sampled-negative metrics are either absent or live only in separately named benchmarks marked `sampled` in their name.
- *Why:* Krichene and Rendle (2020) showed that sampled metrics are inconsistent and can reverse algorithm rankings.
- *Verify:* A unit test checks that the evaluator calls the scorer with output shape `(batch, n_items)`, or with top-K over all items. `grep -rn "sample" src/recsys/evaluate*.py src/recsys/metrics.py` finds no negative sampling in the default path.

**B9 (Must): Metric definitions match references exactly.**
- NDCG@k, Recall@k, HR@k, Precision@k, and MRR@k are checked on hand-computed toy cases and against an independent implementation (`ranx`, or a plain-numpy reference in the test) on random inputs.
- AUC and LogLoss match `sklearn.metrics.roc_auc_score` and `log_loss` to 1e-6. sklearn may be a test-only dependency.
- The Recall@k denominator, `min(k, |relevant|)` or `|relevant|`, is chosen once and documented.
- *Why:* Many published discrepancies come from metric definitions that differ silently (Sun et al., 2020).
- *Verify:* `uv run pytest tests/test_metrics.py`. It runs at least 50 randomized cases per metric with max absolute error ≤ 1e-6.

**B10 (Must): Ties don't inflate scores.** An algorithm that returns constant scores gets ranking metrics equal to the random expectation within ±10% relative, for example HR@10 ≈ 10/n_items under leave-one-out. It must not get 0, and it must not get a perfect score.
- *Why:* Breaking ties in favor of the target item, for example by stable sort on item id, is a known bug that inflates metrics.
- *Verify:* A unit test with a `ConstantScorer` on a fixture benchmark with at least 500 items, averaged over the fixture users.

**B11 (Must): The history mask is correct.** At test time, items in the user's train ∪ val history get score −∞ before top-K, and the target is never masked. A target item that the user also consumed earlier is either impossible, because the benchmark deduplicates, or handled as documented.
- *Why:* A wrong mask either rewards recommending already-consumed items or hides the target.
- *Verify:*
  - Unit test: an oracle scorer gives the test item the highest score and gets HR@1 = 1.0.
  - Unit test: an "anti-oracle" scorer gives history items the highest scores and still gets the same metrics as with those items removed.

**B12 (Should): Benchmark catalog coverage.** Built-in benchmarks include, at minimum:
- ranking with leave-last-out on ML-1M and one Amazon category
- ranking with a temporal split on ML-1M or ML-20M
- the ML-20M strong-generalization protocol of Liang et al. (2018), which EASE, Mult-VAE, and iALS-revisited reference numbers use
- Gowalla with the LightGCN split
- CTR on Criteo_x1/x4 and Avazu following the BARS splits
- CTR on KuaiRand
- *Why:* Owner goals 1 and 2. Each item ties to a protocol with published reference numbers, which D2 needs.
- *Verify:* `uv run recsys list benchmarks` contains each one.

---

## C. Algorithm interface and extensibility

**C1 (Must): One base class covers all paradigms.** A single `Recommender` base class (plus at most one optional torch convenience subclass) supports:
- **CTR / pointwise prediction:** `predict(rows) → probabilities`
- **Ranking and retrieval:** `score(users, history) → (batch, n_items)` and/or `recommend(users, history, k) → top-k ids`
- **Generative:** `recommend(...)` returns the item lists the model decoded, with no full score matrix required

An algorithm declares the tasks it supports, and the evaluator chooses the method based on that.
- *Why:* This is owner goal 3. Generative models such as TIGER-style semantic-ID decoders can't produce dense full-catalog scores cheaply, so a score-only interface leaves them out.
- *Verify:* Three test algorithms subclass the base class and all evaluate successfully on fixture benchmarks:
  - a dense scorer
  - a two-tower retrieval model that exposes user and item embeddings and uses `recommend` via a top-k matmul
  - a toy generative model that returns `k` decoded ids per user without implementing `score`

**C2 (Must): A new algorithm takes a few lines of code.** A complete user-defined classical ranking algorithm, such as item-popularity-with-recency, fits in 15 lines or fewer of user code including imports. One call runs it on every compatible built-in benchmark:

```python
recsys.run_all(MyAlgo, seeds=[0])
```

- *Why:* This is the measurable form of owner goal 3.
- *Verify:* `examples/minimal_algo.py`, or the README snippet, has `wc -l` ≤ 15 excluding blank and comment lines. A test runs it against all fixture benchmarks. The README snippet must be the same code (see H2).

**C3 (Must): A new neural algorithm also takes a few lines.** A user-defined torch model, such as a 2-layer MF with BPR loss, fits in 40 lines or fewer of user code. It gets these for free, with no code from the user:
- the training loop
- device selection
- the optimizer
- validation-based early stopping and best-weight restore
- seeding
- *Why:* This is owner goal 3. The built-in loop replaces Lightning (owner goal 7).
- *Verify:* `examples/minimal_torch_algo.py` is 40 lines or fewer and passes a test that trains it on a fixture. The test asserts that validation was called, that early stopping fired with `patience=1` on a deliberately overfitting setup, and that the restored weights equal the best epoch's weights.

**C4 (Must): Compatibility is resolved automatically.** `run_all` skips incompatible benchmarks with a one-line reason instead of raising, so a CTR-only algorithm is never run on ranking benchmarks. It runs on every compatible one. Compatibility comes from declared tasks and declared required inputs, such as side features, timestamps, or sequences.
- *Why:* The goal says "automatically tested on ALL existing compatible benchmarks."
- *Verify:* A test runs `run_all` with a CTR-only algorithm and a ranking-only algorithm. The set of benchmarks actually run equals the set computed from the declared tasks and inputs, and there are no exceptions.

**C5 (Must): One data contract serves every model family.** `fit()` gets one training object that offers:
- the interaction matrix (CSR)
- time-ordered sequences (padded, with configurable max length)
- CTR feature rows with cardinalities
- side features where the benchmark has them

Sequential, matrix, graph, and generative models all read from this same object.
- *Why:* If each model family needs its own data path, the unified skeleton is broken.
- *Verify:* All six shipped algorithms use only the public `TrainData` / `History` API. Run `grep -rn "benchmarks\.\|_private" src/recsys/algorithms/` to find any reach-around into internals, and it finds none.

**C6 (Must): Registration is simple and visible.** Algorithms, datasets, and benchmarks are each registered in one obvious place, such as a plain dict or a decorator imported from one `__init__`. Running a user class by passing the class object needs no registration at all.
- *Why:* The old side-effect registration trap made new modules silently invisible.
- *Verify:*
  - `recsys.run_all(MyAlgo)` works with an unregistered class.
  - Every `.py` file under `src/recsys/algorithms/` except bases and helpers appears in `recsys list algorithms`, checked by a test that compares filenames to registry keys.

**C7 (Should): Hyperparameters are plain constructor arguments.** Every shipped algorithm has documented defaults per benchmark (or global defaults) and accepts overrides through `params` / `-p key=value`. Tuning uses a fixed, equal search budget for all algorithms.
- *Why:* Dacrema et al. (2019) and Rendle et al. (2022) showed that untuned baselines produce false progress.
- *Verify:* `recsys run --algo X -p dim=32` changes the logged `params` and `params_hash`. A tuning entry point, such as `recsys tune --budget N`, exists and records its search space in the results row. Tuning itself is a Should; the logging is a Must.

**C8 (Should): The method library grows from the catalog.** Every **Implemented** row in `docs/methods.md` maps to a registered algorithm and the reverse. Implemented methods include every Tier-1 method of each task family before Tier-2 methods are added:
- ranking: Random, Pop, ItemKNN, EASE, iALS, BPR-MF, LightGCN, SASRec
- CTR: LR, FM, DeepFM, DCNv2
- generative: at least one, such as TIGER or a decoder-only item-id model
- *Why:* Owner goal 4. Strong simple baselines are what keep a benchmark honest.
- *Verify:* A test parses the methods.md tables and checks that the set of `Status == Implemented` rows equals the set from `recsys list algorithms`, compared case-insensitively through an alias map.

---

## D. Correctness of implemented algorithms

**D1 (Must): Sanity ordering.** On every built-in ranking benchmark: Random < Popularity, and Popularity < each learned ranking model (EASE, BPR-MF, LightGCN, SASRec) on the model-selection metric. Each uses default hyperparameters, mean over 3 seeds. On CTR benchmarks, DeepFM's AUC is greater than logistic regression's, or at least greater than the 0.5 of a constant predictor.
- *Why:* This is the cheapest check that catches broken training or evaluation.
- *Verify:*
  - `uv run recsys run --algo random,popularity,ease,bpr,lightgcn,sasrec --benchmark ml-1m-loo --seeds 0,1,2 && uv run recsys report` (repeat for each benchmark). The order holds, with gaps larger than 2 × the pooled std.
  - A fast fixture version runs in the default test suite, using a synthetic dataset with planted structure.

**D2 (Must, where a matching protocol exists): Reproduce reference numbers.** On a benchmark whose protocol matches the paper, the implementation lands within the stated tolerance of the reference. Pins:

| Algo | Benchmark / protocol | Reference (verified from the source) | Target | Tolerance |
|---|---|---|---|---|
| EASE | `ml-20m-strong` (Liang et al. 2018 strong generalization) | Steck (2019), Table 1, λ≈500 | nRecall@20 0.391, nRecall@50 0.521, NDCG@100 0.420 | ±0.005 abs |
| iALS | `ml-20m-strong` | Rendle et al. (2022), Table 3 (d=2048, λ=0.003, α0=0.1, 16 epochs) | nRecall@20 0.395, nRecall@50 0.532, NDCG@100 0.425 | ±0.01 abs |
| Mult-VAE | `ml-20m-strong` | Liang et al. (2018), Table 2 | nRecall@20 0.395, nRecall@50 0.537, NDCG@100 0.426 | ±0.01 abs |
| LightGCN | `gowalla-lightgcn` (LightGCN release split) | He et al. (2020), Table 3, 3 layers | Recall@20 0.1823, NDCG@20 0.1555 | ±3% rel |
| BPR-MF | `gowalla-lightgcn` | Wang et al. (2019, NGCF), Table 2 | Recall@20 0.1291 | floor ≥ 0.125 |
| DeepFM | `criteo-x4` (BARS Criteo_x4_001) | BARS README, seed 2019 | AUC 0.8143 | ±0.002 AUC |
| SASRec (full CE) | `ml-1m-loo` (identical to the BERT4Rec ML-1M: 6,040 users, 3,416 items, 999,611 events) | Klenitskiy & Vasilev (2023), Table 2, SASRec+ | NDCG@10 0.1821, HR@10 0.3152 | ±5% rel |

Targets and the exact run configurations are pinned in `tests/reference.toml`; the full provenance (tables, protocol details, URLs) is in the verification report summarized there. Known protocol deviations: the LightGCN and HSTU releases have no validation split (their numbers were selected on test); our benchmarks carve a validation split from train, which can cost a little accuracy.

- *Why:* Owner goal 1. Credibility needs evidence that the implementations aren't weaker than the published ones, which was the issue raised in Rendle et al. (2022) and Dacrema et al. (2019).
- *Verify:* `uv run pytest -m reference` (slow, opt-in) runs each row with its pinned config and asserts the tolerance. Rows whose benchmark doesn't exist yet are marked `xfail(reason="benchmark missing")`, which makes the gap visible rather than hiding it.

**D3 (Must): Closed-form models are exact.** EASE matches a direct numpy closed-form solve `B = I − P·diag(1/diag(P))` with `P = (XᵀX + λI)⁻¹` to 1e-5 on a fixture. Popularity scores equal the train counts exactly.
- *Why:* Deterministic models have an exact answer, so any deviation is a bug.
- *Verify:* Unit tests in `tests/test_algorithms.py`.

**D4 (Must): Neural models can learn.** Each torch algorithm's training loss decreases by at least 30% from epoch 1 to epoch N on a fixture. Each model can overfit a tiny dataset (≤ 50 users): train HR@10 is at least 0.9 for ranking models, and train AUC is at least 0.95 for CTR models.
- *Why:* This catches gradient, masking, or padding bugs that sanity ordering on real data can hide.
- *Verify:* Parametrized test over the torch algorithms, which takes 30 s or less in total on CPU.

**D5 (Should): Sequential models are causal.** In SASRec and every other sequence model, the output at position t doesn't depend on items after t, and padding (`-1`/`0`) doesn't affect the scores.
- *Why:* Future leakage in the attention mask is a classic silent bug.
- *Verify:* A unit test perturbs items after position t and asserts that the logits at t are unchanged (to 1e-6). Changing the padding length leaves the last-position scores unchanged.

---

## E. Reproducibility and results

**E1 (Must): Determinism.** The same algorithm, benchmark, params, and seed on the same machine produces identical metrics: exactly equal on CPU, and within 1e-4 absolute on MPS. Different seeds give different metrics for stochastic algorithms.
- *Why:* A seed that isn't actually used makes "mean ± std over seeds" meaningless.
- *Verify:* A test runs BPR-MF on a fixture twice with seed 0 and asserts equality, then runs it with seed 1 and asserts inequality.

**E2 (Must): The split seed is independent of the run seed.** The benchmark's split and candidate set are identical across run seeds and algorithms.
- *Why:* If the split changes with the run seed, the reported variance mixes data variance with model variance, and algorithms aren't compared on the same test set.
- *Verify:* A test checks that `prepare()` gives byte-identical splits under run seeds 0 and 1, hashing the test frame.

**E3 (Must): Complete provenance in each results row.** Every row records:
- benchmark name and version
- algorithm name, full params, and params_hash
- seed
- all metrics
- `fit_s`, `eval_s`, and peak memory
- timestamp
- `code_sha`, plus a dirty flag
- package versions (`recsys`, torch, numpy, scipy, polars)
- device
- *Why:* Without this, a row can't be reproduced or traced to its code and environment.
- *Verify:* A test runs one experiment on a fixture and checks that the parquet schema contains every field, none null, and `code_sha` matches `git rev-parse HEAD`.

**E4 (Must): An append-only store that respects versions.** Results are appended and never rewritten. `recsys report` aggregates only within a single benchmark version, and either warns about rows from other versions or groups them separately.
- *Why:* This enforces the guarantee from B2 at read time.
- *Verify:* A test writes rows under two versions of one benchmark name and asserts that the report shows two groups or emits a warning, never one merged mean.

**E5 (Should): Statistical reporting.** `recsys report` shows mean ± std and the number of seeds, and can run a paired significance test between two algorithms over seeds or over per-user metrics. A paired t-test or Wilcoxon test, with the p-value shown, is enough.
- *Why:* Sun et al. (2020) and Dacrema et al. (2019) found that many claimed gains are not significant.
- *Verify:* `uv run recsys report --benchmark X --compare ease,sasrec` prints the p-value. A unit test on synthetic metric vectors checks it against `scipy.stats`.

**E6 (Should): Per-user metrics are kept.** You can optionally save per-user metric vectors, or per-row predictions for CTR, next to each run.
- *Why:* This enables significance tests, subgroup analysis such as cold users, and re-scoring with new metrics without retraining.
- *Verify:* With `--save-per-user`, a test asserts that a sidecar file exists with `n_test_users` rows and that its means match the aggregate row.

---

## F. Efficiency

**F1 (Must): Bounded evaluation memory.** Ranking evaluation processes users in batches and never materializes a dense `n_users × n_items` matrix. Peak extra memory during evaluation is O(batch × n_items).
- *Why:* ML-20M has 138k users and 27k items, about 15 GB as dense float32. Generative and retrieval benchmarks at catalog scale need this bound even more.
- *Verify:* Run `/usr/bin/time -l uv run recsys run --algo popularity --benchmark ml-20m-<any>`. The maximum resident set size is at most 4 GB, excluding the in-memory dataset. A unit test with a mock scorer asserts that no call to `score` receives more than `batch_size` users.

**F2 (Must): Wall-clock sanity bounds on an Apple Silicon laptop.** These bounds are regression guards, not targets. Data is cached, the times exclude download, and the device is MPS or CPU.

| Run | Bound |
|---|---|
| `prepare()` for ML-1M | ≤ 10 s |
| `prepare()` for ML-20M | ≤ 90 s |
| Popularity, ML-1M, full pipeline | ≤ 20 s |
| EASE, ML-1M, full pipeline | ≤ 30 s |
| EASE, ML-20M | ≤ 10 min |
| SASRec, ML-1M, 1 epoch | ≤ 60 s |
| DeepFM, ML-1M CTR, 1 epoch | ≤ 60 s |

- *Why:* Owner goal 5. These bounds catch accidental Python loops over interactions and repeated preprocessing.
- *Verify:* A `scripts/bench_speed.py` or `pytest -m perf` reads `fit_s` and `eval_s` from the results rows and asserts the bounds.

**F3 (Must): No Python-level per-interaction loops in hot paths.** Preprocessing, splitting, history construction, batching, and metrics are vectorized with polars, numpy, scipy, or torch.
- *Why:* This is the most common source of 10–100× slowdowns.
- *Verify:*
  - Code review of `protocol.py`, `evaluate.py`, `metrics.py`, and `History`: any `for` loop over rows or users must be over batches, not elements.
  - A perf test shows evaluation time grows close to linearly. Going from 10% to 100% of ML-20M users takes at most 12× the time.

**F4 (Should): Prepared data is reused.** Several seeds or algorithms on the same benchmark don't repeat preprocessing within a process. Should: they also don't repeat it across processes, because a prepared-split cache is keyed by `Benchmark.version`.
- *Why:* Multi-seed and multi-algorithm sweeps dominate runtime otherwise.
- *Verify:* `run_all(..., seeds=[0,1,2])` calls `prepare()` once per benchmark, checked with a call counter in the test. A second process on a warm cache shows `prepare_s` at most 20% of the cold time.

**F5 (Should): Training tensors live on the device.** The built-in training loop keeps training tensors on the device and builds batches without per-batch host-to-device copies or DataLoader workers by default. This also avoids the macOS shared-memory problem.
- *Why:* Host↔device transfer and DataLoader overhead dominate small recommendation models on MPS.
- *Verify:* A profiler test (`torch.profiler`) over one SASRec epoch on a fixture finds no `aten::copy_` from CPU to the device inside the batch loop after `build()`.

**F6 (Deferred): Throughput parity with reference libraries.** On the same hardware, training throughput is at least RecBole's for BPR, LightGCN, and SASRec, and at least FuxiCTR's for DeepFM. EASE on ML-20M fits in 2 min or less. Peak training memory is no more than 1.2× the reference library's.
- *Why:* Owner goal 5 says detailed optimization comes later, so this records the bar now.
- *Verify:* A head-to-head script logs samples per second, to be enforced once optimization work starts.

**F7 (Deferred): Retrieval and generative scale.** On catalogs of 1M items or more, `recommend()` uses top-k without full sorting (`torch.topk` or ANN). Generative beam search is batched across users.
- *Why:* These are needed before large-catalog retrieval and generative benchmarks are added.
- *Verify:* To be defined with the first catalog of 1M items or more.

---

## G. Simplicity and code health

**G1 (Must): No Lightning or heavy framework dependencies.** The runtime dependency closure contains no `lightning`, `pytorch-lightning`, `hydra`, `omegaconf`, `pandas`, or `tensorflow`. There are at most 6 direct runtime dependencies (currently numpy, scipy, polars, torch, huggingface-hub, plus at most one more, such as pyarrow).
- *Why:* This is owner goal 7.
- *Verify:*
  - `grep -Ei '^name = "(lightning|pytorch-lightning|hydra-core|omegaconf|tensorflow|pandas)"' uv.lock` prints nothing. If pandas comes in only through a dev dependency, check with `uv tree --no-dev` instead.
  - `[project].dependencies` has 6 entries or fewer.
  - `grep -rn "import lightning\|pytorch_lightning" src/` prints nothing.

**G2 (Must): Code size budgets.**
- The core is everything except `algorithms/<model>.py`, `data/<dataset>.py`, and the catalog. It must total 1,500 LOC or fewer.
- No single file exceeds 300 LOC.
- Each algorithm file is 150 LOC or fewer.
- Each dataset file is 120 LOC or fewer.
- *Why:* Owner goal 7, "do not overcomplicate." The dev guide promises the core can be read in one sitting.
- *Verify:* `uvx pygount` or `find src -name '*.py' | xargs wc -l` (non-blank, non-comment lines), compared against the budgets by a short script or test.

**G3 (Must): No backward-compatibility code or dead code.**
- No shims, aliases, deprecation paths, or old modules: the old `conf/`, YAML experiments, `engine`/`CTRTask`, and `recsysk` must be gone.
- No unreachable code.
- *Why:* This is owner goal 7.
- *Verify:*
  - `grep -rniE 'deprecat|backward|legacy|compat|shim' src/` prints nothing.
  - `uvx vulture src --min-confidence 80` reports nothing.
  - `uvx ruff check src tests` passes.
  - `git ls-files conf src/recsys/conf` prints nothing.

**G4 (Must): A fast offline test suite that actually exists.** `uv run pytest` runs offline, passes, and finishes in 60 s or less on an M-series laptop. It covers the Must criteria marked with a unit or fixture test above.
- *Why:* Contributors only run the suite if it's fast, and correctness claims need evidence.
- *Verify:* `time uv run pytest -q` passes with an elapsed time of 60 s or less while sockets are disabled. `uv run pytest --co -q | tail -1` shows at least 40 tests.

**G5 (Should): One way to do each thing.** There is a single run entry point (`recsys.run` / `run_all`, wrapped by the CLI) and a single training loop. There's no second config system: no YAML layer on top of constructor arguments.
- *Why:* Duplicate paths drift apart, which is how the old classical-vs-Lightning branch split happened.
- *Verify:* `grep -rn "yaml" src/` prints nothing. Code review confirms that `cli.py` only parses arguments and calls the public API (`cli.py` is 120 LOC or fewer).

**G6 (Should): Public API surface.** `recsys.__all__` exports at most about 15 names:
- `run`, `run_all`, `evaluate`, `summary`
- `Recommender`, `TorchRecommender`
- `Dataset`, `Benchmark`
- the protocol primitives
- the registries
- *Why:* A small surface is easier to learn and to keep stable after 1.0.
- *Verify:* `uv run python -c "import recsys; print(len(recsys.__all__))"` prints 15 or less (soft cap 20).

---

## H. Documentation

**H1 (Must): Docs match the registries.** The README and `docs/` list exactly the registered datasets, benchmarks (with version and protocol summary), and algorithms. The "Implemented" status in `methods.md` matches C8.
- *Why:* Stale docs are the first thing users trip over.
- *Verify:* `tests/test_docs.py` parses the doc tables and compares them to `recsys list {datasets,benchmarks,algorithms}`. They must be equal as sets.

**H2 (Must): The README quickstart runs verbatim.** Every `bash`/`python` code block in the README's quickstart and "Adding an algorithm" sections runs without error. Network blocks run as a network test. The algorithm example is the same file tested by C2/C3.
- *Why:* The "several lines of code" promise only counts if the shown code runs.
- *Verify:* `tests/test_docs.py` extracts the fenced blocks and runs them against a fixture benchmark. `pytest --doctest-glob` or a small extractor is fine.

**H3 (Must): Each benchmark has a documented protocol.** For each benchmark, the docs state:
- dataset and variant
- preprocessing
- split type and its known leakage characteristics (B6/B7)
- candidate set (full catalog)
- metrics, and which one selects the model
- pinned statistics (B3)
- the matching reference protocol from the literature, if any
- *Why:* Users have to know exactly what a number means before comparing it to a paper.
- *Verify:* For each benchmark, `recsys list benchmarks --verbose` (or a generated docs table) shows all seven fields, and a test asserts that none are empty.

**H4 (Should): Extension guides.** `docs/` contains "Add a dataset," "Add a benchmark," and "Add an algorithm" guides, one each for classical, torch, and generative algorithms. Each guide is 1 page or less and ends with the one-line command that runs the new component on every compatible benchmark.
- *Why:* Owner goals 2 and 3.
- *Verify:* Manual review. The code in each guide is covered by H2.

---

## Most likely to fail in a typical implementation

1. **C1 (generative and retrieval coverage).** Most harnesses have only `score() → (batch, n_items)` and `predict()`. Generative models that decode item ids, and retrieval models that want `recommend(k)`, are pushed through dense scoring or not supported at all. "Generative uses the ranking task" is fine as a concept, but the interface still needs a top-k path.
2. **D2 (reference numbers).** Without the exact reference protocols (ML-20M strong generalization, the Gowalla LightGCN split, BARS Criteo), there's nothing to compare against. When the protocols do exist, default hyperparameters usually fall short of tuned paper numbers by more than the tolerance.
3. **A3 (pinned checksums) and B3 (pinned statistics).** These are usually skipped, so upstream changes (HF dataset revisions, re-uploaded Zenodo archives) silently change benchmark contents under an unchanged version hash.
4. **B10 (ties) and B11 (masking).** Stable `argsort`/`topk` on constant or tied scores quietly favors low item ids. Off-by-one masking that includes the target is common and not visible in aggregate numbers.
5. **B6 and B7 (leakage).** Leave-last-out is the default almost everywhere. Popularity or CTR vocabularies are often computed before the split, and no temporal benchmark exists to show the effect.
6. **F1 (eval memory) at ML-20M scale.** Code that works on ML-1M often builds a dense score matrix or a dense history mask that runs out of memory on ML-20M.
7. **C8, H1, and H2 (catalog and docs drift).** Without automated doc and registry tests, `methods.md` and the README fall behind the code within a few commits.
8. **G4 (test suite).** Suites tend to be either missing or network-dependent and slow. The criteria in this document assume most checks are automated, which needs a real fixture-based suite.
