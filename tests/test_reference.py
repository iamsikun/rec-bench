"""Reproduce published numbers under matching protocols (criteria D1, D2); run with ``-m reference``.

Targets live in ``reference.toml`` next to this file, each copied from its cited table.
"""

import tomllib
from pathlib import Path

import pytest

import recsys
from recsys import BENCHMARKS

pytestmark = pytest.mark.reference
REFERENCES = tomllib.loads((Path(__file__).parent / "reference.toml").read_text())["reference"]


@pytest.mark.parametrize("ref", REFERENCES, ids=lambda r: f"{r['algo']}-{r['benchmark']}")
def test_d2_reference_numbers(ref, tmp_path):
    if ref["benchmark"] not in BENCHMARKS:
        pytest.xfail(f"benchmark {ref['benchmark']} missing")
    records = [recsys.run(ref["algo"], ref["benchmark"], seed=seed, params=ref.get("params", {}), results_dir="results")
               for seed in ref.get("seeds", [0])]
    record = {m: sum(r[m] for r in records) / len(records) for m in ref["targets"]}  # papers report seed means
    for metric, target in ref["targets"].items():
        if "floor" in ref:
            assert record[metric] >= ref["floor"][metric], (metric, record[metric])
        elif "abs" in ref:
            assert abs(record[metric] - target) <= ref["abs"], (metric, record[metric], target)
        else:
            assert abs(record[metric] - target) <= ref["rel"] * target, (metric, record[metric], target)


RANKING = [n for n, b in BENCHMARKS.items() if b.task == "ranking" and not n.startswith("synthetic")]
LEARNED = ["ease", "bpr", "lightgcn", "sasrec"]


@pytest.mark.parametrize("bench", RANKING)
def test_d1_sanity_ordering(bench):
    from recsys.algorithms.base import incompatibility

    algos = ["random", "popularity", *[a for a in LEARNED if incompatibility(recsys.ALGORITHMS[a], BENCHMARKS[bench])
                                        is None]]
    out = recsys.run_all(algos, bench, seeds=[0, 1, 2], results_dir="results")
    select = BENCHMARKS[bench].metrics[0]
    stats = out.group_by("algo").agg(mean=recsys.store.pl.col(select).mean(), std=recsys.store.pl.col(select).std())
    m = dict(zip(stats["algo"], stats["mean"]))
    s = dict(zip(stats["algo"], stats["std"]))
    gap = lambda a, b: m[b] - m[a] > 2 * ((s[a] or 0) + (s[b] or 0)) / 2  # noqa: E731
    assert gap("random", "popularity")
    for algo in algos[2:]:
        assert gap("popularity", algo), (algo, m)
