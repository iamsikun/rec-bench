"""Command line: ``recsys list | run | report``."""

from __future__ import annotations

import argparse
import json
import logging

from recsys import store
from recsys.algorithms import ALGORITHMS
from recsys.benchmarks import BENCHMARKS
from recsys.data import DATASETS
from recsys.harness import run_all


def _param(text: str) -> tuple[str, object]:
    key, _, value = text.partition("=")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def _first_line(obj) -> str:
    return (obj.__doc__ or "").strip().splitlines()[0]


def protocol(b) -> dict[str, str]:
    """The documented protocol of a benchmark (every field must be non-empty)."""
    return {
        "dataset": repr(b.dataset),
        "preprocessing": ", ".join(map(repr, b.preprocess)) or "none",
        "split": repr(b.split),
        "leakage": b.leakage,
        "candidates": "full catalog, known items masked" if b.task == "ranking" else "all held-out rows",
        "metrics": f"{', '.join(b.metrics)} (model selection: {b.metrics[0]})",
        "stats": str(b.stats),
        "reference": b.reference,
    }


def _list(what: str, verbose: bool) -> None:
    if what == "algorithms":
        for name, cls in ALGORITHMS.items():
            print(f"{name:<14} {','.join(sorted(cls.tasks)):<12} {_first_line(cls)}")
    elif what == "benchmarks":
        for name, b in BENCHMARKS.items():
            print(f"{name:<24} {b.task:<8} {b.version}  {b.description}")
            if verbose:
                for key, value in protocol(b).items():
                    print(f"    {key:<14} {value}")
    else:
        for name, cls in DATASETS.items():
            print(f"{name:<14} {_first_line(cls)}")


def _report(results_dir: str, benchmark: str | None) -> None:
    tables = store.summary(results_dir, benchmark)
    if not tables:
        print(f"No results under {results_dir}")
    for name, table in tables.items():
        metrics = [c for c in table.columns if f"{c} std" in table.columns]
        print(f"\n{name}")
        print(f"  {'algo':<14} {'version':<10} {'params':<10} {'seeds':>5}  "
              + "  ".join(f"{m:>15}" for m in metrics) + "    fit_s")
        for row in table.iter_rows(named=True):
            cells = "  ".join(f"{row[m]:.4f} ± {row[f'{m} std'] or 0:.4f}" for m in metrics)
            print(f"  {row['algo']:<14} {row['version']:<10} {row['params_hash']:<10} {row['seeds']:>5}  "
                  f"{cells}  {row['fit_s']:7.1f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="recsys", description="Standardized recommender-system benchmark.")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list datasets, benchmarks or algorithms")
    p_list.add_argument("what", choices=["datasets", "benchmarks", "algorithms"])
    p_list.add_argument("-v", "--verbose", action="store_true", help="show each benchmark's full protocol")

    p_run = sub.add_parser("run", help="run algorithms on every compatible benchmark")
    p_run.add_argument("--algo", required=True, help="comma-separated algorithm names")
    p_run.add_argument("--benchmark", default="all", help="comma-separated benchmark names, or 'all'")
    p_run.add_argument("--seeds", default="0", help="comma-separated seeds")
    p_run.add_argument("-p", "--param", action="append", default=[], type=_param, metavar="KEY=VALUE",
                       help="algorithm constructor argument (JSON value), repeatable")
    p_run.add_argument("--results", default="results")
    p_run.add_argument("--save-per-user", action="store_true", help="also store per-user metrics / predictions")

    p_report = sub.add_parser("report", help="summarize stored results (mean and std over seeds)")
    p_report.add_argument("--benchmark")
    p_report.add_argument("--results", default="results")
    p_report.add_argument("--compare", metavar="A,B", help="paired significance test between two algorithms")

    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    if args.command == "list":
        _list(args.what, args.verbose)
    elif args.command == "run":
        benchmarks = "all" if args.benchmark == "all" else args.benchmark.split(",")
        seeds = [int(s) for s in args.seeds.split(",")]
        run_all(args.algo.split(","), benchmarks, seeds, dict(args.param), args.results, args.save_per_user)
        _report(args.results, None if benchmarks == "all" or len(benchmarks) > 1 else benchmarks[0])
    elif args.compare:
        if not args.benchmark:
            parser.error("--compare needs --benchmark")
        a, b = args.compare.split(",")
        result = store.compare(args.results, args.benchmark, a, b)
        print(json.dumps(result, indent=2))
    else:
        _report(args.results, args.benchmark)


if __name__ == "__main__":
    main()
