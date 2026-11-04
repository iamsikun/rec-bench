"""Simplicity, code health and documentation (criteria G1-G3, G5, G6, H1-H4, C8)."""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import recsys
from recsys import ALGORITHMS, BENCHMARKS, DATASETS
from recsys.cli import protocol

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "recsys"
FORBIDDEN = {"lightning", "pytorch-lightning", "hydra-core", "omegaconf", "tensorflow", "pandas"}


def _code_lines(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#"))


def test_g1_small_dependency_set_without_frameworks():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert len(project["dependencies"]) <= 6
    runtime = subprocess.run(["uv", "export", "--no-dev", "--no-hashes", "--frozen"], capture_output=True, text=True,
                             cwd=ROOT, check=True).stdout
    names = {line.split("==")[0].strip().lower() for line in runtime.splitlines() if "==" in line}
    assert names & FORBIDDEN == set()
    assert not re.search(r"import (lightning|pytorch_lightning)", "".join(p.read_text() for p in SRC.rglob("*.py")))


def test_g2_code_size_budgets():
    files = list(SRC.rglob("*.py"))
    algorithms = {p for p in (SRC / "algorithms").glob("*.py") if p.stem not in {"__init__", "base", "trainer"}}
    datasets = {p for p in (SRC / "data").glob("*.py") if p.stem not in {"__init__", "base", "download"}}
    core = [p for p in files if p not in algorithms | datasets and p.name != "catalog.py"]
    assert sum(map(_code_lines, core)) <= 1500
    assert {p.name: _code_lines(p) for p in files if _code_lines(p) > 300} == {}
    assert {p.name: _code_lines(p) for p in algorithms if _code_lines(p) > 150} == {}
    assert {p.name: _code_lines(p) for p in datasets if _code_lines(p) > 120} == {}


def test_g3_no_compat_code_dead_code_or_lint():
    text = "\n".join(p.read_text() for p in SRC.rglob("*.py"))
    assert not re.search(r"deprecat|(?<!\.)backward|legacy|compat\b|shim", text, re.IGNORECASE)
    assert not (ROOT / "conf").exists() and not (SRC / "conf").exists()
    ruff = subprocess.run([sys.executable, "-m", "ruff", "check", "src", "tests", "examples"], cwd=ROOT,
                          capture_output=True, text=True)
    assert ruff.returncode == 0, ruff.stdout
    vulture = subprocess.run([sys.executable, "-m", "vulture", "src", "--min-confidence", "80"], cwd=ROOT,
                             capture_output=True, text=True)
    assert vulture.stdout.strip() == "", vulture.stdout


def test_g5_single_config_path_and_thin_cli():
    assert "yaml" not in "".join(p.read_text() for p in SRC.rglob("*.py")).lower()
    assert _code_lines(SRC / "cli.py") <= 120


def test_g6_small_public_api():
    assert len(recsys.__all__) <= 20


# -------------------------------------------------------------------- documentation


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    nxt = text.find("\n## ", start + len(heading))
    return text[start: nxt if nxt != -1 else len(text)]


def _first_column(section: str) -> set[str]:
    return set(re.findall(r"^\| `([^`]+)` \|", section, re.MULTILINE))


def test_h1_readme_lists_exactly_the_registered_algorithms_and_benchmarks():
    readme = (ROOT / "README.md").read_text()
    assert _first_column(_section(readme, "## Algorithms")) == set(ALGORITHMS)
    assert _first_column(_section(readme, "## Benchmarks")) == set(BENCHMARKS)


def test_h1_datasets_doc_lists_exactly_the_registered_datasets():
    doc = (ROOT / "docs" / "datasets.md").read_text()
    listed = set(re.findall(r"^\| `(\w+)\(", doc, re.MULTILINE))
    assert listed == {cls.__name__ for cls in DATASETS.values()}


# registry key -> row name in docs/methods.md
CATALOG_ROWS = {
    "random": "Random", "popularity": "Popularity (TopPop)", "itemknn": "ItemKNN", "userknn": "UserKNN",
    "slim": "SLIM", "rp3beta": "RP3beta", "puresvd": "PureSVD", "ials": "WMF / iALS", "sknn": "SKNN (session-kNN)",
    "ease": "EASE", "bpr": "BPR-MF", "lightgcn": "LightGCN", "sasrec": "SASRec", "gsasrec": "gSASRec",
    "bert4rec": "BERT4Rec", "gru4rec": "GRU4Rec", "srgnn": "SR-GNN", "neumf": "NeuMF / NCF",
    "multvae": "Mult-VAE / Mult-DAE", "multdae": "Mult-VAE / Mult-DAE", "lr": "LR (FTRL)", "fm": "FM",
    "widedeep": "Wide & Deep", "deepfm": "DeepFM", "dcn": "DCN", "dcnv2": "DCN-V2", "xdeepfm": "xDeepFM",
    "autoint": "AutoInt / AutoInt+", "finalmlp": "FinalMLP", "din": "DIN", "dien": "DIEN", "dssm": "DSSM",
    "youtubednn": "YouTube DNN", "twotower_logq": "Sampling-bias-corrected two-tower", "tiger": "TIGER",
    "hstu": "HSTU",
    "gfcf": "GF-CF",
    "simgcl": "SimGCL",
    "xsimgcl": "XSimGCL",
    "recvae": "RecVAE",
    "gdcn": "GDCN",
    "finalnet": "FinalNet",
    "masknet": "MaskNet",
    "bst": "BST",
    "sim": "SIM",
    "eta": "ETA",
    "letter": "LETTER",
    "liger": "LIGER",
    "tisasrec": "TiSASRec",
    "cl4srec": "CL4SRec",
    "duorec": "DuoRec",
    "fmlprec": "FMLP-Rec",
    "bsarec": "BSARec",
    "mamba4rec": "Mamba4Rec",
}
ALSO_IMPLEMENTED = {"SASRec-CE"}  # our SASRec is trained with full-softmax CE


def _implemented_rows() -> set[str]:
    doc = (ROOT / "docs" / "methods.md").read_text()
    return {line.split("|")[1].strip() for line in doc.splitlines()
            if line.startswith("| ") and "**Implemented**" in line}


def test_c8_methods_catalog_matches_registry():
    assert set(ALGORITHMS) <= set(CATALOG_ROWS)
    assert _implemented_rows() == {CATALOG_ROWS[k] for k in ALGORITHMS} | ALSO_IMPLEMENTED


def test_c8_every_tier1_method_is_implemented():
    doc = (ROOT / "docs" / "methods.md").read_text()
    missing = [line.split("|")[1].strip() for line in doc.splitlines()
               if line.startswith("|") and re.search(r"\|\s*1\s*\|", line) and "**Implemented**" not in line]
    assert missing == []


def _blocks(text: str, lang: str) -> list[str]:
    return re.findall(rf"```{lang}\n(.*?)```", text, re.DOTALL)


def test_h2_readme_algorithm_examples_are_the_tested_files():
    readme = (ROOT / "README.md").read_text()
    blocks = _blocks(readme, "python")
    for example in ("minimal_algo.py", "minimal_torch_algo.py"):
        assert (ROOT / "examples" / example).read_text() in blocks


@pytest.mark.network
def test_h2_readme_quickstart_runs(tmp_path):
    readme = (ROOT / "README.md").read_text()
    examples = {(ROOT / "examples" / f).read_text() for f in ("minimal_algo.py", "minimal_torch_algo.py")}
    for block in _blocks(readme, "python"):
        if block not in examples:
            exec(compile(block, "README", "exec"), {"__name__": "readme"})


def test_h3_every_benchmark_documents_its_protocol():
    for b in BENCHMARKS.values():
        fields = protocol(b)
        assert all(v and v != "None" for v in fields.values()), (b.name, fields)


def test_h4_extension_guides_exist():
    dev = (ROOT / "docs" / "dev.md").read_text()
    for heading in ("### Add a dataset", "### Add a benchmark", "### Add a classical algorithm",
                    "### Add a neural algorithm", "### Add a generative or retrieval algorithm"):
        section = dev[dev.index(heading):]
        section = section[: section.find("\n### ", 4) if section.find("\n### ", 4) != -1 else len(section)]
        assert len(section.splitlines()) <= 60 and "run_all" in section, heading
