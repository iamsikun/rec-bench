"""Datasets and acquisition (criteria A1, A3-A6, A8)."""

import hashlib
import http.client
import http.server
import subprocess
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

import recsys
from recsys.data import DATASETS, Dataset, Interactions, data_root
from recsys.data.download import download, fetch
from recsys.protocol import LeaveLastOut

ROOT = Path(__file__).resolve().parents[1]


class _Server:
    """Serves ``body`` with a configurable Content-Length on localhost."""

    def __init__(self, body: bytes, length: int | None = None):
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                outer.hits += 1
                self.send_response(200)
                self.send_header("Content-Length", str(outer.length))
                self.end_headers()
                self.wfile.write(outer.body)

            def log_message(self, *args):
                pass

        self.body, self.length, self.hits = body, len(body) if length is None else length, 0
        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/file.bin"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


def test_a1_no_data_files_tracked():
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=ROOT).stdout.split()
    data_ext = (".csv", ".dat", ".tsv", ".parquet", ".zip", ".gz", ".jsonl", ".npz", ".pkl")
    assert [f for f in files if f.endswith(data_ext) and not f.startswith("tests/")] == []
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert "datasets/" in ignored and "results/" in ignored


def test_a3_checksum_mismatch_raises_and_leaves_nothing(tmp_path):
    server = _Server(b"real bytes")
    try:
        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            download(server.url, tmp_path / "f.bin", hashlib.sha256(b"other bytes").hexdigest())
        assert _leftovers(tmp_path) == []
        download(server.url, tmp_path / "f.bin", hashlib.sha256(b"real bytes").hexdigest())
        assert (tmp_path / "f.bin").read_bytes() == b"real bytes"
    finally:
        server.close()


def test_a3_every_dataset_pins_its_downloads():
    for name, cls in DATASETS.items():
        if name == "synthetic":
            continue
        pins = cls().pins()
        assert pins and all(p and p != "PENDING" for p in pins), name


def test_a4_truncated_transfer_is_never_kept(tmp_path):
    server = _Server(b"short", length=100)
    try:
        with pytest.raises((RuntimeError, http.client.HTTPException)):
            download(server.url, tmp_path / "f.bin", hashlib.sha256(b"short").hexdigest())
        assert _leftovers(tmp_path) == []
    finally:
        server.close()


def test_a4_extraction_is_atomic(tmp_path, monkeypatch):
    archive = tmp_path / "src" / "a.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("data/one.csv", "x\n1\n")
    server = _Server(archive.read_bytes())
    root = tmp_path / "root"
    try:
        def crash(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(zipfile.ZipFile, "extractall", crash)
        with pytest.raises(KeyboardInterrupt):
            fetch(server.url.replace("file.bin", "a.zip"), root, hashlib.sha256(archive.read_bytes()).hexdigest())
        assert not (root / "data").exists()
        monkeypatch.undo()
        fetch(server.url.replace("file.bin", "a.zip"), root, hashlib.sha256(archive.read_bytes()).hexdigest())
        assert (root / "data" / "one.csv").exists()
    finally:
        server.close()


def test_a5_cache_root_and_offline_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv("RECSYS_DATA", str(tmp_path / "cache"))
    assert data_root() == tmp_path / "cache"
    csv = b"user,item,timestamp\n1,1,1600000000\n"
    server = _Server(csv)

    @dataclass(frozen=True)
    class Remote(Dataset):
        dirname = "remote"

        def files_present(self, root):
            return (root / "file.bin").exists()

        def download(self, root):
            download(server.url, root / "file.bin", hashlib.sha256(csv).hexdigest())

        def read(self, root):
            return Interactions(pl.read_csv(root / "file.bin"))

    try:
        Remote().load()
    finally:
        server.close()
    assert Remote().load().events.height == 1 and server.hits == 1  # second load: no network
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "datasets").exists()


def _check_canonical(dataset) -> None:
    data = dataset.load()
    events = data.events
    assert {"user", "item"} <= set(events.columns)
    if dataset.has_time:
        assert events["timestamp"].dtype.is_integer()
        assert 1e8 <= events["timestamp"].median() <= 3e9
    for key, table in (("user", data.users), ("item", data.items)):
        if table is not None:
            assert key in table.columns


def test_a6_canonical_format_fixture():
    _check_canonical(recsys.data.Synthetic())


@pytest.mark.network
@pytest.mark.parametrize("name", [n for n in DATASETS if n != "synthetic"])
def test_a6_canonical_format_real(name):
    _check_canonical(DATASETS[name]())


TOY = '''
@dataclass(frozen=True)
class ToyCSV(Dataset):
    """Toy: reads a local CSV fixture."""

    path: str = ""
    dirname = "toy"

    def files_present(self, root):
        return True

    def download(self, root):
        pass

    def read(self, root):
        return Interactions(pl.read_csv(self.path))
'''


def test_a8_new_dataset_is_one_short_subclass(tmp_path):
    assert sum(1 for line in TOY.splitlines() if line.strip()) <= 40
    namespace = {"dataclass": dataclass, "Dataset": Dataset, "Interactions": Interactions, "pl": pl}
    exec(TOY, namespace)
    rows = [f"{u},{i},{1_600_000_000 + t}" for u in range(20) for t, i in enumerate(range(u, u + 6))]
    (tmp_path / "toy.csv").write_text("user,item,timestamp\n" + "\n".join(rows))
    toy = namespace["ToyCSV"](str(tmp_path / "toy.csv"))
    bench = recsys.Benchmark("toy", toy, "ranking", LeaveLastOut(), ("ndcg@10",))
    assert recsys.run("popularity", bench, results_dir=None)["ndcg@10"] >= 0
