"""BARS open CTR benchmark splits (Zhu et al., SIGIR 2022 / CIKM 2021): https://openbenchmark.github.io/BARS/"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from recsys.data.base import Dataset, Interactions
from recsys.data.download import extract, sha256

_REPO = "reczoo/{}"
# name -> (Hugging Face dataset revision, sha256 of <name>.zip at that revision)
_PINS = {
    "Criteo_x4": (
        "2582e812efc2f8405221900548591d0ddd89f727",
        "a094d9264f9c176a35f106e140a3e4ac83cba8da466fe72317ace53f9b8dd6e4",
    ),
    "Avazu_x4": (
        "a436154d73738b1ea35acf3d1cc56f72997266d9",
        "4ba8bf46eada70abbe068e4e5bfb3b3d879e6fbd2f2fcf51760cd1388b789b65",
    ),
}
_SPLITS = {"train": "train", "valid": "val", "test": "test"}
_SPLIT_DTYPE = pl.Enum(["train", "val", "test"])
_CRITEO_INT = [f"I{i}" for i in range(1, 14)]
_AVAZU_INT = ["C1", "banner_pos", "device_type", "device_conn_type", *[f"C{i}" for i in range(14, 22)]]
_AVAZU_HEX = [
    "site_id", "site_domain", "site_category", "app_id", "app_domain", "app_category",
    "device_id", "device_ip", "device_model",
]
# name -> (label column in the file, integer columns, 8-hex-digit hashed columns)
_COLUMNS = {
    "Criteo_x4": ("Label", _CRITEO_INT, [f"C{i}" for i in range(1, 27)]),
    "Avazu_x4": ("click", _AVAZU_INT, _AVAZU_HEX),
}


def scan_split(path: Path, name: str, split: str) -> pl.LazyFrame:
    """Lazily read one BARS csv with compact dtypes.

    The anonymised categorical values are 8 hex digits (a 32-bit hash), so they
    are stored losslessly as UInt32 (``int(value, 16)``); empty values are null.
    """
    label, ints, hexes = _COLUMNS[name]
    dtypes = {label: pl.UInt8} | {c: pl.Int32 for c in ints} | {c: pl.String for c in hexes}
    if name == "Avazu_x4":
        dtypes |= {"id": pl.UInt64, "hour": pl.Int32}
    lf = pl.scan_csv(path, schema_overrides=dtypes)
    exprs = [
        pl.col(label).alias("label"), *ints,
        *(pl.col(c).str.to_integer(base=16, strict=True).cast(pl.UInt32) for c in hexes),
    ]
    if name == "Avazu_x4":
        hour = (pl.col("hour").cast(pl.String) + "00").str.to_datetime("%y%m%d%H%M", time_zone="UTC")
        exprs += ["id", "hour", hour.dt.epoch("s").alias("timestamp")]
    return lf.select(*exprs, pl.lit(split, dtype=_SPLIT_DTYPE).alias("split"))


@dataclass(frozen=True)
class BARS(Dataset):
    """BARS (Zhu et al., Huawei Noah's Ark / reczoo): fixed Criteo_x4 and Avazu_x4 CTR splits.

    The 8:1:1 random train/valid/test splits used by AutoInt, FuxiCTR and the
    BARS leaderboard, downloaded from the reczoo Hugging Face organisation at a
    pinned revision. ``events`` keeps every raw column: ``label`` (0/1), the
    integer fields as Int32, the hashed categorical fields as UInt32, and
    ``split`` (train/val/test). Criteo: I1..I13, C1..C26 (no time). Avazu: id,
    hour (YYMMDDHH), timestamp (hour parsed as UTC seconds), C1, banner_pos,
    site_*, app_*, device_*, C14..C21. No user/item columns or side tables:
    CTR benchmarks pick their own fields. ``users`` / ``items`` are None.
    """

    name: str = "Avazu_x4"
    dirname = "bars"

    def __post_init__(self) -> None:
        if self.name not in _PINS:
            raise ValueError(f"Unknown BARS dataset {self.name!r}; choose from {list(_PINS)}")

    def pins(self) -> tuple[str, ...]:
        return _PINS[self.name]

    def files_present(self, root: Path) -> bool:
        return all((root / self.name / f"{s}.csv").exists() for s in _SPLITS)

    def download(self, root: Path) -> None:
        from huggingface_hub import hf_hub_download

        revision, digest = _PINS[self.name]
        archive = Path(hf_hub_download(
            _REPO.format(self.name), f"{self.name}.zip", repo_type="dataset", revision=revision, local_dir=root,
        ))
        if sha256(archive) != digest:
            raise RuntimeError(f"Checksum mismatch for {archive}; expected {digest}")
        (root / self.name).mkdir(exist_ok=True)
        extract(archive, root / self.name)
        archive.unlink()  # The extracted csv files are what read() needs; the zip only doubles disk use.
        shutil.rmtree(root / ".cache", ignore_errors=True)

    def scan(self, root: Path) -> pl.LazyFrame:
        """All three splits as one lazy frame, for callers that want to select or filter before collecting."""
        return pl.concat(self._scans(root))

    def _scans(self, root: Path) -> list[pl.LazyFrame]:
        return [scan_split(root / self.name / f"{f}.csv", self.name, s) for f, s in _SPLITS.items()]

    def read(self, root: Path) -> Interactions:
        # Collect split by split and concatenate without rechunking, so peak memory stays near the result size
        # (Criteo_x4: ~7 GB in memory).
        parts = [lf.collect(engine="streaming") for lf in self._scans(root)]
        return Interactions(events=pl.concat(parts, rechunk=False))
