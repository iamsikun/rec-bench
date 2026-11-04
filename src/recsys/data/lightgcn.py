"""LightGCN (He et al., SIGIR 2020) published train/test splits: https://github.com/kuandeng/LightGCN"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from recsys.data.base import Dataset, Interactions
from recsys.data.download import download

# Files at a pinned commit of the authors' official repository (Data/<name>/{train,test}.txt).
_COMMIT = "b067ec05f16a24e0e1efe63cd82385f0c469af20"
_URL = "https://raw.githubusercontent.com/kuandeng/LightGCN/{commit}/Data/{name}/{split}.txt"
# name -> (train.txt sha256, test.txt sha256)
_PINS = {
    "gowalla": (
        "7244696fdeb23911319cf7b689f2e15a3c4c92549c873a5677b44e3f13cc3e62",
        "099a2a73924e4b754dc6efd730d764f492321b53c38eb93298850c83acf57be3",
    ),
    "yelp2018": (
        "1e9568725a9e833d0ac3bae01c94cd95367f73b3bb6104fd26b5f65ea5054093",
        "9cd9b37926cd1447a06625ffa7914527573fb097c10cb0bbdb1d1b8ded3742fe",
    ),
    "amazon-book": (
        "529066026085919381acd4ce19f69e9eafa4af3fef1a10119ea5551cc2ef8530",
        "6250754994dd953b91b48dccdd77a73f318e0634b4757c2abfcc9d33a54ab0f5",
    ),
}
_SPLITS = ("train", "test")


def read_adjacency(path: Path) -> pl.DataFrame:
    """Parse lines of ``user item item ...`` into (user, item) rows; lines without items are skipped."""
    tokens = (
        pl.read_csv(path, has_header=False, new_columns=["line"], separator="\x1f", quote_char=None)
        .select(pl.col("line").str.strip_chars().str.split(" ").alias("ids"))
        .filter(pl.col("ids").list.len() > 1)
    )
    return (
        tokens.select(
            pl.col("ids").list.first().cast(pl.Int64).alias("user"),
            pl.col("ids").list.slice(1).alias("item"),
        )
        .explode("item")
        .filter(pl.col("item") != "")
        .with_columns(pl.col("item").cast(pl.Int64))
    )


@dataclass(frozen=True)
class LightGCNData(Dataset):
    """LightGCNData (He et al., SIGIR 2020): the gowalla / yelp2018 / amazon-book splits released with LightGCN.

    These are the exact 10-core, 80/20 random train/test files used by NGCF
    and LightGCN, with ids already remapped to 0..n-1. ``events``: user, item,
    split (``train`` / ``test``; no validation rows are published, so
    :class:`~recsys.protocol.Predefined` carves one from train), timestamp.
    The release carries no time information: ``timestamp`` is always 0, so
    temporal splits and sequential models are meaningless on these data.
    """

    name: str = "gowalla"
    dirname = "lightgcn"
    has_time = False

    def __post_init__(self) -> None:
        if self.name not in _PINS:
            raise ValueError(f"Unknown LightGCN dataset {self.name!r}; choose from {list(_PINS)}")

    def pins(self) -> tuple[str, ...]:
        return _PINS[self.name]

    def files_present(self, root: Path) -> bool:
        return all((root / self.name / f"{s}.txt").exists() for s in _SPLITS)

    def download(self, root: Path) -> None:
        for split, digest in zip(_SPLITS, _PINS[self.name]):
            url = _URL.format(commit=_COMMIT, name=self.name, split=split)
            download(url, root / self.name / f"{split}.txt", digest)

    def read(self, root: Path) -> Interactions:
        events = pl.concat([
            read_adjacency(root / self.name / f"{s}.txt").with_columns(
                pl.lit(s).alias("split"), pl.lit(0, dtype=pl.Int64).alias("timestamp"),
            )
            for s in _SPLITS
        ])
        return Interactions(events=events)
