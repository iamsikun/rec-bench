"""Amazon product reviews 2014 (McAuley et al., SIGIR 2015 / WWW 2016): https://jmcauley.ucsd.edu/data/amazon/"""

from __future__ import annotations

import ast
import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from recsys.data.base import Dataset, Interactions
from recsys.data.download import download

# The UCSD host no longer answers; SNAP serves the same release files.
_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/{}"
# category -> (reviews_<category>_5.json.gz sha256, meta_<category>.json.gz sha256)
_PINS = {
    "Beauty": (
        "e5925ec99023f1dc9c7d3dff7ef34cfeacd40d1b981b5525e1e4ba9c8abc18fe",
        "c7977dc0e0ead14ac7df8c1c6de74da00e4bebfad87a0f0c5763d4c5bc9b53a0",
    ),
    "Sports_and_Outdoors": (
        "8130a2206c3062db1b4b9d4353efc303b0ebd13f48372559fee26c47a1d7fe51",
        "677108a0fe0f47cce08885a348879a5a89dd69e75fbc5d66410ae436750df086",
    ),
    "Toys_and_Games": (
        "f9f0cc56ad6b5de15a1b0dd1c07fcfc9e7367efc73481f5d94d8065f91149286",
        "beffb81f3632050b28d8b0acb1ee27f0ddc2142e9a6abba293357a5312f5e876",
    ),
}
_ASIN = re.compile(r"""^\{'asin': '([^']+)'""")
_ITEM_SCHEMA = {"item": pl.String, "category": pl.String, "brand": pl.String, "price": pl.Float64}


def read_meta(path: Path, keep: set[str]) -> pl.DataFrame:
    """Parse the Python-literal metadata lines of the items in ``keep``.

    Each row: item, category (most specific level of the first category
    path), brand, price (USD; null when absent).
    """
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            match = _ASIN.match(line)
            if match is None or match.group(1) not in keep:
                continue
            meta = ast.literal_eval(line)
            paths = meta.get("categories") or [[]]
            price = meta.get("price")
            rows.append({
                "item": meta["asin"],
                "category": paths[0][-1] if paths[0] else None,
                "brand": meta.get("brand") or None,
                "price": float(price) if price is not None else None,
            })
    return pl.DataFrame(rows, schema=_ITEM_SCHEMA)


@dataclass(frozen=True)
class Amazon2014(Dataset):
    """Amazon2014 (McAuley et al., UCSD): 5-core product reviews per category, the SASRec / TIGER standard.

    ``category`` is ``Beauty``, ``Sports_and_Outdoors`` or ``Toys_and_Games``
    (files ``reviews_<category>_5.json.gz``; every user and item has at least
    5 reviews). ``events``: user (reviewerID), item (asin), rating (overall,
    1-5), timestamp (unixReviewTime, seconds, day resolution). ``items`` (when
    ``meta=True``): item, category, brand, price, from ``meta_<category>.json.gz``.
    """

    category: str = "Beauty"
    meta: bool = True
    dirname = "amazon-2014"

    def __post_init__(self) -> None:
        if self.category not in _PINS:
            raise ValueError(f"Unknown Amazon 2014 category {self.category!r}; choose from {list(_PINS)}")

    def pins(self) -> tuple[str, ...]:
        reviews, meta = _PINS[self.category]
        return (reviews, meta) if self.meta else (reviews,)

    @property
    def _files(self) -> tuple[str, ...]:
        files = (f"reviews_{self.category}_5.json.gz", f"meta_{self.category}.json.gz")
        return files if self.meta else files[:1]

    def files_present(self, root: Path) -> bool:
        return all((root / f).exists() for f in self._files)

    def download(self, root: Path) -> None:
        for filename, digest in zip(self._files, _PINS[self.category]):
            download(_URL.format(filename), root / filename, digest)

    def read(self, root: Path) -> Interactions:
        events = pl.read_ndjson(
            root / self._files[0],
            schema={"reviewerID": pl.String, "asin": pl.String, "overall": pl.Float64, "unixReviewTime": pl.Int64},
        ).select(
            pl.col("reviewerID").alias("user"), pl.col("asin").alias("item"),
            pl.col("overall").alias("rating"), pl.col("unixReviewTime").alias("timestamp"),
        )
        items = read_meta(root / self._files[1], set(events["item"].unique())) if self.meta else None
        return Interactions(events=events, items=items)
