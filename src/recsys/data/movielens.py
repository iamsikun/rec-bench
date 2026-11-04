"""MovieLens (GroupLens) rating datasets: https://grouplens.org/datasets/movielens/"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from recsys.data.base import Dataset, Interactions
from recsys.data.download import fetch

_URL = "https://files.grouplens.org/datasets/movielens/{}.zip"
# variant -> (archive stem, extracted directory, archive sha256)
_VARIANTS = {
    "100k": ("ml-100k", "ml-100k", "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"),
    "1m": ("ml-1m", "ml-1m", "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"),
    "10m": ("ml-10m", "ml-10M100K", "813c411ccb6122564edfe752e7f80c4dcc5aa25fa94c93622f6877a7ba252862"),
    "20m": ("ml-20m", "ml-20m", "96f243c338a8665f6bcc89c53edf6ee39162a846940de6b7c8c48aeada765ff3"),
    "25m": ("ml-25m", "ml-25m", "8b21cfb7eb1706b4ec0aac894368d90acf26ebdfb6aced3ebd4ad5bd1eb9c6aa"),
    "32m": ("ml-32m", "ml-32m", "e4a68655d7386b8f95f2f2424b2ff975dfdd15ffd59e0d864a14dca43e99d6ee"),
}
_GENRES_100K = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def _read_sep(path: Path, columns: list[str], sep: str) -> pl.DataFrame:
    """Read a headerless file with a (possibly multi-character) separator."""
    text = path.read_bytes().decode("latin-1").replace(sep, "\t")
    return pl.read_csv(
        io.StringIO(text), separator="\t", has_header=False, new_columns=columns,
        quote_char=None, infer_schema_length=10000,
    )


@dataclass(frozen=True)
class MovieLens(Dataset):
    """MovieLens explicit 1-5 star ratings with timestamps (GroupLens).

    ``events``: user, item, rating, timestamp. ``items``: item, genre (first
    listed genre). ``users`` (100k and 1m only): user, gender, age,
    occupation, zip.
    """

    variant: str = "1m"
    dirname = "movielens"

    def __post_init__(self) -> None:
        if self.variant not in _VARIANTS:
            raise ValueError(f"Unknown MovieLens variant {self.variant!r}; choose from {list(_VARIANTS)}")

    def pins(self) -> tuple[str, ...]:
        return (_VARIANTS[self.variant][2],)

    @property
    def _dir(self) -> str:
        return _VARIANTS[self.variant][1]

    def files_present(self, root: Path) -> bool:
        return (root / self._dir).is_dir()

    def download(self, root: Path) -> None:
        stem, _, digest = _VARIANTS[self.variant]
        fetch(_URL.format(stem), root, digest)

    def read(self, root: Path) -> Interactions:
        d = root / self._dir
        users = None
        if self.variant == "100k":
            events = _read_sep(d / "u.data", ["user", "item", "rating", "timestamp"], "\t")
            users = _read_sep(d / "u.user", ["user", "age", "gender", "occupation", "zip"], "|")
            raw = _read_sep(d / "u.item", ["item", "title", "date", "video", "url", *_GENRES_100K], "|")
            first = pl.coalesce([pl.when(pl.col(g) == 1).then(pl.lit(g)) for g in _GENRES_100K])
            items = raw.select("item", first.alias("genre"))
        elif self.variant in ("1m", "10m"):
            events = _read_sep(d / "ratings.dat", ["user", "item", "rating", "timestamp"], "::")
            if self.variant == "1m":
                users = _read_sep(d / "users.dat", ["user", "gender", "age", "occupation", "zip"], "::")
            items = _read_sep(d / "movies.dat", ["item", "title", "genres"], "::")
        else:
            events = pl.read_csv(d / "ratings.csv").rename({"userId": "user", "movieId": "item"})
            items = pl.read_csv(d / "movies.csv").rename({"movieId": "item"})
        if "genres" in items.columns:
            items = items.select("item", pl.col("genres").str.split("|").list.first().alias("genre"))
        events = events.with_columns(pl.col("timestamp").cast(pl.Int64))
        return Interactions(events=events, users=users, items=items)
