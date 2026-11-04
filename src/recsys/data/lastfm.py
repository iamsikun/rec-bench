"""Last.fm 1K listening histories (Celma, 2010): http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from recsys.data.base import Dataset, Interactions
from recsys.data.download import fetch

_URL = "https://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"
_SHA256 = "c8075a048b0e14737c0b38d708f9e8e56a37193f33c6592e599bfcaa982bdb70"
_DIR = "lastfm-dataset-1K"
_EVENTS = "userid-timestamp-artid-artname-traid-traname.tsv"
_PROFILE = "userid-profile.tsv"
_LEVELS = {"artist": "artist_id", "track": "track_id"}


def read_tsv(path: Path, columns: list[str]) -> pl.DataFrame:
    """Read a headerless, unquoted tab-separated file of strings (names may contain quotes)."""
    return pl.read_csv(
        path, separator="\t", has_header=False, new_columns=columns, quote_char=None,
        schema_overrides=dict.fromkeys(columns, pl.String), comment_prefix="#",
    )


@dataclass(frozen=True)
class LastFM1K(Dataset):
    """LastFM1K (Celma / MTG-UPF): full scrobble histories of 992 Last.fm users, 2005-2009.

    ``level`` chooses the item: ``artist`` (MusicBrainz artist id) or
    ``track`` (MusicBrainz track id). Scrobbles whose chosen id is missing in
    the release are dropped. ``events``: user, item, timestamp (seconds),
    artist_id, artist_name, track_id, track_name. ``users``: user, gender,
    age, country, registered (as released, e.g. ``Aug 13, 2006``).
    """

    level: str = "artist"
    dirname = "lastfm-1k"

    def __post_init__(self) -> None:
        if self.level not in _LEVELS:
            raise ValueError(f"Unknown LastFM-1K level {self.level!r}; choose from {list(_LEVELS)}")

    def pins(self) -> tuple[str, ...]:
        return (_SHA256,)

    def files_present(self, root: Path) -> bool:
        return all((root / _DIR / f).exists() for f in (_EVENTS, _PROFILE))

    def download(self, root: Path) -> None:
        fetch(_URL, root, _SHA256)

    def read(self, root: Path) -> Interactions:
        cols = ["user", "time", "artist_id", "artist_name", "track_id", "track_name"]
        events = (
            read_tsv(root / _DIR / _EVENTS, cols)
            .filter(pl.col(_LEVELS[self.level]).is_not_null())
            .select(
                "user", pl.col(_LEVELS[self.level]).alias("item"),
                pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC").dt.epoch("s").alias("timestamp"),
                "artist_id", "artist_name", "track_id", "track_name",
            )
        )
        users = read_tsv(root / _DIR / _PROFILE, ["user", "gender", "age", "country", "registered"])
        users = users.with_columns(pl.col("age").cast(pl.Int32, strict=False))
        return Interactions(events=events, users=users)
