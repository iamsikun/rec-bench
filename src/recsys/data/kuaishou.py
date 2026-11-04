"""Kuaishou short-video datasets: KuaiRec (https://kuairec.com/) and KuaiRand (https://kuairand.com/)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from recsys.data.base import Dataset, Interactions
from recsys.data.download import fetch

_KUAIREC = (
    "https://zenodo.org/records/18164998/files/KuaiRec.zip",
    "c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c",
)
# variant -> (directory name, archive sha256)
_KUAIRAND = {
    "pure": ("KuaiRand-Pure", "c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0"),
    "1k": ("KuaiRand-1K", "dfaafbb5fd16e9e6d2f9a6adaa4ea25df20a14bc26a90961c136e26c00a7bb2c"),
}


def _find(root: Path, name: str) -> Path | None:
    return next(root.rglob(name), None) if root.is_dir() else None


@dataclass(frozen=True)
class KuaiRec(Dataset):
    """KuaiRec (Gao et al., CIKM 2022) short-video watch logs (Zenodo).

    ``variant="small"`` is the fully observed 1,411 x 3,327 matrix; ``"big"``
    the 7,176 x 10,728 sparse log. ``events``: user, item, timestamp,
    watch_ratio, play_duration, video_duration. ``items``: item, category
    (first listed feature id).
    """

    variant: str = "small"
    dirname = "kuairec"

    def pins(self) -> tuple[str, ...]:
        return (_KUAIREC[1],)

    def files_present(self, root: Path) -> bool:
        return _find(root, f"{self.variant}_matrix.csv") is not None

    def download(self, root: Path) -> None:
        fetch(_KUAIREC[0], root, _KUAIREC[1])

    def read(self, root: Path) -> Interactions:
        events = (
            pl.read_csv(_find(root, f"{self.variant}_matrix.csv"))
            .rename({"user_id": "user", "video_id": "item"})
            .select("user", "item", "watch_ratio", "play_duration", "video_duration",
                    pl.col("timestamp").fill_nan(None).cast(pl.Int64))
        )
        items = (
            pl.read_csv(_find(root, "item_categories.csv"))
            .select(pl.col("video_id").alias("item"),
                    pl.col("feat").str.extract(r"(\d+)").cast(pl.Int64).alias("category"))
        )
        return Interactions(events=events, items=items)


@dataclass(frozen=True)
class KuaiRand(Dataset):
    """KuaiRand (Gao et al., CIKM 2022) exposure logs with click and engagement signals (Zenodo).

    ``variant`` is ``pure`` (27K users x 7.5K videos) or ``1k``. ``events``
    are the standard (non-randomized) logs: user, item, timestamp, time_ms,
    tab, is_click, is_like, long_view, play_time_ms, ... ``users`` and
    ``items`` hold the released user / basic video features.
    """

    variant: str = "pure"
    dirname = "kuairand"

    def pins(self) -> tuple[str, ...]:
        return (_KUAIRAND[self.variant][1],)

    def files_present(self, root: Path) -> bool:
        return (root / _KUAIRAND[self.variant][0] / "data").is_dir()

    def download(self, root: Path) -> None:
        name, digest = _KUAIRAND[self.variant]
        fetch(f"https://zenodo.org/records/10439422/files/{name}.tar.gz", root, digest)

    def read(self, root: Path) -> Interactions:
        name = _KUAIRAND[self.variant][0]
        data = root / name / "data"
        suffix = name.split("-")[1].lower()
        events = (
            pl.concat([pl.read_csv(p) for p in sorted(data.glob("log_standard_*.csv"))])
            .rename({"user_id": "user", "video_id": "item"})
            .with_columns((pl.col("time_ms") // 1000).alias("timestamp"))
        )
        users = pl.read_csv(data / f"user_features_{suffix}.csv").rename({"user_id": "user"})
        items = pl.read_csv(data / f"video_features_basic_{suffix}.csv").rename({"video_id": "item"})
        return Interactions(events=events, users=users, items=items)
