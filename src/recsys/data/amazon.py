"""Amazon Reviews 2023 (McAuley Lab): https://amazon-reviews-2023.github.io/"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from recsys.data.base import Dataset, Interactions

_REPO = "McAuley-Lab/Amazon-Reviews-2023"
# Pinned dataset revision on the Hugging Face Hub; the Hub verifies file hashes against it.
_REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"


@dataclass(frozen=True)
class Amazon2023(Dataset):
    """Amazon Reviews 2023 per-category product reviews (Hugging Face Hub, pinned revision).

    ``category`` is the release's file name, e.g. ``All_Beauty``, ``Books``,
    ``Video_Games``. ``events``: user, item (parent ASIN), rating, timestamp.
    ``items``: item, store, main_category.
    """

    category: str = "All_Beauty"
    dirname = "amazon-2023"

    def pins(self) -> tuple[str, ...]:
        return (_REVISION,)

    @property
    def _files(self) -> tuple[str, str]:
        return (
            f"raw/review_categories/{self.category}.jsonl",
            f"raw/meta_categories/meta_{self.category}.jsonl",
        )

    def files_present(self, root: Path) -> bool:
        return all((root / f).exists() for f in self._files)

    def download(self, root: Path) -> None:
        from huggingface_hub import hf_hub_download

        for filename in self._files:
            hf_hub_download(_REPO, filename, repo_type="dataset", revision=_REVISION, local_dir=root)

    def read(self, root: Path) -> Interactions:
        reviews, meta = (root / f for f in self._files)
        events = (
            pl.scan_ndjson(reviews)
            .select(
                pl.col("user_id").alias("user"), pl.col("parent_asin").alias("item"), "rating",
                (pl.col("timestamp") // 1000).cast(pl.Int64).alias("timestamp"),
            )
            .collect()
        )
        items = (
            pl.scan_ndjson(meta, infer_schema_length=None)
            .select(pl.col("parent_asin").alias("item"), "store", "main_category")
            .unique("item")
            .collect()
        )
        return Interactions(events=events, items=items)
