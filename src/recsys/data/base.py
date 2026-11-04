"""Dataset base class and the canonical in-memory format."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import polars as pl

LOGGER = logging.getLogger(__name__)


def data_root() -> Path:
    """Directory holding all downloaded datasets (``$RECSYS_DATA``, default ``~/.cache/recsys``)."""
    return Path(os.environ.get("RECSYS_DATA", "~/.cache/recsys")).expanduser()


@dataclass
class Interactions:
    """Raw tables of a dataset.

    ``events`` has at least ``user`` and ``item`` columns, plus ``timestamp``
    (integer Unix seconds) when the source provides one and any raw signal
    columns (``rating``, ``is_click``, ...). ``users`` / ``items`` are optional side-feature tables
    keyed by ``user`` / ``item``. Ids are raw source ids; benchmarks encode them.
    """

    events: pl.DataFrame
    users: pl.DataFrame | None = None
    items: pl.DataFrame | None = None


class Dataset(ABC):
    """A public dataset that knows how to download and read itself.

    Subclasses are frozen dataclasses (their ``repr`` feeds benchmark version
    hashes) and set ``dirname``, the per-source directory under :func:`data_root`.
    Every downloaded artifact is pinned by SHA-256 (or, on the Hugging Face
    Hub, by revision); :meth:`pins` exposes those pins to the version hash.
    """

    dirname: str
    #: False when the source publishes no interaction times (``timestamp`` is then 0).
    has_time: bool = True

    @property
    def root(self) -> Path:
        return data_root() / self.dirname

    def pins(self) -> tuple[str, ...]:
        """Checksums / revisions of the artifacts this instance downloads."""
        return ()

    @abstractmethod
    def files_present(self, root: Path) -> bool:
        """Whether the files needed by :meth:`read` exist under ``root``."""

    @abstractmethod
    def download(self, root: Path) -> None:
        """Fetch the original files into ``root``."""

    @abstractmethod
    def read(self, root: Path) -> Interactions:
        """Read the files under ``root`` into :class:`Interactions`."""

    def load(self, download: bool = True) -> Interactions:
        """Return the dataset, downloading it first if it is not cached."""
        root = self.root
        if not self.files_present(root):
            if not download:
                raise FileNotFoundError(f"{self!r} not found under {root}")
            LOGGER.info("Downloading %r into %s", self, root)
            root.mkdir(parents=True, exist_ok=True)
            self.download(root)
        return self.read(root)
