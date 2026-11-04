"""Download helpers: checksum-verified atomic HTTP fetch and atomic archive extraction."""

from __future__ import annotations

import hashlib
import logging
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# Zenodo rejects urllib's default User-Agent.
_HEADERS = {"User-Agent": "curl/8.4.0", "Accept": "*/*"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected_sha256: str | None, timeout_s: float = 60.0) -> Path:
    """Stream ``url`` to ``dest`` via a ``.part`` file; rename only if complete and the checksum matches.

    ``expected_sha256=None`` is only for computing a new pin: the digest is logged.
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    LOGGER.info("Downloading %s", url)
    request = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response, part.open("wb") as fh:
            expected_size = response.headers.get("Content-Length")
            shutil.copyfileobj(response, fh, length=1 << 20)
        if expected_size is not None and part.stat().st_size != int(expected_size):
            raise RuntimeError(f"Truncated download of {url}: {part.stat().st_size} of {expected_size} bytes")
        digest = sha256(part)
        if expected_sha256 is None:
            LOGGER.warning("No pinned checksum for %s; sha256=%s", url, digest)
        elif digest != expected_sha256:
            raise RuntimeError(f"Checksum mismatch for {url}: got {digest}, expected {expected_sha256}")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)
    return dest


def extract(archive: Path, dest: Path) -> None:
    """Extract into a staging directory, then move each top-level entry into ``dest``.

    A crash mid-extraction leaves only the staging directory, never a partial tree in ``dest``.
    """
    staging = dest / f".staging-{archive.name}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    LOGGER.info("Extracting %s", archive)
    try:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(staging)
        else:
            with tarfile.open(archive) as tf:
                tf.extractall(staging, filter="data")
        for entry in staging.iterdir():
            target = dest / entry.name
            if target.is_dir():
                shutil.rmtree(target)
            entry.replace(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def fetch(url: str, root: Path, expected_sha256: str | None) -> None:
    """Download a checksum-pinned archive into ``root`` and extract it there."""
    archive = download(url, root / url.rsplit("/", 1)[-1], expected_sha256)
    extract(archive, root)
