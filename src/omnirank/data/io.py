"""Deterministic Parquet and checksum helpers.

Every processed artifact in Phase 2 goes through this module, for three reasons:

* **Determinism.** Two runs over the same input must produce byte-identical
  files, otherwise the manifest checksums are noise. That means a fixed column
  order, a fixed row order, no index column, and a fixed compression codec.
* **One place to change the format.** Swapping compression or engine is a
  one-line edit here rather than a search across a dozen writers.
* **Checksums at the point of writing**, so the manifest never has to re-read a
  large file to describe it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from omnirank.core.exceptions import DataError
from omnirank.core.logging import get_logger

logger = get_logger(__name__)

#: Read in 8 MiB blocks when hashing: large enough to be fast, small enough
#: that hashing a multi-gigabyte file never grows the resident set.
_HASH_BLOCK_BYTES = 8 * 1024 * 1024

#: zstd beats snappy on both size and speed for these columnar string-heavy
#: tables, and pyarrow ships it by default.
PARQUET_COMPRESSION: Literal["zstd"] = "zstd"


def sha256_file(path: Path | str, *, block_bytes: int = _HASH_BLOCK_BYTES) -> str:
    """Return the SHA-256 of a file, read incrementally.

    Raises:
        DataError: The file does not exist.
    """
    target = Path(path)
    if not target.is_file():
        raise DataError("Cannot checksum a file that does not exist", path=str(target))
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_frame(frame: pd.DataFrame) -> str:
    """Return a content hash of a dataframe's values, independent of row order.

    Used for mapping checksums, where "the same mapping" must hash identically
    regardless of how the rows happened to be materialised.
    """
    canonical = frame.sort_values(list(frame.columns)).reset_index(drop=True)
    # `to_numpy()` rather than `.values`: the latter is typed as possibly an
    # ExtensionArray, which has no tobytes(). hash_pandas_object always yields
    # uint64, so the conversion is exact.
    hashed = pd.util.hash_pandas_object(canonical, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def write_parquet(
    frame: pd.DataFrame,
    path: Path | str,
    *,
    columns: Sequence[str] | None = None,
    sort_by: Sequence[str] | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Write a dataframe deterministically and describe what was written.

    Args:
        frame: Data to write.
        path: Destination; parent directories are created.
        columns: Exact column order to emit. Acts as a schema assertion - a
            missing column is an error rather than a silently absent field.
        sort_by: Columns to sort by before writing, for byte-stable output.
        overwrite: When False, refuse to replace an existing file.

    Returns:
        A descriptor with ``path``, ``rows``, ``columns``, ``bytes``, ``sha256``.

    Raises:
        DataError: A requested column is missing, or the file exists and
            ``overwrite`` is False.
    """
    target = Path(path)
    if target.exists() and not overwrite:
        raise DataError(
            "Refusing to overwrite an existing output. Pass --overwrite to replace it.",
            path=str(target),
        )

    out = frame
    if columns is not None:
        missing = [name for name in columns if name not in out.columns]
        if missing:
            raise DataError(
                "Output frame is missing required columns",
                path=str(target),
                missing=missing,
                present=list(out.columns),
            )
        out = out.loc[:, list(columns)]
    if sort_by:
        out = out.sort_values(list(sort_by), kind="mergesort")
    out = out.reset_index(drop=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(target, index=False, compression=PARQUET_COMPRESSION, engine="pyarrow")

    descriptor = {
        "path": str(target),
        "rows": len(out),
        "columns": list(out.columns),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }
    logger.info(
        "io.parquet_written",
        path=str(target),
        rows=descriptor["rows"],
        bytes=descriptor["bytes"],
    )
    return descriptor


def write_json(payload: Any, path: Path | str, *, overwrite: bool = True) -> dict[str, Any]:
    """Write JSON with sorted keys and a trailing newline, then describe it."""
    target = Path(path)
    if target.exists() and not overwrite:
        raise DataError(
            "Refusing to overwrite an existing output. Pass --overwrite to replace it.",
            path=str(target),
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def write_text(content: str, path: Path | str, *, overwrite: bool = True) -> dict[str, Any]:
    """Write a text report and describe it."""
    target = Path(path)
    if target.exists() and not overwrite:
        raise DataError(
            "Refusing to overwrite an existing output. Pass --overwrite to replace it.",
            path=str(target),
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content if content.endswith("\n") else content + "\n")
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def concat_chunks(chunks: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate chunk frames, returning an empty frame when there are none.

    ``pd.concat`` raises on an empty sequence, which turns "the input file had
    no rows" - a legitimate, reportable state - into a crash.
    """
    collected = [chunk for chunk in chunks if len(chunk)]
    if not collected:
        return pd.DataFrame()
    return pd.concat(collected, ignore_index=True)


__all__ = [
    "PARQUET_COMPRESSION",
    "concat_chunks",
    "sha256_file",
    "sha256_frame",
    "write_json",
    "write_parquet",
    "write_text",
]
