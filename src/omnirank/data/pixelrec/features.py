"""Multimodal feature validation and alignment - component 6 (feature side).

PixelRec publishes pre-extracted text and image vectors as two JSON objects of
the shape ``{"<item_id>": [float, ...], ...}``. Measured 2026-08-24:

============  ==========  =====  ==================================
file          size        dim    coverage
============  ==========  =====  ==================================
text_feature   8.65 GiB    1024   all 408,374 full-PixelRec items
image_feature  8.60 GiB    1024   all 408,374 full-PixelRec items
============  ==========  =====  ==================================

PixelRec50K needs only 82,865 of those items - 20% - so 17.3 GB must be read to
keep about 3.4 GB. Two consequences shape this module:

**Nothing is ever fully parsed.** ``json.load`` on an 8.65 GiB file would need
tens of gigabytes of Python objects. :func:`stream_feature_vectors` walks the
file incrementally and emits only the wanted ids, so peak memory is the output,
not the input.

**Absence is a first-class state.** The files are not downloaded by default.
When they are missing, alignment still emits schema-correct index tables with
``has_*_feature = False`` throughout and coverage honestly reported as 0.0.
A missing modality must degrade, never crash (ADR-003), and it must never be
silently reported as present.

Phase 2 does not compute features. No CLIP, no SentenceTransformers, no
fine-tuning - only validation and alignment of what the source published.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from omnirank.core.exceptions import DataSourceError
from omnirank.core.logging import get_logger

logger = get_logger(__name__)

TEXT_FEATURE_FILENAME: Final = "text_feature.json"
IMAGE_FEATURE_FILENAME: Final = "image_feature.json"

#: Read block for the streaming parser. 4 MiB balances syscall overhead against
#: the buffer that must be held while a single record is completed.
_READ_BLOCK_BYTES: Final = 4 * 1024 * 1024

#: A single ``"id": [...]`` record must fit in this much buffer. A 1024-float
#: record is roughly 20 KB; 8 MiB is a wide margin that still catches a
#: malformed file rather than growing memory without bound.
_MAX_RECORD_BYTES: Final = 8 * 1024 * 1024

#: float32 halves the memory of the stored matrix against float64 and is the
#: precision every downstream retrieval model uses anyway.
FEATURE_DTYPE: Final = "float32"

TEXT_INDEX_COLUMNS: Final = (
    "internal_item_id",
    "external_item_id",
    "has_text_feature",
    "text_feature_row",
)
IMAGE_INDEX_COLUMNS: Final = (
    "internal_item_id",
    "external_item_id",
    "has_image_feature",
    "image_feature_row",
)


@dataclass(slots=True)
class FeatureValidation:
    """Validation outcome for one feature modality."""

    modality: str
    available: bool
    source_path: str | None
    source_bytes: int | None
    dimension: int | None
    rows_in_source: int
    rows_matched: int
    items_expected: int
    duplicate_ids: int
    rows_with_nan: int
    rows_with_inf: int
    dimension_mismatches: int
    dtype: str
    normalized: bool
    encoder: str | None
    notes: str = ""

    @property
    def coverage(self) -> float:
        """Fraction of catalogue items that have a feature vector."""
        return self.rows_matched / self.items_expected if self.items_expected else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Report-ready description."""
        return {
            "modality": self.modality,
            "available": self.available,
            "source_path": self.source_path,
            "source_bytes": self.source_bytes,
            "dimension": self.dimension,
            "rows_in_source": self.rows_in_source,
            "rows_matched": self.rows_matched,
            "items_expected": self.items_expected,
            "items_missing": self.items_expected - self.rows_matched,
            "coverage": round(self.coverage, 6),
            "duplicate_ids": self.duplicate_ids,
            "rows_with_nan": self.rows_with_nan,
            "rows_with_inf": self.rows_with_inf,
            "dimension_mismatches": self.dimension_mismatches,
            "dtype": self.dtype,
            "normalized": self.normalized,
            "encoder": self.encoder,
            "notes": self.notes,
        }


def stream_feature_vectors(
    path: Path | str,
    *,
    wanted_ids: set[str] | None = None,
    read_block_bytes: int = _READ_BLOCK_BYTES,
) -> Iterator[tuple[str, list[float]]]:
    """Yield ``(item_id, vector)`` from a large JSON feature object.

    Walks the file incrementally rather than parsing it, so an 8.65 GiB source
    is processed in bounded memory. The expected shape is a flat object whose
    values are arrays of numbers; anything else raises rather than silently
    producing partial output.

    Args:
        path: The JSON feature file.
        wanted_ids: When given, only these ids are emitted. Filtering here
            rather than downstream is what keeps peak memory proportional to the
            wanted subset instead of the file.
        read_block_bytes: Bytes per read.

    Yields:
        ``(item_id, vector)`` pairs in file order.

    Raises:
        DataSourceError: The file is missing, is not a JSON object, contains a
            non-numeric value array, or holds a record larger than the buffer
            bound.
    """
    source = Path(path)
    if not source.is_file():
        raise DataSourceError("Feature file not found", path=str(source))

    buffer = ""
    started = False
    finished = False

    with source.open("r", encoding="utf-8") as stream:
        while not finished:
            block = stream.read(read_block_bytes)
            if block:
                buffer += block
            elif not buffer.strip():
                break

            if not started:
                stripped = buffer.lstrip()
                if not stripped:
                    if not block:
                        break
                    continue
                if not stripped.startswith("{"):
                    raise DataSourceError(
                        "Feature file must be a JSON object mapping item ids to vectors",
                        path=str(source),
                        found=stripped[:40],
                    )
                buffer = stripped[1:]
                started = True

            while True:
                key_start = buffer.find('"')
                if key_start == -1:
                    if buffer.strip().startswith("}") or not buffer.strip():
                        finished = not block or buffer.strip().startswith("}")
                    break
                key_end = buffer.find('"', key_start + 1)
                if key_end == -1:
                    break
                colon = buffer.find(":", key_end + 1)
                open_bracket = buffer.find("[", key_end + 1)
                if colon == -1 or open_bracket == -1:
                    break
                # Values are numeric arrays only, so the first ']' closes this
                # record - no bracket matching is required.
                close_bracket = buffer.find("]", open_bracket + 1)
                if close_bracket == -1:
                    if len(buffer) > _MAX_RECORD_BYTES:
                        raise DataSourceError(
                            "A single feature record exceeded the buffer bound; the "
                            "file is malformed or is not the expected flat object",
                            path=str(source),
                            buffered_bytes=len(buffer),
                        )
                    break

                item_id = buffer[key_start + 1 : key_end]
                payload = buffer[open_bracket : close_bracket + 1]
                buffer = buffer[close_bracket + 1 :]

                if wanted_ids is None or item_id in wanted_ids:
                    try:
                        vector = json.loads(payload)
                    except json.JSONDecodeError as exc:
                        raise DataSourceError(
                            "Feature vector is not a valid JSON array",
                            path=str(source),
                            item_id=item_id,
                            reason=str(exc)[:200],
                        ) from exc
                    yield item_id, vector

            if not block:
                finished = True


def align_features(
    modality: str,
    feature_path: Path | str | None,
    item_mapping: pd.DataFrame,
    *,
    expected_dimension: int | None = None,
    encoder: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None, FeatureValidation]:
    """Align a feature file to the canonical item mapping.

    Args:
        modality: ``"text"`` or ``"image"``.
        feature_path: The JSON feature file, or ``None`` when not downloaded.
        item_mapping: Frame with ``external_item_id``/``internal_item_id``.
        expected_dimension: Assert every vector has this width when given.
        encoder: Source encoder name, recorded in the validation record when
            the source documents one. ``None`` rather than a guess.

    Returns:
        ``(index_frame, matrix, validation)``. The index frame has one row per
        catalogue item with a ``has_*_feature`` flag and a row pointer into
        ``matrix``; ``-1`` means no vector. ``matrix`` is ``None`` when the
        modality is unavailable.
    """
    has_flag = f"has_{modality}_feature"
    row_column = f"{modality}_feature_row"
    columns = TEXT_INDEX_COLUMNS if modality == "text" else IMAGE_INDEX_COLUMNS
    expected_items = len(item_mapping)

    index = item_mapping.loc[:, ["internal_item_id", "external_item_id"]].copy()
    index[has_flag] = False
    index[row_column] = -1

    if feature_path is None or not Path(feature_path).is_file():
        validation = FeatureValidation(
            modality=modality,
            available=False,
            source_path=str(feature_path) if feature_path else None,
            source_bytes=None,
            dimension=None,
            rows_in_source=0,
            rows_matched=0,
            items_expected=expected_items,
            duplicate_ids=0,
            rows_with_nan=0,
            rows_with_inf=0,
            dimension_mismatches=0,
            dtype=FEATURE_DTYPE,
            normalized=False,
            encoder=encoder,
            notes=(
                f"{modality} feature file not present. PixelRec publishes it as a "
                "~8.6 GiB JSON object covering all 408,374 full-PixelRec items; it is "
                "not downloaded by default. Run "
                "`python scripts/download_pixelrec50k.py --with-features` to fetch it. "
                "Coverage is reported as 0.0, not assumed."
            ),
        )
        logger.warning("features.unavailable", **validation.to_dict())
        return (
            index.loc[:, list(columns)].astype({has_flag: "bool", row_column: "int64"}),
            None,
            validation,
        )

    source = Path(feature_path)
    wanted = set(item_mapping["external_item_id"].astype(str))
    external_to_internal = dict(
        zip(
            item_mapping["external_item_id"].astype(str),
            item_mapping["internal_item_id"].astype(int),
            strict=True,
        )
    )

    dimension = expected_dimension
    rows_in_source = 0
    duplicates = 0
    nan_rows = 0
    inf_rows = 0
    mismatches = 0
    seen: set[str] = set()
    matrix: np.ndarray | None = None
    # Accumulated in arrays indexed by internal id, then written to the frame
    # once. The obvious `index.loc[index[...] == internal] = True` inside the
    # loop is a full frame scan per matched item -- O(n^2), about 4.8 billion
    # comparisons at this catalogue size. It is not slow, it does not finish.
    has_feature = np.zeros(expected_items, dtype=bool)
    feature_rows = np.full(expected_items, -1, dtype="int64")

    for item_id, vector in stream_feature_vectors(source, wanted_ids=wanted):
        rows_in_source += 1
        if item_id in seen:
            duplicates += 1
            continue
        seen.add(item_id)

        if dimension is None:
            dimension = len(vector)
            matrix = np.zeros((expected_items, dimension), dtype=FEATURE_DTYPE)
        if len(vector) != dimension:
            mismatches += 1
            continue
        if matrix is None:
            matrix = np.zeros((expected_items, dimension), dtype=FEATURE_DTYPE)

        array = np.asarray(vector, dtype=FEATURE_DTYPE)
        if np.isnan(array).any():
            nan_rows += 1
            continue
        if np.isinf(array).any():
            inf_rows += 1
            continue

        internal = external_to_internal[item_id]
        matrix[internal] = array
        has_feature[internal] = True
        feature_rows[internal] = internal

    # Mapped through the frame's own internal ids rather than assuming row order
    # equals internal id, which happens to be true here and need not stay true.
    internal_ids = index["internal_item_id"].to_numpy(dtype="int64")
    index[has_flag] = has_feature[internal_ids]
    index[row_column] = feature_rows[internal_ids]

    matched = int(index[has_flag].sum())
    validation = FeatureValidation(
        modality=modality,
        available=True,
        source_path=str(source),
        source_bytes=source.stat().st_size,
        dimension=dimension,
        rows_in_source=rows_in_source,
        rows_matched=matched,
        items_expected=expected_items,
        duplicate_ids=duplicates,
        rows_with_nan=nan_rows,
        rows_with_inf=inf_rows,
        dimension_mismatches=mismatches,
        dtype=FEATURE_DTYPE,
        # The source publishes raw encoder outputs and documents no
        # normalisation, so none is claimed and none is applied.
        normalized=False,
        encoder=encoder,
        notes="Feature row index equals internal_item_id; -1 means no vector.",
    )
    logger.info("features.aligned", **validation.to_dict())
    return (
        index.loc[:, list(columns)].astype({has_flag: "bool", row_column: "int64"}),
        matrix,
        validation,
    )


def write_feature_matrix(matrix: np.ndarray | None, path: Path | str) -> dict[str, Any] | None:
    """Persist a feature matrix as a memory-mappable ``.npy`` file.

    Format decision, per the Phase 2 requirement to choose one and justify it:
    ``.npy`` float32. At 82,865 x 1024 that is 339 MiB, small enough to keep and
    large enough that it must be memory-mappable rather than loaded - which
    ``np.load(..., mmap_mode="r")`` gives for free. Parquet fixed-size lists
    would need a decode step on every read, and a ``.pt`` tensor file would make
    torch a dependency of the data layer.
    """
    if matrix is None:
        return None
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, matrix)
    return {
        "path": str(target),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "bytes": target.stat().st_size,
    }


__all__ = [
    "FEATURE_DTYPE",
    "IMAGE_FEATURE_FILENAME",
    "IMAGE_INDEX_COLUMNS",
    "TEXT_FEATURE_FILENAME",
    "TEXT_INDEX_COLUMNS",
    "FeatureValidation",
    "align_features",
    "stream_feature_vectors",
    "write_feature_matrix",
]
