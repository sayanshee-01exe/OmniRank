"""PixelRec50K raw file loaders - component 1, concrete implementation.

Reads the two CSV files the official PixelRec50K distribution provides and
nothing else. The loader's job stops at "these are the rows, in the source's own
vocabulary"; interpreting them is :mod:`omnirank.data.pixelrec.canonical`'s job,
and rejecting them is :mod:`omnirank.data.cleaning`'s.

Source (verified 2026-08-24):
``https://drive.google.com/drive/folders/1bQPgM-6yAnzcD0jKBoUUheA9LL5xnCHG``

* ``interaction.csv`` - ``item_id,user_id,timestamp``, 989,494 rows
* ``item_info.csv``   - ``item_id`` + 7 engagement counters + ``title,tag,description``,
  82,865 rows

Chunked reads are used throughout. At PixelRec50K scale the whole file fits in
memory comfortably (~50 MB as a dataframe), but the same code must survive
Pixel8M and full PixelRec, where it does not - and a loader that only works at
the size you tested it on is a loader you will rewrite.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pandas as pd

from omnirank.core.exceptions import DataSourceError
from omnirank.core.logging import get_logger
from omnirank.data.io import sha256_file

logger = get_logger(__name__)

INTERACTION_FILENAME: Final = "interaction.csv"
ITEM_INFO_FILENAME: Final = "item_info.csv"

#: Exact header of the official interaction file, asserted on load. A silent
#: column rename upstream would otherwise be discovered as a mis-trained model.
INTERACTION_COLUMNS: Final = ("item_id", "user_id", "timestamp")

#: Exact header of the official item information file.
ITEM_INFO_COLUMNS: Final = (
    "item_id",
    "view_number",
    "comment_number",
    "thumbup_number",
    "share_number",
    "coin_number",
    "favorite_number",
    "barrage_number",
    "title",
    "tag",
    "description",
)

#: Engagement counters. Item-side popularity signals recorded by the source
#: platform - NOT interaction events, and never usable as labels: they describe
#: the whole platform's behaviour, not this dataset's 50,000 users, and they
#: carry no timestamp so they cannot be point-in-time bounded.
ENGAGEMENT_COLUMNS: Final = (
    "view_number",
    "comment_number",
    "thumbup_number",
    "share_number",
    "coin_number",
    "favorite_number",
    "barrage_number",
)

#: Read as strings and cast explicitly, so pandas never infers a numeric type
#: for an identifier column and turns "i0001" into 1.
_INTERACTION_DTYPES: Final = {"item_id": "string", "user_id": "string", "timestamp": "int64"}


@dataclass(frozen=True, slots=True)
class SourceFile:
    """A raw file the pipeline read, described for the manifest."""

    name: str
    path: Path
    bytes: int
    sha256: str
    rows: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Manifest-ready description."""
        return {
            "name": self.name,
            "path": str(self.path),
            "bytes": self.bytes,
            "sha256": self.sha256,
            "rows": self.rows,
        }


@dataclass(slots=True)
class RawPixelRec:
    """The raw PixelRec50K tables, plus provenance for the manifest."""

    interactions: pd.DataFrame
    items: pd.DataFrame
    source_files: list[SourceFile] = field(default_factory=list)

    @property
    def provenance(self) -> dict[str, Any]:
        """Source-file descriptions for the dataset manifest."""
        return {file.name: file.to_dict() for file in self.source_files}


class PixelRec50KLoader:
    """Loads PixelRec50K from a directory of raw files.

    Args:
        raw_dir: Directory holding ``interaction.csv`` and ``item_info.csv``.
        chunk_size: Rows per read block. Bounds peak memory during load.
        compute_checksums: Hash source files. Costs one extra full read of
            ~51 MB; skip it in tight development loops via configuration.
        subset_users: Keep only the first N users encountered, for development
            runs. ``None`` loads everything. Selection is deterministic:
            users are sorted before truncation, so the same N always yields the
            same users regardless of file order.
    """

    def __init__(
        self,
        raw_dir: Path | str,
        *,
        chunk_size: int = 100_000,
        compute_checksums: bool = True,
        subset_users: int | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.chunk_size = max(1, chunk_size)
        self.compute_checksums = compute_checksums
        self.subset_users = subset_users

    # -- paths and validation ---------------------------------------------- #
    @property
    def interaction_path(self) -> Path:
        """Path to the interaction CSV."""
        return self.raw_dir / INTERACTION_FILENAME

    @property
    def item_info_path(self) -> Path:
        """Path to the item information CSV."""
        return self.raw_dir / ITEM_INFO_FILENAME

    def check_files_present(self) -> None:
        """Verify the expected files exist before any expensive work.

        Raises:
            DataSourceError: The directory or a required file is missing, with
                the download instructions named rather than a bare path.
        """
        if not self.raw_dir.is_dir():
            raise DataSourceError(
                "PixelRec50K raw directory not found. Run "
                "`python scripts/download_pixelrec50k.py` or see "
                "docs/data/pixelrec50k_overview.md for manual download steps.",
                raw_dir=str(self.raw_dir),
            )
        missing = [
            name
            for name, path in (
                (INTERACTION_FILENAME, self.interaction_path),
                (ITEM_INFO_FILENAME, self.item_info_path),
            )
            if not path.is_file()
        ]
        if missing:
            raise DataSourceError(
                "PixelRec50K raw directory is missing required files. Run "
                "`python scripts/download_pixelrec50k.py`.",
                raw_dir=str(self.raw_dir),
                missing=missing,
            )

    def _validate_header(self, path: Path, expected: tuple[str, ...]) -> None:
        """Assert a CSV's header matches the official schema exactly."""
        try:
            header = pd.read_csv(path, nrows=0, encoding="utf-8").columns.tolist()
        except Exception as exc:
            raise DataSourceError(
                "Could not read the CSV header. The file may be truncated, "
                "compressed, or not a CSV at all.",
                path=str(path),
                reason=str(exc)[:300],
            ) from exc
        if tuple(header) != expected:
            raise DataSourceError(
                "Source file header does not match the expected PixelRec50K schema. "
                "Verify the download came from the official PixelRec50K folder and "
                "not another PixelRec variant.",
                path=str(path),
                expected=list(expected),
                found=header,
            )

    def _describe(self, path: Path, rows: int | None = None) -> SourceFile:
        """Build a manifest descriptor for a source file."""
        return SourceFile(
            name=path.name,
            path=path,
            bytes=path.stat().st_size,
            sha256=sha256_file(path) if self.compute_checksums else "",
            rows=rows,
        )

    # -- chunked readers ---------------------------------------------------- #
    def iter_interaction_chunks(self) -> Iterator[pd.DataFrame]:
        """Yield interaction chunks in source order.

        Source order carries no chronological meaning - PixelRec50K's file is
        not sorted by timestamp - so downstream code must sort explicitly. See
        docs/data/interaction_ordering.md.
        """
        reader = pd.read_csv(
            self.interaction_path,
            dtype=_INTERACTION_DTYPES,
            chunksize=self.chunk_size,
            encoding="utf-8",
        )
        offset = 0
        for chunk in reader:
            # A stable, reproducible source row identifier. Needed because
            # PixelRec provides no interaction id, and rejected-record reports
            # must point back at a specific line of the original file.
            chunk = chunk.assign(source_row_id=range(offset, offset + len(chunk)))
            offset += len(chunk)
            yield chunk

    def load_interactions(self) -> pd.DataFrame:
        """Read the full interaction table, applying the development subset."""
        self._validate_header(self.interaction_path, INTERACTION_COLUMNS)
        chunks = list(self.iter_interaction_chunks())
        frame = (
            pd.concat(chunks, ignore_index=True)
            if chunks
            else pd.DataFrame(columns=[*INTERACTION_COLUMNS, "source_row_id"])
        )
        logger.info(
            "pixelrec.interactions_loaded",
            path=str(self.interaction_path),
            rows=len(frame),
            chunks=len(chunks),
            chunk_size=self.chunk_size,
            memory_mb=round(frame.memory_usage(deep=True).sum() / 1e6, 1),
        )
        if self.subset_users is not None:
            frame = self._apply_user_subset(frame)
        return frame

    def _apply_user_subset(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keep every interaction of the first N users, sorted by user id.

        Whole users are kept rather than a random sample of rows: truncating a
        user's history mid-way would silently change the very thing the split
        and sequence stages measure.
        """
        assert self.subset_users is not None  # noqa: S101 - guarded by the caller
        keep = sorted(frame["user_id"].dropna().unique())[: self.subset_users]
        subset = frame[frame["user_id"].isin(set(keep))].reset_index(drop=True)
        logger.warning(
            "pixelrec.user_subset_applied",
            requested_users=self.subset_users,
            kept_users=len(keep),
            rows_before=len(frame),
            rows_after=len(subset),
            note="development subset - results are NOT comparable to a full run",
        )
        return subset

    def load_items(self) -> pd.DataFrame:
        """Read the full item information table.

        Not chunked on the read path: 82,865 rows with free-text descriptions is
        ~60 MB in memory, and the columns must be parsed together anyway because
        quoted descriptions can span the chunk boundary.
        """
        self._validate_header(self.item_info_path, ITEM_INFO_COLUMNS)
        frame = pd.read_csv(
            self.item_info_path,
            dtype={
                "item_id": "string",
                "title": "string",
                "tag": "string",
                "description": "string",
            },
            encoding="utf-8",
        )
        logger.info(
            "pixelrec.items_loaded",
            path=str(self.item_info_path),
            rows=len(frame),
            memory_mb=round(frame.memory_usage(deep=True).sum() / 1e6, 1),
        )
        return frame

    # -- entry point --------------------------------------------------------- #
    def load(self) -> RawPixelRec:
        """Load both tables with provenance.

        Returns:
            A :class:`RawPixelRec` holding the source-vocabulary frames and the
            checksummed file descriptions the manifest needs.
        """
        self.check_files_present()
        interactions = self.load_interactions()
        items = self.load_items()
        return RawPixelRec(
            interactions=interactions,
            items=items,
            source_files=[
                self._describe(self.interaction_path, rows=len(interactions)),
                self._describe(self.item_info_path, rows=len(items)),
            ],
        )


__all__ = [
    "ENGAGEMENT_COLUMNS",
    "INTERACTION_COLUMNS",
    "INTERACTION_FILENAME",
    "ITEM_INFO_COLUMNS",
    "ITEM_INFO_FILENAME",
    "PixelRec50KLoader",
    "RawPixelRec",
    "SourceFile",
]
