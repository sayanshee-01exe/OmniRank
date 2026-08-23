"""Data ingestion contracts - component 1.

A loader's only job is to turn *some* source (CSV, parquet, a database table, a
vendor export) into the canonical records of :mod:`omnirank.data.schemas`. It
does not clean, filter, deduplicate, or split; those belong to components 3-5.
Keeping ingestion this thin is what makes a new vertical a single new file.

Phase 1 defines the protocol. Concrete loaders (CSV and parquet first) arrive in
Phase 2 - see ``docs/phase_reports/phase_01_report.md``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from omnirank.data.schemas import Interaction, Item, User


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """The three entity streams that make up one dataset snapshot.

    Sequences rather than iterators, because validation and splitting need more
    than one pass. A loader for a dataset too large to materialise should expose
    :class:`StreamingInteractionSource` instead and be consumed in chunks.
    """

    users: Sequence[User]
    items: Sequence[Item]
    interactions: Sequence[Interaction]
    # Free-form provenance recorded into artifact metadata: source paths, row
    # counts, upstream export date. Never contains credentials.
    provenance: dict[str, str]


@runtime_checkable
class DatasetLoader(Protocol):
    """Loads a complete dataset snapshot into canonical records."""

    @property
    def dataset_version(self) -> str:
        """Version string for the snapshot, recorded in artifact metadata."""
        ...

    def load(self) -> DatasetBundle:
        """Read the source and return canonical, *unvalidated* records.

        Records are unvalidated by design: rejecting rows is component 2's
        decision, made against the domain profile, not the loader's.
        """
        ...


@runtime_checkable
class StreamingInteractionSource(Protocol):
    """Yields interactions in chunks, for datasets larger than memory.

    Chunks must be emitted in non-decreasing timestamp order. Temporal splitting
    and sequence building both rely on that ordering, and re-sorting a
    larger-than-memory stream afterwards is exactly what this avoids.
    """

    def iter_chunks(self, chunk_size: int) -> Iterator[Sequence[Interaction]]:
        """Yield successive chunks of at most ``chunk_size`` interactions."""
        ...


__all__ = ["DatasetBundle", "DatasetLoader", "StreamingInteractionSource"]
