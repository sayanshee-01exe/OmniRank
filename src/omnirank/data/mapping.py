"""Dataset-scoped internal ID mappings - component 4.

Phase 1's :class:`~omnirank.data.id_mapping.IdMapping` remains the in-memory
contract: append-only, fingerprinted, JSON-persisted. This module adds what a
dataset build needs on top of it - Parquet persistence with reverse lookups, and
a metadata record tying the mapping to a dataset version.

Two properties matter more than anything else here:

**Determinism.** External ids are sorted before indices are assigned, so the
same set of ids always produces the same mapping regardless of row order in the
source file. Without that, two runs over identical data produce embedding
matrices whose rows mean different things.

**Fitted on the full post-filtering population, never per split.** Fitting a
mapping on training data alone would leave validation and test items unmappable;
fitting it per split would give the three splits different index spaces. Note
that this is *not* leakage: a mapping is an identifier registry, not a learned
statistic - it encodes which entities exist, which the evaluation protocol
already assumes. Statistics that *are* learned - popularity, user profiles - are
fitted on training rows only. See ``docs/data/leakage_prevention.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd

from omnirank.core.exceptions import IdMappingError
from omnirank.core.logging import get_logger
from omnirank.data.id_mapping import IdMapping
from omnirank.data.io import sha256_frame, write_json, write_parquet

logger = get_logger(__name__)

#: Internal ids start at 0 and are contiguous, matching embedding-matrix row
#: indices directly so no offset arithmetic is needed anywhere downstream.
FIRST_INTERNAL_ID: Final = 0

MAPPING_VERSION: Final = "1"

#: Returned for an external id absent from the mapping. Negative so it can never
#: be mistaken for a valid row index, and never silently indexes from the end of
#: an embedding matrix the way -1 would in NumPy.
UNKNOWN_INTERNAL_ID: Final = -1

USER_MAPPING_COLUMNS: Final = ("external_user_id", "internal_user_id")
ITEM_MAPPING_COLUMNS: Final = ("external_item_id", "internal_item_id")


@dataclass(frozen=True, slots=True)
class EntityMapping:
    """A bidirectional external-to-internal id mapping for one entity type."""

    entity: str
    frame: pd.DataFrame
    external_column: str
    internal_column: str

    @property
    def size(self) -> int:
        """Number of mapped entities."""
        return len(self.frame)

    @property
    def checksum(self) -> str:
        """Order-independent content hash, recorded in the mapping metadata."""
        return sha256_frame(self.frame)

    def to_internal(self) -> dict[str, int]:
        """Forward lookup: external id to internal id."""
        return dict(
            zip(
                self.frame[self.external_column].astype(str),
                self.frame[self.internal_column].astype(int),
                strict=True,
            )
        )

    def to_external(self) -> dict[int, str]:
        """Reverse lookup: internal id to external id."""
        return dict(
            zip(
                self.frame[self.internal_column].astype(int),
                self.frame[self.external_column].astype(str),
                strict=True,
            )
        )

    def as_id_mapping(self) -> IdMapping:
        """Convert to the Phase 1 :class:`IdMapping` contract.

        Lets any Phase 3 model consume this mapping through the interface it
        already expects, and lets the fingerprint be recorded in artifact
        metadata (ADR-006).
        """
        ordered = self.frame.sort_values(self.internal_column)
        return IdMapping(self.entity, ordered[self.external_column].astype(str).tolist())

    def check_contiguous(self) -> None:
        """Assert internal ids are 0..n-1 with no gaps or duplicates.

        Raises:
            IdMappingError: The invariant is broken, which would make an
                embedding matrix the wrong size or misaligned.
        """
        values = self.frame[self.internal_column].to_numpy()
        expected = range(FIRST_INTERNAL_ID, FIRST_INTERNAL_ID + len(values))
        if sorted(values.tolist()) != list(expected):
            raise IdMappingError(
                "Internal ids must be contiguous from the documented start value",
                entity=self.entity,
                size=len(values),
                first_internal_id=FIRST_INTERNAL_ID,
            )


def build_entity_mapping(
    external_ids: pd.Series, *, entity: str, external_column: str, internal_column: str
) -> EntityMapping:
    """Assign contiguous internal ids to sorted, de-duplicated external ids.

    Args:
        external_ids: The external identifiers to map. Duplicates and nulls are
            dropped.
        entity: ``"user"`` or ``"item"``; used in errors and metadata.
        external_column: Output column name for the external id.
        internal_column: Output column name for the internal id.

    Raises:
        IdMappingError: No usable identifiers were supplied.
    """
    unique = external_ids.dropna().astype(str).drop_duplicates().sort_values()
    if unique.empty:
        raise IdMappingError("Cannot build a mapping from zero identifiers", entity=entity)
    frame = pd.DataFrame(
        {
            external_column: unique.to_numpy(),
            internal_column: range(FIRST_INTERNAL_ID, FIRST_INTERNAL_ID + len(unique)),
        }
    )
    mapping = EntityMapping(
        entity=entity,
        frame=frame,
        external_column=external_column,
        internal_column=internal_column,
    )
    mapping.check_contiguous()
    logger.info("mapping.built", entity=entity, size=mapping.size)
    return mapping


@dataclass(frozen=True, slots=True)
class DatasetMappings:
    """User and item mappings for one dataset version."""

    users: EntityMapping
    items: EntityMapping
    dataset_version: str
    mapping_version: str = MAPPING_VERSION
    # Defaults to "now" at construction. A default_factory keeps the field a
    # real datetime rather than an Optional that every reader must narrow.
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def attach_internal_ids(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Add ``internal_user_id``/``internal_item_id`` columns.

        Raises:
            IdMappingError: A row references an id absent from the mapping,
                which means the mapping was fitted on a different population
                than the one being mapped - the exact bug that produces silent
                mis-resolution at serving time.
        """
        frame = interactions.copy()
        frame["internal_user_id"] = (
            frame["external_user_id"].astype(str).map(self.users.to_internal()).astype("Int64")
        )
        frame["internal_item_id"] = (
            frame["external_item_id"].astype(str).map(self.items.to_internal()).astype("Int64")
        )
        unmapped_users = int(frame["internal_user_id"].isna().sum())
        unmapped_items = int(frame["internal_item_id"].isna().sum())
        if unmapped_users or unmapped_items:
            raise IdMappingError(
                "Interactions reference identifiers that are not in the mapping. "
                "The mapping was fitted on a different population than the rows "
                "being mapped.",
                unmapped_users=unmapped_users,
                unmapped_items=unmapped_items,
            )
        frame["internal_user_id"] = frame["internal_user_id"].astype("int64")
        frame["internal_item_id"] = frame["internal_item_id"].astype("int64")
        return frame

    def metadata(self, *, strategy: str = "sorted_external_id_dense_rank") -> dict[str, Any]:
        """The mapping metadata record required by the Phase 2 contract."""
        return {
            "mapping_version": self.mapping_version,
            "dataset_version": self.dataset_version,
            "created_at": self.created_at.isoformat(),
            "number_of_users": self.users.size,
            "number_of_items": self.items.size,
            "user_mapping_checksum": self.users.checksum,
            "item_mapping_checksum": self.items.checksum,
            "user_mapping_fingerprint": self.users.as_id_mapping().fingerprint,
            "item_mapping_fingerprint": self.items.as_id_mapping().fingerprint,
            "mapping_strategy": strategy,
            "first_internal_id": FIRST_INTERNAL_ID,
            "unknown_user_policy": (
                f"Users absent from the mapping resolve to {UNKNOWN_INTERNAL_ID}. Phase 2 "
                "produces no such users: the mapping is fitted on the full "
                "post-filtering population, so every processed row maps. At serving "
                "time an unmapped user is a cold user and is routed to the fallback "
                "chain, never to an embedding lookup."
            ),
            "unknown_item_policy": (
                f"Items absent from the mapping resolve to {UNKNOWN_INTERNAL_ID} and are "
                "excluded from candidate generation. They remain recommendable through "
                "content features once those exist (ADR-003)."
            ),
            "id_reassignment_policy": (
                "Internal ids are stable for a given dataset_version. Changing the "
                "dataset version rebuilds the mapping and invalidates every embedding "
                "trained against the previous one; the fingerprint recorded in artifact "
                "metadata is what makes that mismatch detectable (ADR-006)."
            ),
        }


def write_mappings(
    mappings: DatasetMappings, output_dir: Path | str, *, overwrite: bool = True
) -> dict[str, Any]:
    """Persist both mappings plus their metadata record.

    Returns:
        Descriptors for the three written files, for the dataset manifest.
    """
    directory = Path(output_dir)
    user_descriptor = write_parquet(
        mappings.users.frame,
        directory / "user_id_mapping.parquet",
        columns=USER_MAPPING_COLUMNS,
        sort_by=["internal_user_id"],
        overwrite=overwrite,
    )
    item_descriptor = write_parquet(
        mappings.items.frame,
        directory / "item_id_mapping.parquet",
        columns=ITEM_MAPPING_COLUMNS,
        sort_by=["internal_item_id"],
        overwrite=overwrite,
    )
    metadata_descriptor = write_json(
        mappings.metadata(), directory / "mapping_metadata.json", overwrite=overwrite
    )
    return {
        "user_id_mapping.parquet": user_descriptor,
        "item_id_mapping.parquet": item_descriptor,
        "mapping_metadata.json": metadata_descriptor,
    }


def load_mappings(directory: Path | str, *, dataset_version: str) -> DatasetMappings:
    """Read previously written mappings back.

    Raises:
        IdMappingError: A mapping file is missing.
    """
    path = Path(directory)
    user_path = path / "user_id_mapping.parquet"
    item_path = path / "item_id_mapping.parquet"
    for candidate in (user_path, item_path):
        if not candidate.is_file():
            raise IdMappingError("Mapping file not found", path=str(candidate))
    return DatasetMappings(
        users=EntityMapping(
            entity="user",
            frame=pd.read_parquet(user_path),
            external_column="external_user_id",
            internal_column="internal_user_id",
        ),
        items=EntityMapping(
            entity="item",
            frame=pd.read_parquet(item_path),
            external_column="external_item_id",
            internal_column="internal_item_id",
        ),
        dataset_version=dataset_version,
    )


def build_dataset_mappings(interactions: pd.DataFrame, *, dataset_version: str) -> DatasetMappings:
    """Build user and item mappings from a post-filtering interaction log."""
    return DatasetMappings(
        users=build_entity_mapping(
            interactions["external_user_id"],
            entity="user",
            external_column="external_user_id",
            internal_column="internal_user_id",
        ),
        items=build_entity_mapping(
            interactions["external_item_id"],
            entity="item",
            external_column="external_item_id",
            internal_column="internal_item_id",
        ),
        dataset_version=dataset_version,
    )


__all__ = [
    "FIRST_INTERNAL_ID",
    "ITEM_MAPPING_COLUMNS",
    "MAPPING_VERSION",
    "UNKNOWN_INTERNAL_ID",
    "USER_MAPPING_COLUMNS",
    "DatasetMappings",
    "EntityMapping",
    "build_dataset_mappings",
    "build_entity_mapping",
    "load_mappings",
    "write_mappings",
]
