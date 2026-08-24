"""Deterministic cleaning and rejected-record capture - component 3.

Two rules govern this module.

**Nothing is dropped silently.** Every rejected row is written to
``rejected_records.parquet`` with its source file, source row identifier, entity
type, reason, and original identifier. A pipeline that quietly discards 3% of
its input produces a model nobody can explain.

**Every stage reconciles.** Each :class:`CleaningStep` reports
``input_rows = output_rows + removed_rows`` and fails loudly if that does not
hold. Row-count arithmetic that does not add up means a join fanned out or a
filter matched something unintended, and it is far cheaper to catch here than to
discover as an unexplained metric shift three phases later.

Cleaning is a pure function of its inputs and configuration: no clock, no RNG,
no I/O beyond the rejected-record sink.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from omnirank.core.exceptions import DataError
from omnirank.core.logging import get_logger

logger = get_logger(__name__)


class RejectionReason(StrEnum):
    """Stable identifiers for every reason a row is rejected.

    Counted, logged, and written into the rejected-records table, so these are
    part of the observable contract and must not be renamed casually.
    """

    MISSING_USER_ID = "missing_user_id"
    MISSING_ITEM_ID = "missing_item_id"
    EMPTY_USER_ID = "empty_user_id"
    EMPTY_ITEM_ID = "empty_item_id"
    MISSING_TIMESTAMP = "missing_timestamp"
    INVALID_TIMESTAMP = "invalid_timestamp"
    TIMESTAMP_OUT_OF_RANGE = "timestamp_out_of_range"
    FUTURE_TIMESTAMP = "future_timestamp"
    UNKNOWN_EVENT_TYPE = "unknown_event_type"
    INVALID_WEIGHT = "invalid_weight"
    DUPLICATE_INTERACTION = "duplicate_interaction"
    DUPLICATE_ITEM_ID = "duplicate_item_id"
    UNKNOWN_ITEM_REFERENCE = "unknown_item_reference"
    FILTERED_SPARSE_USER = "filtered_sparse_user"
    FILTERED_SPARSE_ITEM = "filtered_sparse_item"


REJECTED_COLUMNS = (
    "source_file",
    "source_row_identifier",
    "entity_type",
    "rejection_reason",
    "original_identifier",
)


@dataclass(slots=True)
class CleaningStep:
    """Row-count reconciliation for one cleaning operation."""

    name: str
    input_rows: int
    output_rows: int
    modified_rows: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)

    @property
    def removed_rows(self) -> int:
        """Rows the step removed."""
        return self.input_rows - self.output_rows

    def check(self) -> None:
        """Assert the arithmetic holds.

        Raises:
            DataError: Rows were gained, or the recorded reasons do not account
                for every removal.
        """
        if self.output_rows > self.input_rows:
            raise DataError(
                f"cleaning step {self.name!r} produced more rows than it consumed, "
                "which means a join fanned out",
                input_rows=self.input_rows,
                output_rows=self.output_rows,
            )
        accounted = sum(self.reason_counts.values())
        if accounted != self.removed_rows:
            raise DataError(
                f"cleaning step {self.name!r} removed {self.removed_rows} rows but "
                f"recorded reasons for {accounted}; every removal must be explained",
                step=self.name,
                reason_counts=self.reason_counts,
            )

    def to_dict(self) -> dict[str, Any]:
        """Report-ready description."""
        return {
            "step": self.name,
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "removed_rows": self.removed_rows,
            "modified_rows": self.modified_rows,
            "reason_counts": dict(sorted(self.reason_counts.items())),
        }


class RejectedRecords:
    """Accumulates rejected rows for later persistence."""

    def __init__(self) -> None:
        self._frames: list[pd.DataFrame] = []

    def add(
        self,
        rows: pd.DataFrame,
        *,
        source_file: str,
        entity_type: str,
        reason: RejectionReason,
        identifier_column: str,
        row_id_column: str | None = None,
    ) -> int:
        """Record a set of rejected rows. Returns how many were recorded."""
        if rows.empty:
            return 0
        row_ids = (
            rows[row_id_column].astype("string")
            if row_id_column and row_id_column in rows.columns
            else pd.Series(rows.index.astype(str), index=rows.index, dtype="string")
        )
        identifiers = (
            rows[identifier_column].astype("string")
            if identifier_column in rows.columns
            else pd.Series([pd.NA] * len(rows), index=rows.index, dtype="string")
        )
        self._frames.append(
            pd.DataFrame(
                {
                    "source_file": source_file,
                    "source_row_identifier": row_ids.to_numpy(),
                    "entity_type": entity_type,
                    "rejection_reason": reason.value,
                    "original_identifier": identifiers.to_numpy(),
                }
            )
        )
        return len(rows)

    def to_frame(self) -> pd.DataFrame:
        """All rejected rows as one frame, empty but typed when there are none."""
        if not self._frames:
            return pd.DataFrame({column: pd.Series(dtype="string") for column in REJECTED_COLUMNS})
        return pd.concat(self._frames, ignore_index=True).loc[:, list(REJECTED_COLUMNS)]

    @property
    def count(self) -> int:
        """Total rejected rows recorded."""
        return sum(len(frame) for frame in self._frames)


@dataclass(slots=True)
class CleaningResult:
    """Cleaned canonical frames plus the audit trail explaining every removal."""

    users: pd.DataFrame
    items: pd.DataFrame
    interactions: pd.DataFrame
    steps: list[CleaningStep]
    rejected: pd.DataFrame

    def report(self) -> dict[str, Any]:
        """Report-ready summary."""
        return {
            "steps": [step.to_dict() for step in self.steps],
            "total_rejected": len(self.rejected),
            "rejected_by_reason": (
                self.rejected["rejection_reason"].value_counts().to_dict()
                if len(self.rejected)
                else {}
            ),
            "output_rows": {
                "users": len(self.users),
                "items": len(self.items),
                "interactions": len(self.interactions),
            },
        }


def clean_items(
    items: pd.DataFrame, sink: RejectedRecords, *, source_file: str = "item_info.csv"
) -> tuple[pd.DataFrame, CleaningStep]:
    """Remove items with unusable identifiers and collapse duplicate ids.

    Deliberately *not* removed: items missing a title, description, category, or
    engagement counters. Those are documented coverage gaps, and an item with an
    id is still recommendable via collaborative signal alone.
    """
    step = CleaningStep(name="clean_items", input_rows=len(items), output_rows=0)
    frame = items

    blank = frame["external_item_id"].isna() | frame["external_item_id"].astype(
        "string"
    ).str.strip().eq("")
    if blank.any():
        step.reason_counts[RejectionReason.MISSING_ITEM_ID.value] = sink.add(
            frame[blank],
            source_file=source_file,
            entity_type="item",
            reason=RejectionReason.MISSING_ITEM_ID,
            identifier_column="external_item_id",
        )
        frame = frame[~blank]

    duplicated = frame["external_item_id"].duplicated(keep="first")
    if duplicated.any():
        # Keep-first is deterministic only because the source order is stable and
        # checksummed; a differing duplicate is reported, not merged, because
        # choosing between two conflicting titles is a data-owner decision.
        step.reason_counts[RejectionReason.DUPLICATE_ITEM_ID.value] = sink.add(
            frame[duplicated],
            source_file=source_file,
            entity_type="item",
            reason=RejectionReason.DUPLICATE_ITEM_ID,
            identifier_column="external_item_id",
        )
        frame = frame[~duplicated]

    frame = frame.reset_index(drop=True)
    step.output_rows = len(frame)
    step.check()
    return frame, step


def clean_interactions(
    interactions: pd.DataFrame,
    known_item_ids: set[str],
    sink: RejectedRecords,
    *,
    min_timestamp: pd.Timestamp,
    max_timestamp: pd.Timestamp,
    allowed_event_types: set[str],
    drop_duplicates: bool = True,
    source_file: str = "interaction.csv",
) -> tuple[pd.DataFrame, CleaningStep]:
    """Remove unusable interactions, in a fixed order, recording every removal.

    Order matters and is fixed: identifier checks, then timestamp checks, then
    vocabulary and weight checks, then referential integrity, then
    deduplication. Deduplication runs last so that a duplicate of an
    already-rejected row is not counted twice.

    Args:
        interactions: Canonical interaction frame.
        known_item_ids: Item ids that survived item cleaning.
        sink: Where rejected rows are recorded.
        min_timestamp: Floor below which a timestamp is a clock error.
        max_timestamp: Ceiling above which a timestamp is in the future.
        allowed_event_types: Vocabulary declared by the domain profile.
        drop_duplicates: Apply the documented deduplication policy.
        source_file: Name recorded in the rejected-records table.
    """
    step = CleaningStep(name="clean_interactions", input_rows=len(interactions), output_rows=0)
    frame = interactions

    def _reject(mask: pd.Series, reason: RejectionReason, identifier: str) -> None:
        nonlocal frame
        if not mask.any():
            return
        step.reason_counts[reason.value] = step.reason_counts.get(reason.value, 0) + sink.add(
            frame[mask],
            source_file=source_file,
            entity_type="interaction",
            reason=reason,
            identifier_column=identifier,
            row_id_column="source_row_id",
        )
        frame = frame[~mask]

    user_blank = frame["external_user_id"].isna() | frame["external_user_id"].astype(
        "string"
    ).str.strip().eq("")
    _reject(user_blank, RejectionReason.MISSING_USER_ID, "external_user_id")

    item_blank = frame["external_item_id"].isna() | frame["external_item_id"].astype(
        "string"
    ).str.strip().eq("")
    _reject(item_blank, RejectionReason.MISSING_ITEM_ID, "external_item_id")

    _reject(frame["timestamp"].isna(), RejectionReason.MISSING_TIMESTAMP, "external_user_id")

    # Non-positive epoch seconds are sentinel values, not instants in 1970.
    _reject(frame["timestamp"] <= 0, RejectionReason.INVALID_TIMESTAMP, "external_user_id")

    below = frame["event_timestamp_utc"] < min_timestamp
    _reject(below, RejectionReason.TIMESTAMP_OUT_OF_RANGE, "external_user_id")

    ahead = frame["event_timestamp_utc"] > max_timestamp
    _reject(ahead, RejectionReason.FUTURE_TIMESTAMP, "external_user_id")

    unknown_event = ~frame["event_type"].isin(allowed_event_types)
    _reject(unknown_event, RejectionReason.UNKNOWN_EVENT_TYPE, "external_user_id")

    # Weights must be finite and non-negative. `errors="coerce"` turns a
    # non-numeric value into NaN, which the isna() branch then catches, so a
    # garbage string and a missing value are rejected by the same rule.
    weight = pd.to_numeric(frame["interaction_weight"], errors="coerce")
    bad_weight = weight.isna() | ~np.isfinite(weight.to_numpy(dtype="float64")) | (weight < 0)
    _reject(bad_weight, RejectionReason.INVALID_WEIGHT, "external_user_id")

    dangling = ~frame["external_item_id"].isin(known_item_ids)
    _reject(dangling, RejectionReason.UNKNOWN_ITEM_REFERENCE, "external_item_id")

    if drop_duplicates:
        # Deduplication policy: (user, item, event_type, timestamp) is the
        # business key. PixelRec assigns no interaction id, and a source that
        # re-emits the same event must not double-count it. Documented in
        # docs/data/cleaning_rules.md.
        duplicated = frame.duplicated(
            subset=["external_user_id", "external_item_id", "event_type", "timestamp"],
            keep="first",
        )
        _reject(duplicated, RejectionReason.DUPLICATE_INTERACTION, "external_user_id")

    frame = frame.reset_index(drop=True)
    step.output_rows = len(frame)
    step.check()
    logger.info("cleaning.interactions", **step.to_dict())
    return frame, step


__all__ = [
    "REJECTED_COLUMNS",
    "CleaningResult",
    "CleaningStep",
    "RejectedRecords",
    "RejectionReason",
    "clean_interactions",
    "clean_items",
]
