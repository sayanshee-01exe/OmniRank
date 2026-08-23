"""Temporal splitting contracts - component 5.

Recommendation systems predict the future, so evaluation must too. Random
splitting lets a model see event *t + 1* while predicting event *t* for the same
user, which inflates every metric and produces a model that is worse online than
offline. OmniRank therefore splits strictly by time. See ADR-002 for the full
argument and the alternatives considered.

Phase 1 defines the split contract and the invariants any implementation must
satisfy; :func:`check_split_integrity` is a real, runnable guard that Phase 2's
splitter will be tested against.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from omnirank.core.config import SplittingConfig
from omnirank.core.exceptions import DataError
from omnirank.data.schemas import Interaction


@dataclass(frozen=True, slots=True)
class SplitBoundaries:
    """The two instants that separate the three windows.

    Everything strictly before ``validation_start`` trains; ``[validation_start,
    test_start)`` validates; everything from ``test_start`` onward tests.
    """

    validation_start: datetime
    test_start: datetime

    def __post_init__(self) -> None:
        if self.validation_start >= self.test_start:
            raise DataError(
                "validation_start must precede test_start",
                validation_start=self.validation_start.isoformat(),
                test_start=self.test_start.isoformat(),
            )


@dataclass(frozen=True, slots=True)
class DataSplit:
    """One temporal train/validation/test partition of an interaction log."""

    train: Sequence[Interaction]
    validation: Sequence[Interaction]
    test: Sequence[Interaction]
    boundaries: SplitBoundaries
    strategy: str

    @property
    def sizes(self) -> dict[str, int]:
        """Row count per split, for the run report."""
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


@runtime_checkable
class Splitter(Protocol):
    """Partitions an interaction log into train/validation/test."""

    def split(self, interactions: Sequence[Interaction], config: SplittingConfig) -> DataSplit:
        """Partition by time according to ``config.strategy``.

        Implementations must be deterministic: the same input and config produce
        the same partition, with no dependence on input ordering or RNG state.
        """
        ...


def check_split_integrity(split: DataSplit, *, embargo_seconds: float = 0.0) -> None:
    """Assert the invariants that make a temporal split trustworthy.

    Checks, in order:

    1. No interaction sits in more than one split.
    2. Every training event precedes ``validation_start``.
    3. Every validation event lies in ``[validation_start, test_start)``.
    4. Every test event is at or after ``test_start``.
    5. The gap between the last training event and ``validation_start`` (and
       likewise for validation/test) respects the embargo.

    Args:
        split: The partition to check.
        embargo_seconds: Minimum required gap at each boundary. Mirrors
            ``data.splitting.embargo_days``.

    Raises:
        DataError: Any invariant is violated, naming which one.
    """
    train_ids = {event.interaction_id for event in split.train}
    validation_ids = {event.interaction_id for event in split.validation}
    test_ids = {event.interaction_id for event in split.test}

    for left_name, left, right_name, right in (
        ("train", train_ids, "validation", validation_ids),
        ("train", train_ids, "test", test_ids),
        ("validation", validation_ids, "test", test_ids),
    ):
        overlap = left & right
        if overlap:
            raise DataError(
                f"{len(overlap)} interactions appear in both the {left_name} and "
                f"{right_name} splits, which leaks labels into training",
                example=sorted(overlap)[:3],
            )

    bounds = split.boundaries
    for name, events, lower, upper in (
        ("train", split.train, None, bounds.validation_start),
        ("validation", split.validation, bounds.validation_start, bounds.test_start),
        ("test", split.test, bounds.test_start, None),
    ):
        for event in events:
            if lower is not None and event.timestamp < lower:
                raise DataError(
                    f"{name} split contains an event before its window",
                    interaction_id=event.interaction_id,
                    timestamp=event.timestamp.isoformat(),
                    window_start=lower.isoformat(),
                )
            if upper is not None and event.timestamp >= upper:
                raise DataError(
                    f"{name} split contains an event at or after its window",
                    interaction_id=event.interaction_id,
                    timestamp=event.timestamp.isoformat(),
                    window_end=upper.isoformat(),
                )

    if embargo_seconds > 0:
        for name, events, boundary in (
            ("train", split.train, bounds.validation_start),
            ("validation", split.validation, bounds.test_start),
        ):
            if not events:
                continue
            latest = max(event.timestamp for event in events)
            gap = (boundary - latest).total_seconds()
            if gap < embargo_seconds:
                raise DataError(
                    f"{name} split violates the embargo: its last event is {gap:.0f}s "
                    f"before the boundary, but {embargo_seconds:.0f}s are required",
                    boundary=boundary.isoformat(),
                )


__all__ = ["DataSplit", "SplitBoundaries", "Splitter", "check_split_integrity"]
