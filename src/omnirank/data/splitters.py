"""Per-user leave-last-N splitting - component 5, concrete implementation.

Implements the Phase 1 :class:`~omnirank.data.splitting.Splitter` contract for
datasets whose evaluation question is "given everything this user did before,
what did they do next?".

The protocol, for each user with enough history:

* the **last** ``test_interactions`` events become the test targets,
* the ``validation_interactions`` immediately before them become validation
  targets,
* everything earlier is training history.

Users with fewer than ``minimum_eligible_interactions`` events are **not
discarded** - they contribute all of their events as training history and are
simply absent from the evaluation sets. Discarding them would shrink the item
catalogue and the collaborative signal for no benefit, and evaluating them would
mean scoring a user whose entire history is the target.

Ordering is by the source's genuine event timestamp, with ``source_row_id`` as a
deterministic tiebreak. PixelRec50K contains no per-user timestamp ties at all,
but the tiebreak is unconditional so that behaviour does not depend on a property
of one dataset. See ``docs/data/interaction_ordering.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import pandas as pd

from omnirank.core.config import SplittingConfig
from omnirank.core.exceptions import DataError
from omnirank.core.logging import get_logger

logger = get_logger(__name__)

TRAIN: Final = "train"
VALIDATION: Final = "validation"
TEST: Final = "test"
SPLIT_NAMES: Final = (TRAIN, VALIDATION, TEST)

#: Columns that define the deterministic chronological order of the log.
ORDER_KEYS: Final = ("external_user_id", "timestamp", "source_row_id")


@dataclass(slots=True)
class SplitResult:
    """A split interaction log plus the statistics describing it."""

    interactions: pd.DataFrame
    strategy: str
    ordering_field: str
    eligible_users: int
    ineligible_users: int
    minimum_user_history: int

    def frame_for(self, split: str) -> pd.DataFrame:
        """Rows belonging to one split."""
        return self.interactions[self.interactions["split"] == split].reset_index(drop=True)

    @property
    def sizes(self) -> dict[str, int]:
        """Row count per split."""
        counts = self.interactions["split"].value_counts().to_dict()
        return {name: int(counts.get(name, 0)) for name in SPLIT_NAMES}

    def statistics(self) -> dict[str, Any]:
        """Full split statistics for the split metadata file."""
        stats: dict[str, Any] = {
            "split_strategy": self.strategy,
            "ordering_field": self.ordering_field,
            "eligible_users": self.eligible_users,
            "ineligible_users": self.ineligible_users,
            "minimum_user_history": self.minimum_user_history,
        }
        for name in SPLIT_NAMES:
            frame = self.interactions[self.interactions["split"] == name]
            stats[f"{name}_rows"] = len(frame)
            stats[f"{name}_users"] = int(frame["external_user_id"].nunique())
            stats[f"{name}_items"] = int(frame["external_item_id"].nunique())
        return stats


def assign_interaction_order(interactions: pd.DataFrame) -> pd.DataFrame:
    """Sort chronologically per user and assign a 0-based per-user rank.

    ``interaction_order`` is the position of an event within its own user's
    history, not a global counter. Sequential models consume it directly, and it
    makes "did this event precede that one?" answerable without re-deriving an
    ordering that must match the splitter's exactly.

    Raises:
        DataError: A required ordering column is missing.
    """
    missing = [column for column in ORDER_KEYS if column not in interactions.columns]
    if missing:
        raise DataError("Cannot order interactions: missing columns", missing=missing)

    ordered = interactions.sort_values(list(ORDER_KEYS), kind="mergesort").reset_index(drop=True)
    ordered["interaction_order"] = ordered.groupby(
        "external_user_id", observed=True, sort=False
    ).cumcount()
    return ordered


def split_leave_last_n(interactions: pd.DataFrame, config: SplittingConfig) -> SplitResult:
    """Partition the log by holding out each eligible user's most recent events.

    Args:
        interactions: Canonical interactions carrying the :data:`ORDER_KEYS`.
        config: Splitting configuration. ``strategy`` must be a leave-last-N one.

    Returns:
        A :class:`SplitResult` whose ``interactions`` frame carries an added
        ``split`` column and the derived ``interaction_order``.

    Raises:
        DataError: The configuration selects a non-leave-last-N strategy, or the
            input is empty.
    """
    if interactions.empty:
        raise DataError("Cannot split an empty interaction log")

    validation_n = config.validation_interactions
    test_n = config.test_interactions
    minimum = config.minimum_eligible_interactions

    ordered = assign_interaction_order(interactions)
    per_user = ordered.groupby("external_user_id", observed=True)["interaction_order"].transform(
        "size"
    )

    # Distance from the end of each user's history: 0 is the most recent event.
    from_end = per_user - 1 - ordered["interaction_order"]
    eligible = per_user >= minimum

    # Default everything to train, then carve the tail out of eligible users.
    split = pd.Series(TRAIN, index=ordered.index, dtype="object")
    is_test = eligible & (from_end < test_n)
    is_validation = eligible & (from_end >= test_n) & (from_end < test_n + validation_n)
    split[is_validation] = VALIDATION
    split[is_test] = TEST
    ordered["split"] = split

    eligible_users = int(ordered.loc[eligible, "external_user_id"].nunique())
    total_users = int(ordered["external_user_id"].nunique())

    result = SplitResult(
        interactions=ordered,
        strategy=config.strategy,
        ordering_field="timestamp",
        eligible_users=eligible_users,
        ineligible_users=total_users - eligible_users,
        minimum_user_history=minimum,
    )
    logger.info(
        "splitting.completed",
        **result.sizes,
        **{
            "eligible_users": eligible_users,
            "ineligible_users": total_users - eligible_users,
            "strategy": config.strategy,
        },
    )
    return result


class PerUserLeaveLastNSplitter:
    """Splitter implementing the leave-last-N protocol.

    Satisfies the Phase 1 ``Splitter`` shape while working on frames rather than
    record sequences, which is what makes it usable at a million rows.
    """

    def __init__(self, config: SplittingConfig) -> None:
        if config.strategy not in {"per_user_leave_last_n", "leave_one_out"}:
            raise DataError(
                "PerUserLeaveLastNSplitter requires a leave-last-N strategy",
                strategy=config.strategy,
            )
        self.config = config

    def split_frame(self, interactions: pd.DataFrame) -> SplitResult:
        """Partition an interaction frame."""
        return split_leave_last_n(interactions, self.config)


__all__ = [
    "ORDER_KEYS",
    "SPLIT_NAMES",
    "TEST",
    "TRAIN",
    "VALIDATION",
    "PerUserLeaveLastNSplitter",
    "SplitResult",
    "assign_interaction_order",
    "split_leave_last_n",
]
