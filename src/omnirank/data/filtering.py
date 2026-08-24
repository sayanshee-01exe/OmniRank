"""Iterative k-core filtering - component 3.

Removing users with too few interactions can push items below their threshold,
and removing those items can push further users below theirs. A single pass
therefore leaves the invariant unsatisfied, which is why filtering iterates to a
fixed point and reports every round.

Filtering runs **once, before splitting, on the whole interaction log**. Applying
it to train/validation/test separately would give the three splits different
item vocabularies and different user populations, and every comparison between
them would silently be a comparison of different datasets.

Cold-start information is captured *before* filtering runs, because the
population this stage removes - single-interaction items and short-history users
- is exactly the population cold-start analysis is about, and it is unrecoverable
afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from omnirank.core.exceptions import DataError
from omnirank.core.logging import get_logger

logger = get_logger(__name__)

#: Upper bound on iterations. The fixed point is normally reached in single
#: digits; a runaway loop means the thresholds are mutually unsatisfiable and
#: should fail loudly rather than spin.
MAX_ITERATIONS = 100


@dataclass(slots=True)
class FilterIteration:
    """One round of the fixed-point loop."""

    iteration: int
    users_removed: int
    items_removed: int
    interactions_removed: int
    users_remaining: int
    items_remaining: int
    interactions_remaining: int

    def to_dict(self) -> dict[str, Any]:
        """Report-ready description."""
        return {
            "iteration": self.iteration,
            "users_removed": self.users_removed,
            "items_removed": self.items_removed,
            "interactions_removed": self.interactions_removed,
            "users_remaining": self.users_remaining,
            "items_remaining": self.items_remaining,
            "interactions_remaining": self.interactions_remaining,
        }


@dataclass(slots=True)
class PreFilterSnapshot:
    """Cold-start population captured before filtering destroys it."""

    singleton_items: int
    items_below_item_threshold: int
    users_below_user_threshold: int
    total_users: int
    total_items: int
    total_interactions: int

    def to_dict(self) -> dict[str, Any]:
        """Report-ready description."""
        return {
            "singleton_items": self.singleton_items,
            "items_below_item_threshold": self.items_below_item_threshold,
            "users_below_user_threshold": self.users_below_user_threshold,
            "total_users": self.total_users,
            "total_items": self.total_items,
            "total_interactions": self.total_interactions,
        }


@dataclass(slots=True)
class FilteringResult:
    """Filtered interactions and the full audit trail."""

    interactions: pd.DataFrame
    iterations: list[FilterIteration] = field(default_factory=list)
    snapshot: PreFilterSnapshot | None = None
    enabled: bool = True
    min_user_interactions: int = 0
    min_item_interactions: int = 0
    converged: bool = True
    removed_user_ids: tuple[str, ...] = ()
    removed_item_ids: tuple[str, ...] = ()

    def report(self) -> dict[str, Any]:
        """Report-ready summary of the whole filtering stage."""
        first = self.iterations[0] if self.iterations else None
        return {
            "enabled": self.enabled,
            "converged": self.converged,
            "configuration": {
                "min_user_interactions": self.min_user_interactions,
                "min_item_interactions": self.min_item_interactions,
                "iterative": True,
            },
            "before": self.snapshot.to_dict() if self.snapshot else {},
            "iterations": [item.to_dict() for item in self.iterations],
            "iteration_count": len(self.iterations),
            "after": {
                "users": int(self.interactions["external_user_id"].nunique()),
                "items": int(self.interactions["external_item_id"].nunique()),
                "interactions": len(self.interactions),
            },
            "totals": {
                "users_removed": len(self.removed_user_ids),
                "items_removed": len(self.removed_item_ids),
                "interactions_removed": (
                    (first.interactions_remaining + first.interactions_removed)
                    - len(self.interactions)
                    if first
                    else 0
                ),
            },
        }


def snapshot_before_filtering(
    interactions: pd.DataFrame, *, min_user_interactions: int, min_item_interactions: int
) -> PreFilterSnapshot:
    """Capture the cold-start population that filtering is about to remove."""
    per_user = interactions.groupby("external_user_id", observed=True).size()
    per_item = interactions.groupby("external_item_id", observed=True).size()
    return PreFilterSnapshot(
        singleton_items=int((per_item == 1).sum()),
        items_below_item_threshold=int((per_item < min_item_interactions).sum()),
        users_below_user_threshold=int((per_user < min_user_interactions).sum()),
        total_users=int(per_user.size),
        total_items=int(per_item.size),
        total_interactions=len(interactions),
    )


def apply_iterative_filtering(
    interactions: pd.DataFrame,
    *,
    enabled: bool = True,
    min_user_interactions: int = 0,
    min_item_interactions: int = 0,
    max_iterations: int = MAX_ITERATIONS,
) -> FilteringResult:
    """Filter to a k-core fixed point.

    Args:
        interactions: Canonical interactions with external id columns.
        enabled: When False, returns the input untouched with an empty audit
            trail. Kept as a real option so a run can measure filtering's effect.
        min_user_interactions: Minimum interactions a user must retain.
        min_item_interactions: Minimum interactions an item must retain.
        max_iterations: Safety bound on the fixed-point loop.

    Returns:
        A :class:`FilteringResult` with the surviving interactions and a
        per-iteration record of what was removed.

    Raises:
        DataError: The loop failed to converge, or filtering emptied the
            dataset - both configuration errors worth failing on rather than
            handing an empty frame to the splitter.
    """
    original_users = set(interactions["external_user_id"].unique())
    original_items = set(interactions["external_item_id"].unique())

    if not enabled:
        logger.info("filtering.disabled", rows=len(interactions))
        return FilteringResult(
            interactions=interactions.reset_index(drop=True),
            enabled=False,
            min_user_interactions=min_user_interactions,
            min_item_interactions=min_item_interactions,
        )

    snapshot = snapshot_before_filtering(
        interactions,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )

    frame = interactions
    iterations: list[FilterIteration] = []
    converged = False

    for round_number in range(1, max_iterations + 1):
        before_rows = len(frame)
        per_user = frame.groupby("external_user_id", observed=True).size()
        per_item = frame.groupby("external_item_id", observed=True).size()

        weak_users = set(per_user[per_user < min_user_interactions].index)
        weak_items = set(per_item[per_item < min_item_interactions].index)

        if not weak_users and not weak_items:
            converged = True
            break

        # Both removals apply in the same round: dropping users first and then
        # recomputing item counts would double the number of rounds without
        # changing the fixed point.
        keep = ~(
            frame["external_user_id"].isin(weak_users) | frame["external_item_id"].isin(weak_items)
        )
        frame = frame[keep]

        iterations.append(
            FilterIteration(
                iteration=round_number,
                users_removed=len(weak_users),
                items_removed=len(weak_items),
                interactions_removed=before_rows - len(frame),
                users_remaining=int(frame["external_user_id"].nunique()),
                items_remaining=int(frame["external_item_id"].nunique()),
                interactions_remaining=len(frame),
            )
        )
        logger.info("filtering.iteration", **iterations[-1].to_dict())

        if frame.empty:
            raise DataError(
                "Filtering removed every interaction. The thresholds are too "
                "aggressive for this dataset. If this is a --subset-users run, "
                "that is the usual cause: sampling users leaves most items with a "
                "single interaction, and the item threshold then cascades. Either "
                "raise the subset size, or lower "
                "data.filtering.min_interactions_per_item (1 disables it).",
                min_user_interactions=min_user_interactions,
                min_item_interactions=min_item_interactions,
                iterations_run=round_number,
            )

    if not converged:
        raise DataError(
            "Iterative filtering did not converge. Thresholds are likely "
            "mutually unsatisfiable for this dataset.",
            max_iterations=max_iterations,
            min_user_interactions=min_user_interactions,
            min_item_interactions=min_item_interactions,
        )

    frame = frame.reset_index(drop=True)
    result = FilteringResult(
        interactions=frame,
        iterations=iterations,
        snapshot=snapshot,
        enabled=True,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
        converged=True,
        removed_user_ids=tuple(sorted(original_users - set(frame["external_user_id"].unique()))),
        removed_item_ids=tuple(sorted(original_items - set(frame["external_item_id"].unique()))),
    )
    logger.info("filtering.completed", **result.report()["after"], rounds=len(iterations))
    return result


__all__ = [
    "MAX_ITERATIONS",
    "FilterIteration",
    "FilteringResult",
    "PreFilterSnapshot",
    "apply_iterative_filtering",
    "snapshot_before_filtering",
]
