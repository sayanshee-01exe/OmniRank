"""Slice-based evaluation.

A single aggregate number hides the failures that matter. These helpers
re-evaluate a finished recommendation set over sub-populations, so a model that
improves the mean while collapsing on sparse users or the long tail is visible
rather than merely average.

**Slicing has two distinct meanings and they are never mixed:**

* **User slices** restrict *which users are averaged over*. The metric keeps its
  usual definition; only the population changes.
* **Target slices** restrict *which users are averaged over, by a property of
  their held-out target item*. Still a user average - the item property selects
  the users. This is not the same as filtering recommended items, which nothing
  here does.

Slice membership always comes from **training-only** statistics, so it never
depends on the labels being predicted.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from omnirank.core.logging import get_logger
from omnirank.evaluation.base import Recommendations
from omnirank.evaluation.evaluator import OfflineEvaluator
from omnirank.evaluation.ground_truth import EvaluationGroundTruth

logger = get_logger(__name__)

#: Below this many users a sliced metric is too noisy to act on. Reported
#: alongside the value rather than suppressed, so the reader can judge.
SMALL_SLICE_THRESHOLD = 100


class SliceKind(StrEnum):
    """What a slice selects on."""

    USER = "user"
    TARGET_ITEM = "target_item"


#: Phase 2 user-activity slices, in ascending activity order.
USER_ACTIVITY_SLICES = (
    "users_activity_1-3",
    "users_activity_4-10",
    "users_activity_11-30",
    "users_activity_31+",
)

#: Phase 2 item slices used to select users by their target item.
TARGET_ITEM_SLICES = ("items_head", "items_long_tail", "items_cold_start")


@dataclass(slots=True)
class SliceResult:
    """Metrics for one sub-population."""

    slice_name: str
    kind: SliceKind
    users: int
    metrics: dict[str, float]
    #: True when the slice is too small for the metric to be stable.
    small_sample: bool = False
    empty: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Report-ready payload."""
        return {
            "slice_name": self.slice_name,
            "kind": self.kind.value,
            "users": self.users,
            "small_sample": self.small_sample,
            "empty": self.empty,
            **{key: round(value, 6) for key, value in self.metrics.items()},
        }


def _restrict(ground_truth: EvaluationGroundTruth, users: Collection[str]) -> EvaluationGroundTruth:
    """A ground truth covering only the given users."""
    from omnirank.evaluation.base import GroundTruth

    keep = set(users)
    return EvaluationGroundTruth(
        truth=GroundTruth(
            relevant={
                user: items for user, items in ground_truth.truth.relevant.items() if user in keep
            }
        ),
        target_split=ground_truth.target_split,
        fit_splits=ground_truth.fit_splits,
        target_internal_items={
            user: item for user, item in ground_truth.target_internal_items.items() if user in keep
        },
        cold_target_users=frozenset(ground_truth.cold_target_users & keep),
    )


def evaluate_user_slice(
    evaluator: OfflineEvaluator,
    recommendations: Recommendations,
    ground_truth: EvaluationGroundTruth,
    *,
    slice_name: str,
    users: Collection[str],
    k_values: Collection[int],
    kind: SliceKind = SliceKind.USER,
) -> SliceResult:
    """Evaluate one sub-population of users.

    An empty slice returns zeroed metrics flagged ``empty=True`` rather than
    being omitted - ``users_cold_start`` is empty by construction under
    leave-last-N, and silence would read as an oversight.
    """
    members = sorted(set(users) & ground_truth.users)
    if not members:
        return SliceResult(
            slice_name=slice_name,
            kind=kind,
            users=0,
            metrics={},
            small_sample=True,
            empty=True,
        )
    restricted = _restrict(ground_truth, members)
    result = evaluator.evaluate_detailed(
        recommendations, restricted, k_values=sorted(k_values), view="strict"
    )
    return SliceResult(
        slice_name=slice_name,
        kind=kind,
        users=len(members),
        metrics=result.metrics,
        small_sample=len(members) < SMALL_SLICE_THRESHOLD,
    )


def users_in_item_slice(
    ground_truth: EvaluationGroundTruth, internal_item_ids: Collection[int]
) -> list[str]:
    """Users whose held-out **target** belongs to the given item set."""
    members = set(internal_item_ids)
    return [
        user
        for user, internal_item in ground_truth.target_internal_items.items()
        if internal_item in members
    ]


def evaluate_all_slices(
    evaluator: OfflineEvaluator,
    recommendations: Recommendations,
    ground_truth: EvaluationGroundTruth,
    *,
    user_slices: Mapping[str, Collection[str]],
    target_item_slices: Mapping[str, Collection[int]],
    k_values: Collection[int],
) -> list[SliceResult]:
    """Evaluate every user slice and every target-item slice."""
    results = [
        evaluate_user_slice(
            evaluator,
            recommendations,
            ground_truth,
            slice_name=name,
            users=users,
            k_values=k_values,
            kind=SliceKind.USER,
        )
        for name, users in user_slices.items()
    ]
    results += [
        evaluate_user_slice(
            evaluator,
            recommendations,
            ground_truth,
            slice_name=name,
            users=users_in_item_slice(ground_truth, items),
            k_values=k_values,
            kind=SliceKind.TARGET_ITEM,
        )
        for name, items in target_item_slices.items()
    ]
    # Reachability is the slice that explains the strict/warm gap, so it is
    # always reported even though Phase 2 does not ship it as a file.
    results.append(
        evaluate_user_slice(
            evaluator,
            recommendations,
            ground_truth,
            slice_name="targets_reachable_warm",
            users=ground_truth.warm_users,
            k_values=k_values,
            kind=SliceKind.TARGET_ITEM,
        )
    )
    results.append(
        evaluate_user_slice(
            evaluator,
            recommendations,
            ground_truth,
            slice_name="targets_unreachable_cold",
            users=ground_truth.cold_target_users,
            k_values=k_values,
            kind=SliceKind.TARGET_ITEM,
        )
    )
    logger.info(
        "evaluation.slices_completed",
        slices=len(results),
        sizes={result.slice_name: result.users for result in results},
    )
    return results


__all__ = [
    "SMALL_SLICE_THRESHOLD",
    "TARGET_ITEM_SLICES",
    "USER_ACTIVITY_SLICES",
    "SliceKind",
    "SliceResult",
    "evaluate_all_slices",
    "evaluate_user_slice",
    "users_in_item_slice",
]
