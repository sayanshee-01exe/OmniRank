"""Ground-truth construction from a held-out split.

The ground truth is built once, by one function, for every model - so no model
can be scored against an easier target than another. It records which split it
came from and which splits the model was fitted on, because a metric without
that context is not comparable to anything.

Public boundaries use **external** ids: a `GroundTruth` holds the same string ids
a model's `recommend()` returns, so the evaluator never touches an internal
index. Internal ids stay inside the model, where they belong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from omnirank.core.exceptions import DataError
from omnirank.core.logging import get_logger
from omnirank.evaluation.base import GroundTruth

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationGroundTruth:
    """Held-out targets plus the provenance needed to interpret them."""

    truth: GroundTruth
    #: Which split the targets came from: "validation" or "test".
    target_split: str
    #: Which splits the model was fitted on, defining the seen-item boundary.
    fit_splits: tuple[str, ...]
    #: Internal item ids of every target, aligned by user, for warm/cold analysis.
    target_internal_items: dict[str, int]
    #: Targets whose item is absent from the model's fit catalogue.
    cold_target_users: frozenset[str]

    @property
    def users(self) -> frozenset[str]:
        """Every user with a held-out target."""
        return self.truth.users

    @property
    def warm_users(self) -> frozenset[str]:
        """Users whose target the model could possibly retrieve."""
        return self.users - self.cold_target_users

    @property
    def reachable_fraction(self) -> float:
        """Fraction of targets present in the fit catalogue."""
        total = len(self.users)
        return len(self.warm_users) / total if total else 0.0

    def provenance(self) -> dict[str, Any]:
        """Report-ready description of how this ground truth was built."""
        return {
            "target_split": self.target_split,
            "fit_splits": list(self.fit_splits),
            "users": len(self.users),
            "warm_users": len(self.warm_users),
            "cold_target_users": len(self.cold_target_users),
            "reachable_fraction": round(self.reachable_fraction, 6),
        }


def build_ground_truth(
    targets: pd.DataFrame,
    *,
    target_split: str,
    fit_splits: tuple[str, ...],
    fit_item_ids: set[int],
    internal_to_external_item: dict[int, str],
    internal_to_external_user: dict[int, str],
    fit_interactions: pd.DataFrame | None = None,
) -> EvaluationGroundTruth:
    """Build ground truth from a held-out split.

    Args:
        targets: The held-out interactions, with internal ids and
            ``interaction_order``.
        target_split: Name of the split the targets came from.
        fit_splits: Splits the model was fitted on.
        fit_item_ids: Internal item ids the model can retrieve. Used to classify
            targets as warm or cold - **not** to remove them.
        internal_to_external_item: Reverse item mapping.
        internal_to_external_user: Reverse user mapping.
        fit_interactions: When given, asserts that every user's fit history
            strictly precedes their target. Cheap insurance against a
            mis-specified fit boundary silently leaking the future.

    Returns:
        An :class:`EvaluationGroundTruth`.

    Raises:
        DataError: The targets are empty, a user has more than one target with
            conflicting relevance, or a fit history does not precede its target.

    Note:
        Cold targets stay in the ground truth. Removing them would turn a real
        end-to-end failure into an invisible one; the strict protocol counts
        them as misses and the warm protocol reports them separately.
    """
    if targets.empty:
        raise DataError(
            "Cannot build ground truth from an empty target split", target_split=target_split
        )

    if fit_interactions is not None:
        _assert_history_precedes_targets(fit_interactions, targets, target_split=target_split)

    relevant: dict[str, dict[str, float]] = {}
    target_internal: dict[str, int] = {}
    cold: set[str] = set()

    for internal_user, internal_item in zip(
        targets["internal_user_id"].to_numpy(dtype="int64"),
        targets["internal_item_id"].to_numpy(dtype="int64"),
        strict=True,
    ):
        external_user = internal_to_external_user[int(internal_user)]
        external_item = internal_to_external_item[int(internal_item)]
        # Graded relevance is 1.0 throughout: PixelRec records one
        # undifferentiated implicit signal, so there is nothing to grade with.
        relevant.setdefault(external_user, {})[external_item] = 1.0
        target_internal[external_user] = int(internal_item)
        if int(internal_item) not in fit_item_ids:
            cold.add(external_user)

    ground_truth = EvaluationGroundTruth(
        truth=GroundTruth(relevant=relevant),
        target_split=target_split,
        fit_splits=tuple(fit_splits),
        target_internal_items=target_internal,
        cold_target_users=frozenset(cold),
    )
    logger.info("ground_truth.built", **ground_truth.provenance())
    return ground_truth


def _assert_history_precedes_targets(
    fit_interactions: pd.DataFrame, targets: pd.DataFrame, *, target_split: str
) -> None:
    """Verify every user's fit history ends before their held-out target.

    Raises:
        DataError: Any user has a fit interaction at or after their target.
    """
    fit_max = fit_interactions.groupby("internal_user_id", observed=True)["interaction_order"].max()
    target_min = targets.groupby("internal_user_id", observed=True)["interaction_order"].min()
    shared = fit_max.index.intersection(target_min.index)
    offenders = shared[fit_max[shared] >= target_min[shared]]
    if len(offenders):
        raise DataError(
            "Fit history does not precede the evaluation targets for every user. "
            "The fit boundary is wrong, and any metric computed from it would be "
            "optimistic.",
            target_split=target_split,
            offending_users=len(offenders),
            examples=offenders[:5].tolist(),
        )


__all__ = ["EvaluationGroundTruth", "build_ground_truth"]
