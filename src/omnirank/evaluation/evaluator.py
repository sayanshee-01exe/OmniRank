"""The offline evaluator.

Implements the Phase 1 :class:`~omnirank.evaluation.base.Evaluator` contract.

Two rules it never breaks, both of which exist because breaking them produces
numbers that look better and mean less:

* **The denominator is every user in the ground truth.** A user who received no
  recommendations scores zero. Dropping them measures only the traffic the model
  can already serve.
* **It never touches the model.** It receives finished recommendation lists and
  does not reorder, filter, or extend them. A metric that can re-rank the thing
  it measures is not a measurement.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from omnirank.core.logging import get_logger
from omnirank.evaluation.base import Evaluator, GroundTruth, Recommendations
from omnirank.evaluation.beyond_accuracy import BeyondAccuracyResult, compute_beyond_accuracy
from omnirank.evaluation.ground_truth import EvaluationGroundTruth
from omnirank.evaluation.metrics import (
    METRIC_FUNCTIONS,
    RANKING_METRICS,
    metric_name,
    validate_k,
)

logger = get_logger(__name__)

#: Metrics reported with confidence intervals and used for model selection.
PRIMARY_METRICS: tuple[str, ...] = ("recall@20", "ndcg@20")

STRICT = "strict"
WARM = "warm"


@dataclass(slots=True)
class EvaluationResult:
    """Metrics plus everything needed to interpret them."""

    metrics: dict[str, float]
    protocol: str
    #: "strict" (all held-out users) or "warm" (reachable targets only).
    view: str
    users_evaluated: int
    users_with_recommendations: int
    user_coverage: float
    #: Per-user primary metric values, kept for bootstrap resampling.
    per_user: dict[str, dict[str, float]] = field(default_factory=dict)
    beyond_accuracy: list[BeyondAccuracyResult] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def flat(self) -> dict[str, float]:
        """Flat metric dictionary, including beyond-accuracy entries."""
        payload = dict(self.metrics)
        for result in self.beyond_accuracy:
            payload.update({key: float(value) for key, value in result.to_dict().items()})
        payload["user_coverage"] = self.user_coverage
        return payload


def _score_one_user(
    recommended: Sequence[str],
    relevant_grades: Mapping[str, float],
    k_values: tuple[int, ...],
    max_k: int,
    metrics: tuple[str, ...],
) -> dict[str, float]:
    """All metrics at all cut-offs for one user, from a single list scan.

    The list is walked once up to ``max_k``, recording the 1-based rank and gain
    of every relevant hit. Every metric is then a small arithmetic expression
    over that record, which removes the 24 redundant scans the naive loop
    performed per user.
    """
    values: dict[str, float] = {}
    total_relevant = len(relevant_grades)
    if total_relevant == 0:
        return {metric_name(metric, k): 0.0 for metric in metrics for k in k_values}

    # (rank, gain) for each relevant item found within max_k.
    hits: list[tuple[int, float]] = []
    for rank, item in enumerate(recommended[:max_k], start=1):
        grade = relevant_grades.get(item)
        if grade is not None:
            hits.append((rank, grade))

    ideal_gains = sorted(relevant_grades.values(), reverse=True)

    for k in k_values:
        within = [(rank, gain) for rank, gain in hits if rank <= k]
        hit_count = len(within)

        if "recall" in metrics:
            values[metric_name("recall", k)] = hit_count / total_relevant
        if "precision" in metrics:
            values[metric_name("precision", k)] = hit_count / k
        if "hit_rate" in metrics:
            values[metric_name("hit_rate", k)] = 1.0 if hit_count else 0.0
        if "mrr" in metrics:
            values[metric_name("mrr", k)] = 1.0 / within[0][0] if within else 0.0
        if "map" in metrics:
            denominator = min(total_relevant, k)
            accumulated = sum((index + 1) / rank for index, (rank, _) in enumerate(within))
            values[metric_name("map", k)] = accumulated / denominator if denominator else 0.0
        if "ndcg" in metrics:
            actual = sum(gain / math.log2(rank + 1) for rank, gain in within)
            ideal = sum(
                gain / math.log2(position + 1)
                for position, gain in enumerate(ideal_gains[:k], start=1)
            )
            values[metric_name("ndcg", k)] = actual / ideal if ideal > 0 else 0.0

    return values


class OfflineEvaluator(Evaluator):
    """Computes ranking and beyond-accuracy metrics for one recommendation set."""

    def __init__(
        self,
        *,
        metrics: Sequence[str] = RANKING_METRICS,
        novelty_smoothing: float = 1.0,
        gini_includes_zero_exposure: bool = True,
    ) -> None:
        unknown = set(metrics) - set(METRIC_FUNCTIONS)
        if unknown:
            from omnirank.core.exceptions import DataError

            raise DataError(
                "Unknown ranking metric requested",
                unknown=sorted(unknown),
                available=sorted(METRIC_FUNCTIONS),
            )
        self.metrics = tuple(metrics)
        self.novelty_smoothing = novelty_smoothing
        self.gini_includes_zero_exposure = gini_includes_zero_exposure

    # -- Phase 1 contract --------------------------------------------------- #
    def evaluate(
        self,
        recommendations: Recommendations,
        ground_truth: GroundTruth,
        k_values: list[int],
    ) -> dict[str, float]:
        """Score recommendations against held-out truth.

        Returns a flat ``{"<metric>@<k>": value}`` mapping averaged over every
        user in ``ground_truth``.
        """
        return self._score(recommendations, ground_truth, tuple(k_values), ground_truth.users)[0]

    # -- richer entry point ------------------------------------------------- #
    def evaluate_detailed(
        self,
        recommendations: Recommendations,
        ground_truth: EvaluationGroundTruth,
        *,
        k_values: Sequence[int],
        view: str = STRICT,
        eligible_catalogue: Collection[str] | None = None,
        training_counts: Mapping[str, int] | None = None,
        category_by_item: Mapping[str, str] | None = None,
    ) -> EvaluationResult:
        """Evaluate under the strict or warm view, with beyond-accuracy metrics.

        Args:
            recommendations: Finished recommendation lists.
            ground_truth: Targets plus their provenance.
            k_values: Cut-offs.
            view: ``"strict"`` scores every held-out user, counting a target the
                model could never retrieve as a miss - end-to-end system
                performance. ``"warm"`` scores only users whose target is in the
                model's fit catalogue - collaborative ranking quality in
                isolation. Neither is reported without the other.
            eligible_catalogue: Denominator for coverage and Gini. Required for
                beyond-accuracy metrics.
            training_counts: Training-only item counts, for novelty.
            category_by_item: Item categories, for category diversity.

        Returns:
            An :class:`EvaluationResult`.
        """
        users = ground_truth.users if view == STRICT else ground_truth.warm_users
        cuts = tuple(validate_k(int(k)) for k in k_values)
        metrics, per_user = self._score(recommendations, ground_truth.truth, cuts, users)

        with_recommendations = sum(1 for user in users if recommendations.items_for(user))
        coverage = with_recommendations / len(users) if users else 0.0

        beyond: list[BeyondAccuracyResult] = []
        if eligible_catalogue is not None and training_counts is not None:
            for k in cuts:
                lists = [list(recommendations.items_for(user))[:k] for user in users]
                exposure: dict[str, int] = {}
                for items in lists:
                    for item in items:
                        exposure[item] = exposure.get(item, 0) + 1
                beyond.append(
                    compute_beyond_accuracy(
                        lists,
                        exposure,
                        k=k,
                        eligible_catalogue=eligible_catalogue,
                        training_counts=training_counts,
                        category_by_item=category_by_item,
                        novelty_smoothing=self.novelty_smoothing,
                        gini_includes_zero_exposure=self.gini_includes_zero_exposure,
                    )
                )

        result = EvaluationResult(
            metrics=metrics,
            protocol="full",
            view=view,
            users_evaluated=len(users),
            users_with_recommendations=with_recommendations,
            user_coverage=coverage,
            per_user=per_user,
            beyond_accuracy=beyond,
            provenance=ground_truth.provenance(),
        )
        logger.info(
            "evaluation.completed",
            view=view,
            users=len(users),
            user_coverage=round(coverage, 6),
            **{key: round(value, 6) for key, value in metrics.items() if key in PRIMARY_METRICS},
        )
        return result

    # -- internals ---------------------------------------------------------- #
    def _score(
        self,
        recommendations: Recommendations,
        ground_truth: GroundTruth,
        k_values: tuple[int, ...],
        users: Collection[str],
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        """Average each metric over ``users``, keeping per-user values.

        Computes each user's hit positions once and derives every metric at
        every cut-off from them, rather than re-scanning the recommendation list
        24 times per user. Same arithmetic as the reference functions in
        :mod:`omnirank.evaluation.metrics`; a test asserts the two agree on
        randomised inputs, so the reference stays the specification.
        """
        totals: dict[str, float] = {
            metric_name(metric, k): 0.0 for metric in self.metrics for k in k_values
        }
        per_user: dict[str, dict[str, float]] = {}
        population = len(users)
        if population == 0:
            return dict.fromkeys(totals, 0.0), per_user

        max_k = max(k_values)
        for user in users:
            recommended = recommendations.items_for(user)
            relevant_grades = ground_truth.items_for(user)
            user_values = _score_one_user(
                recommended, relevant_grades, k_values, max_k, self.metrics
            )
            for key, value in user_values.items():
                totals[key] += value
            per_user[user] = user_values

        return {key: total / population for key, total in totals.items()}, per_user


__all__ = ["PRIMARY_METRICS", "STRICT", "WARM", "EvaluationResult", "OfflineEvaluator"]
