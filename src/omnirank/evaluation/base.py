"""Offline evaluation contracts - component 9.

An evaluator turns "recommendations plus what actually happened" into numbers.
The contract is narrow on purpose: an :class:`Evaluator` receives already-made
recommendations and never calls a model itself. That separation means the same
evaluator scores a popularity baseline, a LightGCN retriever, and a full
multi-stage pipeline, and that a metric can never accidentally re-rank the thing
it is measuring.

Two contracts:

* :class:`Evaluator` - the metric computation.
* :class:`GroundTruth` - the held-out relevance signal, built from the test
  split by the same code for every model, so no model gets an easier target.

Phase 3 implements these contracts in :mod:`omnirank.evaluation.evaluator` and
:mod:`omnirank.evaluation.metrics`, validated against hand-computed fixtures
before any model result was reported.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from omnirank.core.exceptions import DataError


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Held-out relevant items per user, with optional graded relevance.

    ``relevance`` is used by graded metrics (NDCG); binary metrics (recall,
    hit-rate) read only the key set. Users with no held-out items are excluded
    at construction, because including them makes recall unboundedly pessimistic
    and the number stops being comparable between runs.
    """

    # user_id -> {item_id: relevance grade}
    relevant: Mapping[str, Mapping[str, float]]

    def __post_init__(self) -> None:
        empty = [user for user, items in self.relevant.items() if not items]
        if empty:
            raise DataError(
                f"{len(empty)} users have no held-out items. Exclude them before "
                "constructing GroundTruth, or every recall number is diluted by "
                "users that could not have been satisfied.",
                example_users=empty[:3],
            )

    @property
    def users(self) -> frozenset[str]:
        """Users with held-out relevance."""
        return frozenset(self.relevant)

    def items_for(self, user_id: str) -> Mapping[str, float]:
        """Relevant items for one user; empty mapping when unknown."""
        return self.relevant.get(user_id, {})


@runtime_checkable
class Recommendations(Protocol):
    """What an evaluator is handed: ordered item ids per user."""

    def users(self) -> Sequence[str]:
        """Users that received recommendations."""
        ...

    def items_for(self, user_id: str) -> Sequence[str]:
        """Recommended item ids for one user, best first."""
        ...


class Evaluator(ABC):
    """Computes offline metrics for a set of recommendations."""

    @abstractmethod
    def evaluate(
        self,
        recommendations: Recommendations,
        ground_truth: GroundTruth,
        k_values: list[int],
    ) -> dict[str, float]:
        """Score recommendations against held-out truth.

        Args:
            recommendations: Ordered item ids per user.
            ground_truth: Held-out relevant items per user.
            k_values: Cut-offs to report. Every accuracy metric is reported at
                every cut-off, keyed ``"<metric>@<k>"``.

        Returns:
            Flat metric name to value. Names must be stable across runs so that
            two runs' reports can be diffed mechanically.

        Implementations must state their denominator explicitly: metrics are
        averaged over users present in ``ground_truth``, and a user who received
        no recommendations scores zero rather than being dropped. Dropping them
        is the most common way an offline number ends up flattering a model that
        cannot serve half its traffic.
        """


__all__ = ["Evaluator", "GroundTruth", "Recommendations"]
