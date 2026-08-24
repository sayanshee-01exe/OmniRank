"""Ranking metric implementations.

Pure functions over one user's recommendation list and relevant-item set. They
know nothing about models, splits, or configuration, which is what makes them
testable against hand-computed values - and every one of them is, in
``tests/unit/evaluation/test_metrics.py``.

Conventions applied uniformly:

* ``k`` counts positions, 1-based; a list shorter than ``k`` is scored as-is
  rather than padded.
* A user with no recommendations scores **0.0**, never ``nan`` and never
  excluded. Dropping such users is the most common way an offline number ends up
  flattering a model that cannot serve part of its traffic.
* A target absent from the model's catalogue is simply not retrieved, so it
  scores 0.0 through the ordinary path. No special case is needed, and adding
  one would hide the failure.

**One-positive redundancy.** PixelRec50K holds out exactly one item per user.
Under that condition ``recall@k == hit_rate@k``, ``map@k == mrr@k``, and
``precision@k == recall@k / k`` - the three pairs are algebraically identical,
not independent evidence. All six are implemented correctly because a future
dataset may hold out several items, but reports must not present them as
corroborating one another. See ``docs/evaluation/metric_definitions.md``.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from typing import Final

from omnirank.core.exceptions import DataError

#: Metrics computed at every configured cut-off.
RANKING_METRICS: Final = ("recall", "precision", "ndcg", "map", "mrr", "hit_rate")


def validate_k(k: int) -> int:
    """Return ``k`` if it is a usable cut-off.

    Raises:
        DataError: ``k`` is not a positive integer. A zero or negative cut-off
            has no meaning and would silently return 0.0 for every user.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise DataError("k must be a positive integer", k=k)
    return k


def _hits(recommended: Sequence[str], relevant: Collection[str], k: int) -> list[int]:
    """1/0 relevance flags for the top-k positions, in rank order."""
    return [1 if item in relevant else 0 for item in recommended[:k]]


def recall_at_k(recommended: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of relevant items retrieved in the top ``k``.

    Returns 0.0 when the user has no relevant items, rather than dividing by
    zero - though ground truth construction already excludes such users.
    """
    validate_k(k)
    if not relevant:
        return 0.0
    return sum(_hits(recommended, relevant, k)) / len(relevant)


def precision_at_k(recommended: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of the top ``k`` positions that are relevant.

    The denominator is ``k``, not ``len(recommended)``. A model returning three
    items where twenty were asked for is penalised for the seventeen it did not
    supply - which is the honest accounting when the list is what gets shown.
    """
    validate_k(k)
    return sum(_hits(recommended, relevant, k)) / k


def hit_rate_at_k(recommended: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1.0 if any relevant item appears in the top ``k``, else 0.0."""
    validate_k(k)
    return 1.0 if any(_hits(recommended, relevant, k)) else 0.0


def reciprocal_rank_at_k(recommended: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Reciprocal of the first relevant item's 1-based rank, 0.0 if none."""
    validate_k(k)
    for position, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def average_precision_at_k(recommended: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Average of precision@i over the positions holding a relevant item.

    Normalised by ``min(len(relevant), k)`` - the best achievable number of hits
    within the cut-off - so a user with more relevant items than positions is not
    penalised for the shortfall.
    """
    validate_k(k)
    if not relevant:
        return 0.0
    hits = 0
    accumulated = 0.0
    for position, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            accumulated += hits / position
    denominator = min(len(relevant), k)
    return accumulated / denominator if denominator else 0.0


def dcg_at_k(gains: Sequence[float]) -> float:
    """Discounted cumulative gain with the standard log2(rank + 1) discount."""
    return sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, start=1))


def ndcg_at_k(
    recommended: Sequence[str],
    relevant: Collection[str],
    k: int,
    *,
    relevance: dict[str, float] | None = None,
) -> float:
    """Normalised discounted cumulative gain.

    Args:
        recommended: Ordered item ids.
        relevant: Relevant item ids.
        k: Cut-off.
        relevance: Optional graded relevance. Defaults to binary 1.0, which is
            what PixelRec's single implicit signal supports - grading it would
            mean inventing a preference strength the source never recorded.

    The ideal DCG uses the best ``min(len(relevant), k)`` gains, so a perfect
    ranking scores exactly 1.0 even when there are more relevant items than
    positions.
    """
    validate_k(k)
    if not relevant:
        return 0.0
    grades = relevance or dict.fromkeys(relevant, 1.0)
    actual = dcg_at_k(
        [grades.get(item, 0.0) if item in relevant else 0.0 for item in recommended[:k]]
    )
    ideal_gains = sorted((grades.get(item, 1.0) for item in relevant), reverse=True)[:k]
    ideal = dcg_at_k(ideal_gains)
    return actual / ideal if ideal > 0 else 0.0


#: Dispatch table used by the evaluator. Keeping it here means adding a metric
#: is one entry rather than an edit in three places.
METRIC_FUNCTIONS: Final = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "ndcg": ndcg_at_k,
    "map": average_precision_at_k,
    "mrr": reciprocal_rank_at_k,
    "hit_rate": hit_rate_at_k,
}


def metric_name(metric: str, k: int) -> str:
    """Stable flat metric key, e.g. ``ndcg@20``."""
    return f"{metric}@{k}"


__all__ = [
    "METRIC_FUNCTIONS",
    "RANKING_METRICS",
    "average_precision_at_k",
    "dcg_at_k",
    "hit_rate_at_k",
    "metric_name",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
    "validate_k",
]
