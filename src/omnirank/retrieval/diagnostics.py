"""Diagnostics for a multi-source retrieval stage.

Accuracy metrics measure the *ranked* list. They cannot answer the two questions
that actually decide whether a retrieval stage is worth its complexity:

**Can the candidates contain the answer at all?** Recall@20 conflates two
different failures -- the target was never retrieved, and the target was
retrieved but ranked below 20. Only the first is retrieval's fault, and no
amount of ranker work in Phase 6 can fix it. Candidate recall isolates it and
sets the ceiling every downstream stage inherits.

**Are the sources actually different?** Fusing four generators that return
almost the same list costs four times the compute for nearly one generator's
coverage. Overlap says whether the ensemble is an ensemble.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from omnirank.core.exceptions import DataError
from omnirank.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CandidateRecall:
    """How often the candidate pool contains the target, at a given depth."""

    depth: int
    users_evaluated: int
    users_with_target: int
    #: Users whose target is not reachable by any source at all, counted
    #: separately: a cold target is a legitimate miss, not a retrieval failure.
    users_with_unreachable_target: int

    @property
    def recall(self) -> float:
        """Fraction of evaluated users whose target appeared in the pool."""
        return self.users_with_target / self.users_evaluated if self.users_evaluated else 0.0

    @property
    def reachable_recall(self) -> float:
        """Recall over users whose target was reachable in principle.

        The ceiling a ranker could reach if it were perfect. Distinguishing this
        from :attr:`recall` matters because they are improved by different work:
        the gap to `recall` is closed by content features, the gap to
        `reachable_recall` by better retrieval.
        """
        reachable = self.users_evaluated - self.users_with_unreachable_target
        return self.users_with_target / reachable if reachable else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Report-ready payload."""
        return {
            "depth": self.depth,
            "users_evaluated": self.users_evaluated,
            "users_with_target": self.users_with_target,
            "users_with_unreachable_target": self.users_with_unreachable_target,
            "candidate_recall": round(self.recall, 6),
            "reachable_candidate_recall": round(self.reachable_recall, 6),
        }


@dataclass(frozen=True, slots=True)
class SourceOverlap:
    """Pairwise and aggregate agreement between generators."""

    depth: int
    #: Mean Jaccard index per unordered source pair.
    pairwise_jaccard: dict[str, float]
    #: Mean number of sources that proposed each retrieved item.
    mean_sources_per_item: float
    #: Per source, the mean fraction of its items no other source proposed.
    unique_contribution: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Report-ready payload."""
        return {
            "depth": self.depth,
            "pairwise_jaccard": {k: round(v, 6) for k, v in sorted(self.pairwise_jaccard.items())},
            "mean_sources_per_item": round(self.mean_sources_per_item, 6),
            "unique_contribution": {
                k: round(v, 6) for k, v in sorted(self.unique_contribution.items())
            },
        }


def candidate_recall(
    candidates_by_user: Mapping[str, Sequence[str]],
    targets_by_user: Mapping[str, set[str]],
    *,
    depth: int,
    reachable_items: set[str] | None = None,
) -> CandidateRecall:
    """Measure how often the candidate pool contains a user's target.

    Args:
        candidates_by_user: Retrieved item ids per user, in rank order.
        targets_by_user: Held-out target ids per user.
        depth: Truncate each pool to this many candidates before checking.
        reachable_items: Items any source could return. Targets outside it are
            counted as unreachable rather than as retrieval misses.

    Raises:
        DataError: ``depth`` is not positive.
    """
    if depth < 1:
        raise DataError("Candidate recall depth must be positive", depth=depth)

    evaluated = hits = unreachable = 0
    for user, targets in targets_by_user.items():
        if not targets:
            continue
        evaluated += 1
        if reachable_items is not None and not (targets & reachable_items):
            unreachable += 1
        if set(candidates_by_user.get(user, ())[:depth]) & targets:
            hits += 1

    result = CandidateRecall(
        depth=depth,
        users_evaluated=evaluated,
        users_with_target=hits,
        users_with_unreachable_target=unreachable,
    )
    logger.info("diagnostics.candidate_recall", **result.to_dict())
    return result


def source_overlap(
    per_source_by_user: Mapping[str, Mapping[str, Sequence[str]]], *, depth: int
) -> SourceOverlap:
    """Measure how much the generators duplicate one another.

    Args:
        per_source_by_user: ``{source: {user: [item, ...]}}`` in rank order.
        depth: Truncate each source's list to this many items.

    Raises:
        DataError: Fewer than two sources, or a non-positive depth.
    """
    if depth < 1:
        raise DataError("Overlap depth must be positive", depth=depth)
    sources = sorted(per_source_by_user)
    if len(sources) < 2:
        raise DataError("Overlap needs at least two sources", sources=sources)

    users = sorted({user for lists in per_source_by_user.values() for user in lists})
    pair_totals: dict[str, float] = {}
    unique_totals: dict[str, float] = dict.fromkeys(sources, 0.0)
    sources_per_item_total = 0.0
    counted_users = 0

    for user in users:
        sets = {source: set(per_source_by_user[source].get(user, ())[:depth]) for source in sources}
        if not any(sets.values()):
            continue
        counted_users += 1

        for left, right in combinations(sources, 2):
            union = sets[left] | sets[right]
            key = f"{left}|{right}"
            # An empty union is agreement by vacancy, not similarity; scoring it
            # 1.0 would make two silent sources look identical to two that
            # returned the same list.
            jaccard = len(sets[left] & sets[right]) / len(union) if union else 0.0
            pair_totals[key] = pair_totals.get(key, 0.0) + jaccard

        item_counts: dict[str, int] = {}
        for items in sets.values():
            for item in items:
                item_counts[item] = item_counts.get(item, 0) + 1
        if item_counts:
            sources_per_item_total += sum(item_counts.values()) / len(item_counts)

        for source, items in sets.items():
            if items:
                unique = sum(1 for item in items if item_counts[item] == 1)
                unique_totals[source] += unique / len(items)

    divisor = max(counted_users, 1)
    result = SourceOverlap(
        depth=depth,
        pairwise_jaccard={key: total / divisor for key, total in pair_totals.items()},
        mean_sources_per_item=sources_per_item_total / divisor,
        unique_contribution={source: total / divisor for source, total in unique_totals.items()},
    )
    logger.info("diagnostics.source_overlap", **result.to_dict())
    return result


__all__ = [
    "CandidateRecall",
    "SourceOverlap",
    "candidate_recall",
    "source_overlap",
]
