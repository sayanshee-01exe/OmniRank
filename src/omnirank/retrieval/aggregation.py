"""Concrete candidate aggregators - component 11.

Several generators propose candidates; one list has to come out. The hard part is
not merging but **comparison**: a popularity score is a time-decayed interaction
count in the tens, a BPR score is a dot product that can be negative, and a
SASRec score is a logit. Adding them directly is meaningless arithmetic that
happens to run.

So nothing here compares raw scores across sources. Both aggregators work on
**rank**, which is the one quantity every generator produces on a common scale:

* :class:`WeightedRoundRobinAggregator` - interleave by source weight.
* :class:`ReciprocalRankFusionAggregator` - sum ``w_s / (c + rank_s(i))``.

A score-based union is available only via
:class:`NormalizedScoreUnionAggregator`, which requires each source to be
normalised independently first, and says so in its own docstring.

Every aggregator preserves each candidate's contributing sources and its raw
per-source scores, so a merged list stays explicable: "this item surfaced because
SASRec ranked it 3rd and popularity ranked it 40th" is recoverable afterwards.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from omnirank.core.exceptions import DataError
from omnirank.core.logging import get_logger
from omnirank.models.base import Candidate
from omnirank.retrieval.base import AggregationResult, CandidateAggregator

logger = get_logger(__name__)

#: Standard RRF constant. Damps the influence of top ranks so a single source
#: cannot dominate on its first result alone; 60 is the value from the original
#: Cormack et al. formulation and is a reasonable default at these list lengths.
DEFAULT_RRF_CONSTANT: Final = 60.0

WEIGHTED_ROUND_ROBIN: Final = "weighted_round_robin"
RECIPROCAL_RANK_FUSION: Final = "reciprocal_rank_fusion"
NORMALIZED_SCORE_UNION: Final = "normalized_score_union"


def _validate(per_source: Mapping[str, Sequence[Candidate]], limit: int) -> None:
    """Reject inputs no aggregator can act on."""
    if limit < 1:
        raise DataError("Aggregation limit must be positive", limit=limit)
    if not per_source:
        # Distinct from every source returning nothing, which is a legitimate
        # runtime state reported through `degraded_sources`. No sources *at all*
        # means the pipeline was wired with no generators, and returning an
        # empty result would disguise that as an ordinary empty response.
        raise DataError("Aggregation needs at least one source")
    for source, candidates in per_source.items():
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.item_id in seen:
                raise DataError(
                    "A source produced the same item twice. Deduplicating here "
                    "would hide a generator bug and distort its rank positions.",
                    source=source,
                    item_id=candidate.item_id,
                )
            seen.add(candidate.item_id)


def _count_contributions(
    per_source: Mapping[str, Sequence[Candidate]], emitted: Sequence[Candidate]
) -> dict[str, int]:
    """Credit each source for every emitted candidate it nominated.

    Counted over what was *emitted*, not over what was offered. The number is
    there to answer "why did recall drop", and a source that nominated a
    hundred candidates which all lost their tie-breaks contributed nothing to
    the list the ranker actually saw.

    A source is credited for a shared item, so the total can exceed the number
    of candidates. That excess is the overlap between sources, which is itself
    worth reading: sources that agree completely add no coverage.
    """
    counts = dict.fromkeys(per_source, 0)
    for candidate in emitted:
        for source in candidate.sources:
            if source in counts:
                counts[source] += 1
    return counts


def _merge(existing: Candidate | None, incoming: Candidate, source: str) -> Candidate:
    """Combine two nominations of the same item, preserving both provenances."""
    tagged = Candidate(
        item_id=incoming.item_id,
        score=incoming.score,
        sources=incoming.sources or (source,),
        source_scores=incoming.source_scores or {source: incoming.score},
    )
    return tagged if existing is None else existing.merged_with(tagged)


@dataclass(slots=True)
class SourceRanks:
    """Per-source rank lookup for one aggregation call.

    Ranks are 1-based and reflect the order the source returned, which is the
    only cross-source-comparable quantity available.
    """

    ranks: dict[str, dict[str, int]]

    def rank_of(self, source: str, item_id: str) -> int | None:
        """1-based rank of an item within one source, or None if absent."""
        return self.ranks.get(source, {}).get(item_id)

    @classmethod
    def from_sources(cls, per_source: Mapping[str, Sequence[Candidate]]) -> SourceRanks:
        """Build the lookup."""
        return cls(
            {
                source: {candidate.item_id: rank for rank, candidate in enumerate(items, start=1)}
                for source, items in per_source.items()
            }
        )


class WeightedRoundRobinAggregator(CandidateAggregator):
    """Interleave sources, taking more turns from higher-weighted ones.

    Guarantees every source representation in the final list, which matters when
    one generator is far stronger on aggregate but the others cover populations
    it cannot reach - exactly the popularity/BPR situation Phase 3 measured,
    where popularity wins overall and loses entirely on the long tail.

    Turn order is computed once from the weights and then cycled, so the result
    is deterministic and independent of dict iteration order.
    """

    def __init__(self, source_weights: Mapping[str, float] | None = None) -> None:
        self.source_weights = dict(source_weights or {})
        if any(weight < 0 for weight in self.source_weights.values()):
            raise DataError("Source weights must be non-negative", weights=self.source_weights)

    def aggregate(
        self, per_source: dict[str, Sequence[Candidate]], *, limit: int
    ) -> AggregationResult:
        """Interleave by weight, deduplicating as it goes."""
        _validate(per_source, limit)
        active = {source: list(items) for source, items in per_source.items() if items}
        degraded = tuple(sorted(source for source, items in per_source.items() if not items))
        if not active:
            return AggregationResult(candidates=(), contributions={}, degraded_sources=degraded)

        # Turns per round, proportional to weight. Sorted so the order depends on
        # the weights and names, never on insertion order.
        order = sorted(active, key=lambda source: (-self.source_weights.get(source, 1.0), source))
        turns = {
            source: max(1, round(self.source_weights.get(source, 1.0) * 2)) for source in order
        }

        merged: dict[str, Candidate] = {}
        cursors = dict.fromkeys(active, 0)

        while len(merged) < limit and any(
            cursors[source] < len(active[source]) for source in order
        ):
            progressed = False
            for source in order:
                for _ in range(turns[source]):
                    if len(merged) >= limit:
                        break
                    index = cursors[source]
                    if index >= len(active[source]):
                        break
                    cursors[source] = index + 1
                    progressed = True
                    candidate = active[source][index]
                    merged[candidate.item_id] = _merge(
                        merged.get(candidate.item_id), candidate, source
                    )
            if not progressed:
                break

        # Insertion order is the interleave order, which is the intended ranking.
        candidates = tuple(merged.values())[:limit]
        contributions = _count_contributions(per_source, candidates)
        logger.debug(
            "aggregation.weighted_round_robin",
            sources=len(active),
            emitted=len(candidates),
            contributions=contributions,
        )
        return AggregationResult(
            candidates=candidates, contributions=contributions, degraded_sources=degraded
        )


class ReciprocalRankFusionAggregator(CandidateAggregator):
    """Fuse sources by summing weighted reciprocal ranks.

    ``RRF(i) = sum_s w_s / (c + rank_s(i))``

    Rank-based, so it never compares a popularity count with a dot product. An
    item ranked highly by two sources beats one ranked highly by a single source,
    which is the property that makes fusion worth doing at all.

    ``c`` damps the top ranks: without it the first result of any source would
    dominate every sum.
    """

    def __init__(
        self,
        source_weights: Mapping[str, float] | None = None,
        *,
        rrf_constant: float = DEFAULT_RRF_CONSTANT,
    ) -> None:
        if rrf_constant <= 0:
            raise DataError("RRF constant must be positive", rrf_constant=rrf_constant)
        self.source_weights = dict(source_weights or {})
        if any(weight < 0 for weight in self.source_weights.values()):
            raise DataError("Source weights must be non-negative", weights=self.source_weights)
        self.rrf_constant = float(rrf_constant)

    def aggregate(
        self, per_source: dict[str, Sequence[Candidate]], *, limit: int
    ) -> AggregationResult:
        """Fuse by weighted reciprocal rank."""
        _validate(per_source, limit)
        degraded = tuple(sorted(source for source, items in per_source.items() if not items))

        fused: dict[str, float] = {}
        merged: dict[str, Candidate] = {}

        for source, items in per_source.items():
            weight = self.source_weights.get(source, 1.0)
            for rank, candidate in enumerate(items, start=1):
                fused[candidate.item_id] = fused.get(candidate.item_id, 0.0) + weight / (
                    self.rrf_constant + rank
                )
                merged[candidate.item_id] = _merge(merged.get(candidate.item_id), candidate, source)

        # Ties break on item id so two runs order identically.
        ordered = sorted(fused, key=lambda item_id: (-fused[item_id], item_id))[:limit]
        candidates = tuple(
            Candidate(
                item_id=item_id,
                # The fused score replaces the incomparable per-source ones; the
                # originals stay in `source_scores` for explanation.
                score=fused[item_id],
                sources=merged[item_id].sources,
                source_scores=merged[item_id].source_scores,
            )
            for item_id in ordered
        )
        contributions = _count_contributions(per_source, candidates)
        logger.debug(
            "aggregation.reciprocal_rank_fusion",
            sources=len(per_source),
            emitted=len(candidates),
            rrf_constant=self.rrf_constant,
            contributions=contributions,
        )
        return AggregationResult(
            candidates=candidates, contributions=contributions, degraded_sources=degraded
        )


class NormalizedScoreUnionAggregator(CandidateAggregator):
    """Union by score, after normalising each source independently.

    Only safe because normalisation happens **within** a source before anything
    is compared. Raw-score addition is never performed anywhere in this module.

    Args:
        source_weights: Per-source multipliers applied after normalisation.
        normalization: ``"min_max"`` (within the returned list), ``"z_score"``
            (zero-variance lists collapse to 0.0 rather than dividing by zero),
            or ``"rank_percentile"`` (ignores score magnitude entirely and is
            the safest choice when a source's scale is unknown).
    """

    _NORMALIZATIONS: Final = ("min_max", "z_score", "rank_percentile")

    def __init__(
        self,
        source_weights: Mapping[str, float] | None = None,
        *,
        normalization: str = "rank_percentile",
    ) -> None:
        if normalization not in self._NORMALIZATIONS:
            raise DataError(
                "Unknown normalization",
                normalization=normalization,
                available=list(self._NORMALIZATIONS),
            )
        self.source_weights = dict(source_weights or {})
        self.normalization = normalization

    def _normalize(self, candidates: Sequence[Candidate]) -> list[float]:
        """Map one source's scores onto a common [0, 1]-ish scale."""
        scores = [candidate.score for candidate in candidates]
        if not scores:
            return []
        if self.normalization == "rank_percentile":
            count = len(scores)
            return [1.0 - (rank / count) for rank in range(count)]
        if self.normalization == "min_max":
            low, high = min(scores), max(scores)
            spread = high - low
            # A constant list carries no ordering information; 0.5 says "no
            # preference" rather than inventing one.
            return [0.5] * len(scores) if spread == 0 else [(s - low) / spread for s in scores]
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        if variance == 0:
            return [0.0] * len(scores)
        deviation = math.sqrt(variance)
        return [(s - mean) / deviation for s in scores]

    def aggregate(
        self, per_source: dict[str, Sequence[Candidate]], *, limit: int
    ) -> AggregationResult:
        """Union normalised, weighted scores."""
        _validate(per_source, limit)
        degraded = tuple(sorted(source for source, items in per_source.items() if not items))

        totals: dict[str, float] = {}
        merged: dict[str, Candidate] = {}

        for source, items in per_source.items():
            weight = self.source_weights.get(source, 1.0)
            for candidate, normalized in zip(items, self._normalize(items), strict=True):
                totals[candidate.item_id] = totals.get(candidate.item_id, 0.0) + weight * normalized
                merged[candidate.item_id] = _merge(merged.get(candidate.item_id), candidate, source)

        ordered = sorted(totals, key=lambda item_id: (-totals[item_id], item_id))[:limit]
        candidates = tuple(
            Candidate(
                item_id=item_id,
                score=totals[item_id],
                sources=merged[item_id].sources,
                source_scores=merged[item_id].source_scores,
            )
            for item_id in ordered
        )
        contributions = _count_contributions(per_source, candidates)
        return AggregationResult(
            candidates=candidates, contributions=contributions, degraded_sources=degraded
        )


def build_aggregator(
    strategy: str,
    *,
    source_weights: Mapping[str, float] | None = None,
    rrf_constant: float = DEFAULT_RRF_CONSTANT,
    normalization: str = "rank_percentile",
) -> CandidateAggregator:
    """Construct an aggregator by strategy name.

    Raises:
        DataError: Unknown strategy.
    """
    if strategy == WEIGHTED_ROUND_ROBIN:
        return WeightedRoundRobinAggregator(source_weights)
    if strategy == RECIPROCAL_RANK_FUSION:
        return ReciprocalRankFusionAggregator(source_weights, rrf_constant=rrf_constant)
    if strategy == NORMALIZED_SCORE_UNION:
        return NormalizedScoreUnionAggregator(source_weights, normalization=normalization)
    raise DataError(
        "Unknown aggregation strategy",
        strategy=strategy,
        available=[WEIGHTED_ROUND_ROBIN, RECIPROCAL_RANK_FUSION, NORMALIZED_SCORE_UNION],
    )


__all__: list[str] = [
    "DEFAULT_RRF_CONSTANT",
    "NORMALIZED_SCORE_UNION",
    "RECIPROCAL_RANK_FUSION",
    "WEIGHTED_ROUND_ROBIN",
    "NormalizedScoreUnionAggregator",
    "ReciprocalRankFusionAggregator",
    "SourceRanks",
    "WeightedRoundRobinAggregator",
    "build_aggregator",
]
