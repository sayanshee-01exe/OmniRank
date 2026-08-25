"""Candidate aggregation.

The fused scores are checked against arithmetic done by hand. An aggregator
that is subtly wrong still returns a plausible-looking ranked list -- there is
no exception, no NaN, and no failing metric to notice -- so "the output looks
reasonable" is not evidence of anything here.

Determinism gets equal weight. Retrieval feeds a ranker that will be trained on
its output; if aggregation order depended on dict insertion order, every
downstream experiment would be irreproducible for a reason nobody would think
to look for.
"""

from __future__ import annotations

import pytest

from omnirank.core.exceptions import DataError
from omnirank.models.base import Candidate
from omnirank.retrieval.aggregation import (
    DEFAULT_RRF_CONSTANT,
    NormalizedScoreUnionAggregator,
    ReciprocalRankFusionAggregator,
    SourceRanks,
    WeightedRoundRobinAggregator,
    build_aggregator,
)
from omnirank.retrieval.base import CandidateAggregator


def candidates(source: str, *items: tuple[str, float]) -> list[Candidate]:
    """Build one source's ranked list."""
    return [
        Candidate(item_id=item, score=score, sources=(source,), source_scores={source: score})
        for item, score in items
    ]


@pytest.fixture
def two_sources() -> dict[str, list[Candidate]]:
    """Two sources that overlap on exactly one item, ``b``."""
    return {
        "bpr": candidates("bpr", ("a", 0.9), ("b", 0.8), ("c", 0.7)),
        "popularity": candidates("popularity", ("d", 500.0), ("e", 400.0), ("b", 300.0)),
    }


class TestSourceRanks:
    def test_ranks_are_one_based(self, two_sources: dict[str, list[Candidate]]) -> None:
        ranks = SourceRanks.from_sources(two_sources)
        assert ranks.rank_of("bpr", "a") == 1
        assert ranks.rank_of("bpr", "c") == 3

    def test_absent_item_has_no_rank(self, two_sources: dict[str, list[Candidate]]) -> None:
        assert SourceRanks.from_sources(two_sources).rank_of("bpr", "d") is None


class TestReciprocalRankFusion:
    def test_fused_score_matches_hand_arithmetic(
        self, two_sources: dict[str, list[Candidate]]
    ) -> None:
        """``b`` is rank 2 in bpr and rank 3 in popularity."""
        result = ReciprocalRankFusionAggregator().aggregate(two_sources, limit=10)
        scores = {c.item_id: c.score for c in result.candidates}
        expected = 1 / (DEFAULT_RRF_CONSTANT + 2) + 1 / (DEFAULT_RRF_CONSTANT + 3)
        assert scores["b"] == pytest.approx(expected)
        assert scores["a"] == pytest.approx(1 / (DEFAULT_RRF_CONSTANT + 1))

    def test_agreement_between_sources_beats_a_single_high_rank(
        self, two_sources: dict[str, list[Candidate]]
    ) -> None:
        """The whole point of fusion: two mid-ranked votes outweigh one top vote."""
        result = ReciprocalRankFusionAggregator().aggregate(two_sources, limit=10)
        assert result.candidates[0].item_id == "b"

    def test_weights_scale_each_source_contribution(
        self, two_sources: dict[str, list[Candidate]]
    ) -> None:
        result = ReciprocalRankFusionAggregator({"bpr": 2.0, "popularity": 0.0}).aggregate(
            two_sources, limit=10
        )
        scores = {c.item_id: c.score for c in result.candidates}
        assert scores["a"] == pytest.approx(2.0 / (DEFAULT_RRF_CONSTANT + 1))
        # A zero-weighted source still contributes the item, at zero score.
        assert scores["d"] == pytest.approx(0.0)

    def test_a_smaller_constant_sharpens_the_top_ranks(
        self, two_sources: dict[str, list[Candidate]]
    ) -> None:
        sharp = ReciprocalRankFusionAggregator(rrf_constant=1.0).aggregate(two_sources, limit=10)
        # b is rank 2 in bpr and rank 3 in popularity: 1/(1+2) + 1/(1+3).
        assert sharp.candidates[0].score == pytest.approx(1 / 3 + 1 / 4)

    def test_provenance_survives_fusion(self, two_sources: dict[str, list[Candidate]]) -> None:
        """The fused score replaces the per-source ones, which must still be recoverable."""
        result = ReciprocalRankFusionAggregator().aggregate(two_sources, limit=10)
        fused = next(c for c in result.candidates if c.item_id == "b")
        assert set(fused.sources) == {"bpr", "popularity"}
        assert fused.source_scores == {"bpr": 0.8, "popularity": 300.0}

    def test_rejects_a_non_positive_constant(self) -> None:
        with pytest.raises(DataError):
            ReciprocalRankFusionAggregator(rrf_constant=0.0)

    def test_rejects_negative_weights(self) -> None:
        with pytest.raises(DataError):
            ReciprocalRankFusionAggregator({"bpr": -1.0})


class TestWeightedRoundRobin:
    def test_higher_weight_source_leads(self, two_sources: dict[str, list[Candidate]]) -> None:
        result = WeightedRoundRobinAggregator({"bpr": 3.0, "popularity": 1.0}).aggregate(
            two_sources, limit=4
        )
        assert result.candidates[0].item_id == "a"

    def test_deduplicates_across_sources(self, two_sources: dict[str, list[Candidate]]) -> None:
        result = WeightedRoundRobinAggregator().aggregate(two_sources, limit=10)
        emitted = [c.item_id for c in result.candidates]
        assert len(emitted) == len(set(emitted)) == 5  # 6 candidates, b appears twice

    def test_respects_the_limit(self, two_sources: dict[str, list[Candidate]]) -> None:
        result = WeightedRoundRobinAggregator().aggregate(two_sources, limit=3)
        assert len(result.candidates) == 3


class TestNormalizedScoreUnion:
    def test_min_max_puts_every_source_on_the_same_scale(
        self, two_sources: dict[str, list[Candidate]]
    ) -> None:
        """Raw scores differ by three orders of magnitude; normalisation fixes that."""
        result = NormalizedScoreUnionAggregator(normalization="min_max").aggregate(
            two_sources, limit=10
        )
        top = result.candidates[0]
        # Both sources' rank-1 items normalise to 1.0; the tie breaks on item id.
        assert top.item_id in {"a", "d"}
        assert top.score == pytest.approx(1.0)

    def test_rank_percentile_ignores_score_magnitude(self) -> None:
        """Two sources with wildly different scales must contribute equally."""
        sources = {
            "tiny": candidates("tiny", ("a", 0.001), ("b", 0.0005)),
            "huge": candidates("huge", ("c", 9_000_000.0), ("d", 1.0)),
        }
        result = NormalizedScoreUnionAggregator(normalization="rank_percentile").aggregate(
            sources, limit=4
        )
        scores = {c.item_id: c.score for c in result.candidates}
        assert scores["a"] == pytest.approx(scores["c"])

    def test_identical_scores_do_not_divide_by_zero(self) -> None:
        sources = {"flat": candidates("flat", ("a", 5.0), ("b", 5.0), ("c", 5.0))}
        result = NormalizedScoreUnionAggregator(normalization="min_max").aggregate(sources, limit=3)
        assert all(c.score == c.score for c in result.candidates)  # no NaN

    def test_rejects_an_unknown_normalization(self) -> None:
        with pytest.raises(DataError):
            NormalizedScoreUnionAggregator(normalization="not-a-normalization")


class TestSharedBehaviour:
    """Properties every aggregator must satisfy."""

    @pytest.fixture(
        params=[
            "weighted_round_robin",
            "reciprocal_rank_fusion",
            "normalized_score_union",
        ]
    )
    def aggregator(self, request: pytest.FixtureRequest) -> CandidateAggregator:
        return build_aggregator(request.param)

    def test_is_deterministic_across_repeated_calls(
        self, aggregator, two_sources: dict[str, list[Candidate]]
    ) -> None:
        first = [c.item_id for c in aggregator.aggregate(two_sources, limit=5).candidates]
        second = [c.item_id for c in aggregator.aggregate(two_sources, limit=5).candidates]
        assert first == second

    def test_does_not_depend_on_source_insertion_order(
        self, aggregator, two_sources: dict[str, list[Candidate]]
    ) -> None:
        """A dict-order dependency would make every downstream experiment irreproducible."""
        forward = aggregator.aggregate(two_sources, limit=5)
        reversed_sources = dict(reversed(list(two_sources.items())))
        backward = aggregator.aggregate(reversed_sources, limit=5)
        assert [c.item_id for c in forward.candidates] == [c.item_id for c in backward.candidates]

    def test_never_emits_duplicates(
        self, aggregator, two_sources: dict[str, list[Candidate]]
    ) -> None:
        emitted = [c.item_id for c in aggregator.aggregate(two_sources, limit=10).candidates]
        assert len(emitted) == len(set(emitted))

    def test_never_exceeds_the_limit(
        self, aggregator, two_sources: dict[str, list[Candidate]]
    ) -> None:
        assert len(aggregator.aggregate(two_sources, limit=2).candidates) == 2

    def test_empty_source_is_reported_as_degraded(self, aggregator) -> None:
        """A generator that silently stops producing is the failure this catches."""
        result = aggregator.aggregate({"bpr": candidates("bpr", ("a", 1.0)), "sasrec": []}, limit=5)
        assert result.degraded_sources == ("sasrec",)
        assert [c.item_id for c in result.candidates] == ["a"]

    def test_all_sources_empty_yields_an_empty_result(self, aggregator) -> None:
        result = aggregator.aggregate({"bpr": [], "sasrec": []}, limit=5)
        assert result.is_empty
        assert set(result.degraded_sources) == {"bpr", "sasrec"}

    def test_contributions_are_counted_after_deduplication(
        self, aggregator, two_sources: dict[str, list[Candidate]]
    ) -> None:
        """``contributions`` answers 'why did recall drop' when a source degrades.

        Counted over what was emitted, not what was offered: a source whose
        nominations all lost their tie-breaks contributed nothing to the list
        the ranker saw, and the diagnostic has to say so.
        """
        result = aggregator.aggregate(two_sources, limit=10)
        emitted = {c.item_id for c in result.candidates}
        for source, items in two_sources.items():
            expected = len({c.item_id for c in items} & emitted)
            assert result.contributions[source] == expected

    def test_contributions_ignore_truncated_candidates(self, aggregator) -> None:
        """A source that offers many candidates but places none is credited zero."""
        result = aggregator.aggregate(
            {
                "loud": candidates("loud", *[(f"x{n}", 1.0 - n / 100) for n in range(20)]),
                "quiet": candidates("quiet", ("y1", 0.5)),
            },
            limit=1,
        )
        assert sum(result.contributions.values()) == 1

    def test_rejects_a_non_positive_limit(
        self, aggregator, two_sources: dict[str, list[Candidate]]
    ) -> None:
        with pytest.raises(DataError):
            aggregator.aggregate(two_sources, limit=0)

    def test_rejects_no_sources(self, aggregator) -> None:
        with pytest.raises(DataError):
            aggregator.aggregate({}, limit=5)


class TestBuildAggregator:
    def test_rejects_an_unknown_strategy(self) -> None:
        with pytest.raises(DataError):
            build_aggregator("hand-wavy-fusion")

    def test_passes_the_rrf_constant_through(self) -> None:
        built = build_aggregator("reciprocal_rank_fusion", rrf_constant=17.0)
        assert built.rrf_constant == 17.0
