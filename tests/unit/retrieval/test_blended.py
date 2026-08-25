"""The blended retriever.

Two Phase 4 deliverables reduce to this class -- the popularity+BPR hybrid and
the aggregation experiments -- so the properties checked here are the ones that
make a *fused* result trustworthy: that it goes through the same interface as a
single retriever, that provenance survives, and that over-retrieval keeps the
output length independent of how much the sources happen to agree.

The sources are stubs rather than trained models. A real model would make these
tests slow and would test the model, not the blending.
"""

from __future__ import annotations

from typing import Any

import pytest

from omnirank.core.exceptions import DataError, ModelNotFittedError
from omnirank.models.base import Candidate
from omnirank.retrieval.aggregation import build_aggregator
from omnirank.retrieval.blended import (
    DEFAULT_OVER_RETRIEVAL_FACTOR,
    BlendableSource,
    BlendedRetriever,
)


class StubSource:
    """A fitted source returning a fixed, per-user ranked list."""

    def __init__(self, name: str, lists: dict[str, list[str]], *, fitted: bool = True) -> None:
        self.name = name
        self.lists = lists
        self._fitted = fitted
        self.requested_depths: list[int] = []

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def fit_item_catalogue(self) -> set[int]:
        return {int(item[1:]) for items in self.lists.values() for item in items}

    def recommend(
        self, user_id: str, k: int, context: dict[str, Any] | None = None
    ) -> list[Candidate]:
        self.requested_depths.append(k)
        items = self.lists.get(user_id, [])[:k]
        return [
            Candidate(
                item_id=item,
                score=float(len(items) - position),
                sources=(self.name,),
                source_scores={self.name: float(len(items) - position)},
            )
            for position, item in enumerate(items)
        ]

    def recommend_batch(
        self, user_ids: list[str], k: int, *, filter_seen: bool = True
    ) -> dict[str, list[str]]:
        self.requested_depths.append(k)
        return {user: self.lists.get(user, [])[:k] for user in user_ids}

    def score(self, user_id: str, item_ids: list[str]) -> list[float]:
        ranked = self.lists.get(user_id, [])
        return [
            float(len(ranked) - ranked.index(item)) if item in ranked else 0.0 for item in item_ids
        ]

    def metadata(self) -> dict[str, Any]:
        return {"model": self.name}


@pytest.fixture
def sources() -> dict[str, StubSource]:
    """Two sources overlapping on ``i2``."""
    return {
        "bpr": StubSource("bpr", {"u1": [f"i{n}" for n in (1, 2, 3, 4, 5, 6)]}),
        "popularity": StubSource("popularity", {"u1": [f"i{n}" for n in (7, 8, 2, 9, 10, 11)]}),
    }


@pytest.fixture
def blend(sources: dict[str, StubSource]) -> BlendedRetriever:
    return BlendedRetriever(sources, build_aggregator("reciprocal_rank_fusion"), name="hybrid")


class TestConstruction:
    def test_stub_satisfies_the_source_protocol(self, sources: dict[str, StubSource]) -> None:
        assert isinstance(sources["bpr"], BlendableSource)

    def test_rejects_no_sources(self) -> None:
        with pytest.raises(DataError):
            BlendedRetriever({}, build_aggregator("reciprocal_rank_fusion"))

    def test_rejects_an_unfitted_source(self) -> None:
        """Blending an unfitted model would produce confident noise, not an error."""
        with pytest.raises(ModelNotFittedError):
            BlendedRetriever(
                {"bpr": StubSource("bpr", {}, fitted=False)},
                build_aggregator("reciprocal_rank_fusion"),
            )

    def test_rejects_a_zero_over_retrieval_factor(self, sources: dict[str, StubSource]) -> None:
        with pytest.raises(DataError):
            BlendedRetriever(
                sources, build_aggregator("reciprocal_rank_fusion"), over_retrieval_factor=0
            )

    def test_takes_the_name_it_is_given(self, blend: BlendedRetriever) -> None:
        assert blend.name == "hybrid"

    def test_is_fitted_on_construction(self, blend: BlendedRetriever) -> None:
        assert blend.is_fitted


class TestOverRetrieval:
    def test_asks_each_source_for_more_than_k(
        self, blend: BlendedRetriever, sources: dict[str, StubSource]
    ) -> None:
        """Fusing top-k lists and truncating to k under-fills whenever sources agree."""
        blend.recommend("u1", 2)
        assert sources["bpr"].requested_depths == [2 * DEFAULT_OVER_RETRIEVAL_FACTOR]

    def test_output_length_survives_complete_source_agreement(self) -> None:
        """Identical sources are the worst case for deduplication."""
        shared = [f"i{n}" for n in range(10)]
        blend = BlendedRetriever(
            {
                "a": StubSource("a", {"u1": shared}),
                "b": StubSource("b", {"u1": shared}),
            },
            build_aggregator("reciprocal_rank_fusion"),
        )
        assert len(blend.recommend("u1", 5)) == 5


class TestRecommendation:
    def test_agreed_item_is_promoted(self, blend: BlendedRetriever) -> None:
        """``i2`` is the only item both sources return."""
        assert blend.recommend("u1", 1)[0].item_id == "i2"

    def test_provenance_records_both_sources(self, blend: BlendedRetriever) -> None:
        fused = next(c for c in blend.recommend("u1", 5) if c.item_id == "i2")
        assert set(fused.sources) == {"bpr", "popularity"}
        assert set(fused.source_scores) == {"bpr", "popularity"}

    def test_batch_matches_single_user_ordering(self, blend: BlendedRetriever) -> None:
        """The two paths build candidates differently; they must still agree."""
        assert blend.recommend_batch(["u1"], 5)["u1"] == [
            candidate.item_id for candidate in blend.recommend("u1", 5)
        ]

    def test_unknown_user_yields_nothing(self, blend: BlendedRetriever) -> None:
        assert blend.recommend("stranger", 5) == []

    def test_aggregate_for_exposes_the_audit_trail(self, blend: BlendedRetriever) -> None:
        """``contributions`` is what answers 'why did recall drop'.

        It counts what each source put into the fused pool, which is larger than
        the emitted list: over-retrieval gathers depth that truncation to ``k``
        then discards. Counting only survivors would hide a source that
        contributed plenty but lost every tie-break.
        """
        result = blend.aggregate_for("u1", 5)
        assert len(result.candidates) == 5
        # i2 is emitted and nominated by both, so both are credited for it.
        assert result.contributions["bpr"] >= 1
        assert result.contributions["popularity"] >= 1
        assert sum(result.contributions.values()) >= len(result.candidates)

    def test_a_silent_source_is_reported_as_degraded(self) -> None:
        blend = BlendedRetriever(
            {
                "bpr": StubSource("bpr", {"u1": ["i1", "i2"]}),
                "sasrec": StubSource("sasrec", {}),
            },
            build_aggregator("reciprocal_rank_fusion"),
        )
        assert blend.aggregate_for("u1", 5).degraded_sources == ("sasrec",)


class TestCatalogue:
    def test_is_the_union_of_the_sources(self, blend: BlendedRetriever) -> None:
        """An item any source can retrieve is reachable through the blend."""
        assert blend.fit_item_catalogue == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}


class TestUnsupportedOperations:
    def test_fitting_is_refused(self, blend: BlendedRetriever) -> None:
        """There is no coherent meaning for 'training a fusion' of four objectives."""
        with pytest.raises(DataError):
            blend.fit(None)

    def test_saving_is_refused(self, blend: BlendedRetriever, tmp_path) -> None:
        """Persisting a blend would duplicate every source and let the copy drift."""
        with pytest.raises(DataError):
            blend.save(tmp_path / "blend")


class TestMetadata:
    def test_records_what_was_blended_and_how(self, blend: BlendedRetriever) -> None:
        metadata = blend.metadata()
        assert metadata["sources"] == ["bpr", "popularity"]
        assert metadata["aggregator"] == "ReciprocalRankFusionAggregator"
        assert metadata["over_retrieval_factor"] == DEFAULT_OVER_RETRIEVAL_FACTOR
        assert set(metadata["source_metadata"]) == {"bpr", "popularity"}


class TestBatchScoreSemantics:
    """The batch path reconstructs scores from rank; that has limits worth pinning."""

    @pytest.fixture
    def sources_with_spread(self) -> dict[str, StubSource]:
        """Two sources whose raw scores differ wildly in spread."""
        return {
            "wide": StubSource("wide", {"u1": ["a", "b", "c"]}),
            "narrow": StubSource("narrow", {"u1": ["d", "b", "e"]}),
        }

    def test_rank_based_strategies_agree_between_paths(
        self, sources_with_spread: dict[str, StubSource]
    ) -> None:
        """RRF reads only ordering, which the placeholder scores preserve."""
        blend = BlendedRetriever(sources_with_spread, build_aggregator("reciprocal_rank_fusion"))
        assert blend.recommend_batch(["u1"], 3)["u1"] == [
            candidate.item_id for candidate in blend.recommend("u1", 3)
        ]

    def test_rank_percentile_union_agrees_between_paths(
        self, sources_with_spread: dict[str, StubSource]
    ) -> None:
        """rank_percentile discards magnitude, so it is safe on the batch path."""
        blend = BlendedRetriever(
            sources_with_spread,
            build_aggregator("normalized_score_union", normalization="rank_percentile"),
        )
        assert blend.recommend_batch(["u1"], 3)["u1"] == [
            candidate.item_id for candidate in blend.recommend("u1", 3)
        ]

    def test_round_robin_agrees_between_paths(
        self, sources_with_spread: dict[str, StubSource]
    ) -> None:
        blend = BlendedRetriever(sources_with_spread, build_aggregator("weighted_round_robin"))
        assert blend.recommend_batch(["u1"], 3)["u1"] == [
            candidate.item_id for candidate in blend.recommend("u1", 3)
        ]
