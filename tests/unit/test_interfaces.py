"""Model, retrieval, ranking, reranking, and evaluation contracts.

These tests exercise the interfaces the way Phase 2+ implementations will:
a minimal in-test implementation is written against each ABC, proving the
contract is actually implementable and that the shared behaviour (the
fitted-state guard, candidate merging) works. Nothing here trains anything.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import pytest

from omnirank.core.exceptions import DataError, ModelNotFittedError
from omnirank.evaluation.base import Evaluator, GroundTruth
from omnirank.features.base import UserSequence
from omnirank.models.base import Candidate, CandidateGenerator, RankedItem, Ranker
from omnirank.ranking.base import FeatureBatch, FeatureRow
from omnirank.retrieval.base import AggregationResult


class TestAbstractness:
    """The interfaces must not be instantiable, or half-built models slip through."""

    @pytest.mark.parametrize("interface", [CandidateGenerator, Ranker])
    def test_cannot_instantiate(self, interface):
        with pytest.raises(TypeError):
            interface()

    def test_partial_implementation_is_rejected(self):
        class Incomplete(CandidateGenerator):
            def fit(self, data: Any) -> None:
                self._fitted = True

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


class MinimalGenerator(CandidateGenerator):
    """Smallest thing that satisfies the contract. Not a recommender."""

    name = "minimal"

    def __init__(self, items: tuple[str, ...] = ()) -> None:
        super().__init__()
        self._items = items

    def fit(self, data: Any) -> None:
        self._items = tuple(data)
        self._fitted = True

    def recommend(
        self, user_id: str, k: int, context: dict[str, Any] | None = None
    ) -> list[Candidate]:
        self.ensure_fitted()
        return [
            Candidate(item_id=item, score=1.0 / (rank + 1), sources=(self.name,))
            for rank, item in enumerate(self._items[:k])
        ]

    def score(self, user_id: str, item_ids: list[str]) -> list[float]:
        self.ensure_fitted()
        return [1.0 if item in self._items else 0.0 for item in item_ids]

    def save(self, path: str | Path) -> None:
        Path(path).write_text("\n".join(self._items))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        instance = cls(tuple(Path(path).read_text().splitlines()))
        instance._fitted = True
        return instance


class TestCandidateGenerator:
    def test_unfitted_generator_refuses_to_recommend(self):
        with pytest.raises(ModelNotFittedError):
            MinimalGenerator().recommend("u1", 5)

    def test_unfitted_generator_refuses_to_score(self):
        with pytest.raises(ModelNotFittedError):
            MinimalGenerator().score("u1", ["i1"])

    def test_fit_flips_the_state(self):
        generator = MinimalGenerator()
        assert generator.is_fitted is False
        generator.fit(["a", "b"])
        assert generator.is_fitted is True

    def test_recommend_respects_k(self):
        generator = MinimalGenerator()
        generator.fit(["a", "b", "c"])
        assert len(generator.recommend("u1", 2)) == 2

    def test_recommend_may_return_fewer_than_k(self):
        """A cold user legitimately yields little; that is the fallback's problem."""
        generator = MinimalGenerator()
        generator.fit(["a"])
        assert len(generator.recommend("u1", 10)) == 1

    def test_score_preserves_input_order_and_length(self):
        generator = MinimalGenerator()
        generator.fit(["a", "b"])
        assert generator.score("u1", ["b", "zzz", "a"]) == [1.0, 0.0, 1.0]

    def test_score_returns_zero_for_unknown_items_rather_than_raising(self):
        generator = MinimalGenerator()
        generator.fit(["a"])
        assert generator.score("u1", ["unknown"]) == [0.0]

    def test_save_load_round_trip_yields_a_usable_model(self, tmp_path):
        generator = MinimalGenerator()
        generator.fit(["a", "b"])
        path = tmp_path / "model.txt"
        generator.save(path)

        loaded = MinimalGenerator.load(path)
        assert loaded.is_fitted is True
        assert [c.item_id for c in loaded.recommend("u1", 2)] == ["a", "b"]

    def test_context_is_optional(self):
        generator = MinimalGenerator()
        generator.fit(["a"])
        assert generator.recommend("u1", 1, None) == generator.recommend("u1", 1, {"locale": "en"})


class TestCandidate:
    def test_merge_unions_sources_and_keeps_the_best_score(self):
        left = Candidate("i1", 0.4, ("lightgcn",), {"lightgcn": 0.4})
        right = Candidate("i1", 0.9, ("sasrec",), {"sasrec": 0.9})
        merged = left.merged_with(right)
        assert merged.score == 0.9
        assert set(merged.sources) == {"lightgcn", "sasrec"}
        assert merged.source_scores == {"lightgcn": 0.4, "sasrec": 0.9}

    def test_merge_is_order_independent_in_sources(self):
        left = Candidate("i1", 0.4, ("a",))
        right = Candidate("i1", 0.9, ("b",))
        assert set(left.merged_with(right).sources) == set(right.merged_with(left).sources)

    def test_merge_does_not_duplicate_a_repeated_source(self):
        left = Candidate("i1", 0.4, ("a",))
        right = Candidate("i1", 0.5, ("a",))
        assert left.merged_with(right).sources == ("a",)

    def test_merging_different_items_is_a_bug_and_raises(self):
        with pytest.raises(ValueError):
            Candidate("i1", 0.4).merged_with(Candidate("i2", 0.4))

    def test_candidates_are_immutable(self):
        with pytest.raises(AttributeError):
            setattr(Candidate("i1", 0.4), "score", 0.9)  # noqa: B010


class TestRanker:
    class MinimalRanker(Ranker):
        name = "minimal"

        def fit(self, features: Any, labels: Any, groups: Any | None = None) -> None:
            self._fitted = True

        def rank(
            self, candidates: list[Candidate], context: dict[str, Any] | None = None
        ) -> list[RankedItem]:
            self.ensure_fitted()
            # Stable sort on the negated score: equal scores keep input order.
            ordered = sorted(candidates, key=lambda c: -c.score)
            return [
                RankedItem(item_id=c.item_id, rank=i + 1, score=c.score, sources=c.sources)
                for i, c in enumerate(ordered)
            ]

        def save(self, path: str | Path) -> None:
            Path(path).write_text("ranker")

        @classmethod
        def load(cls, path: str | Path) -> Self:
            instance = cls()
            instance._fitted = True
            return instance

    def test_unfitted_ranker_refuses_to_rank(self):
        with pytest.raises(ModelNotFittedError):
            self.MinimalRanker().rank([])

    def test_ranks_are_one_based_and_contiguous(self):
        ranker = self.MinimalRanker()
        ranker.fit(None, None)
        ranked = ranker.rank([Candidate("a", 0.1), Candidate("b", 0.9)])
        assert [r.rank for r in ranked] == [1, 2]
        assert ranked[0].item_id == "b"

    def test_equal_scores_preserve_input_order(self):
        """Determinism: a cached response and a fresh one must not disagree."""
        ranker = self.MinimalRanker()
        ranker.fit(None, None)
        candidates = [Candidate("a", 0.5), Candidate("b", 0.5), Candidate("c", 0.5)]
        assert [r.item_id for r in ranker.rank(candidates)] == ["a", "b", "c"]


class TestFeatureContract:
    def test_batch_accepts_complete_rows(self):
        batch = FeatureBatch(
            rows=(FeatureRow("u1", "i1", {"x": 1.0, "y": 2.0}),),
            feature_names=("x", "y"),
            feature_version="f1",
        )
        assert len(batch.rows) == 1

    def test_batch_rejects_a_row_missing_a_declared_feature(self):
        """Catches the silent column-misalignment that breaks a trained ranker."""
        with pytest.raises(ValueError) as exc:
            FeatureBatch(
                rows=(FeatureRow("u1", "i1", {"x": 1.0}),),
                feature_names=("x", "y"),
                feature_version="f1",
            )
        assert "y" in str(exc.value)


class TestAggregationResult:
    def test_empty_result_signals_the_fallback_chain(self):
        assert AggregationResult(candidates=(), contributions={}).is_empty is True

    def test_non_empty_result(self):
        result = AggregationResult(
            candidates=(Candidate("i1", 1.0),), contributions={"popularity": 1}
        )
        assert result.is_empty is False
        assert result.degraded_sources == ()


class TestEvaluationContract:
    def test_ground_truth_rejects_users_with_no_held_out_items(self):
        with pytest.raises(DataError) as exc:
            GroundTruth(relevant={"u1": {"i1": 1.0}, "u2": {}})
        assert "u2" in str(exc.value)

    def test_ground_truth_exposes_users_and_lookups(self):
        truth = GroundTruth(relevant={"u1": {"i1": 1.0}})
        assert truth.users == frozenset({"u1"})
        assert truth.items_for("u1") == {"i1": 1.0}
        assert truth.items_for("unknown") == {}

    def test_evaluator_is_abstract(self):
        with pytest.raises(TypeError):
            Evaluator()  # type: ignore[abstract]

    def test_a_minimal_evaluator_satisfies_the_contract(self):
        class HitRate(Evaluator):
            def evaluate(self, recommendations, ground_truth, k_values):
                scores: dict[str, float] = {}
                for k in k_values:
                    hits = sum(
                        any(
                            item in ground_truth.items_for(user)
                            for item in recommendations.items_for(user)[:k]
                        )
                        for user in ground_truth.users
                    )
                    scores[f"hit_rate@{k}"] = hits / len(ground_truth.users)
                return scores

        class Recs:
            def users(self):
                return ["u1"]

            def items_for(self, user_id):
                return ["i9", "i1"]

        result = HitRate().evaluate(Recs(), GroundTruth({"u1": {"i1": 1.0}}), [1, 2])
        assert result == {"hit_rate@1": 0.0, "hit_rate@2": 1.0}


class TestUserSequence:
    def test_valid_sequence(self):
        sequence = UserSequence(
            user_id="u1",
            item_ids=("a", "b"),
            timestamps=(
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 1, 2, tzinfo=UTC),
            ),
        )
        assert len(sequence) == 2

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            UserSequence("u1", ("a", "b"), (datetime(2026, 1, 1, tzinfo=UTC),))

    def test_out_of_order_timestamps_are_rejected(self):
        with pytest.raises(ValueError) as exc:
            UserSequence(
                "u1",
                ("a", "b"),
                (datetime(2026, 1, 2, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)),
            )
        assert "non-decreasing" in str(exc.value)
