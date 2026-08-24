"""Evaluator behaviour: denominators, strict/warm views, and non-interference."""

from __future__ import annotations

import pytest

from omnirank.evaluation.base import GroundTruth
from omnirank.evaluation.evaluator import STRICT, WARM, OfflineEvaluator
from omnirank.evaluation.ground_truth import EvaluationGroundTruth
from omnirank.evaluation.recommendations import RecommendationSet


def make_ground_truth(relevant, *, cold=frozenset()):
    return EvaluationGroundTruth(
        truth=GroundTruth(relevant=relevant),
        target_split="validation",
        fit_splits=("train",),
        target_internal_items={user: index for index, user in enumerate(relevant)},
        cold_target_users=frozenset(cold),
    )


class TestDenominator:
    def test_users_without_recommendations_score_zero_and_stay_in_denominator(self):
        """Dropping them would measure only the traffic the model can serve."""
        truth = make_ground_truth({"u1": {"a": 1.0}, "u2": {"b": 1.0}})
        recs = RecommendationSet.from_mapping({"u1": ["a"], "u2": []})
        result = OfflineEvaluator().evaluate_detailed(recs, truth, k_values=[5])
        assert result.users_evaluated == 2
        assert result.metrics["recall@5"] == pytest.approx(0.5)

    def test_user_missing_from_recommendations_entirely_scores_zero(self):
        truth = make_ground_truth({"u1": {"a": 1.0}, "u2": {"b": 1.0}})
        recs = RecommendationSet.from_mapping({"u1": ["a"]})
        result = OfflineEvaluator().evaluate_detailed(recs, truth, k_values=[5])
        assert result.metrics["recall@5"] == pytest.approx(0.5)
        assert result.user_coverage == pytest.approx(0.5)

    def test_perfect_model(self):
        truth = make_ground_truth({"u1": {"a": 1.0}, "u2": {"b": 1.0}})
        recs = RecommendationSet.from_mapping({"u1": ["a"], "u2": ["b"]})
        result = OfflineEvaluator().evaluate_detailed(recs, truth, k_values=[5])
        assert result.metrics["recall@5"] == pytest.approx(1.0)
        assert result.metrics["ndcg@5"] == pytest.approx(1.0)


class TestStrictVersusWarm:
    def test_strict_counts_cold_targets_as_misses(self):
        truth = make_ground_truth({"u1": {"a": 1.0}, "u2": {"cold": 1.0}}, cold={"u2"})
        recs = RecommendationSet.from_mapping({"u1": ["a"], "u2": ["x", "y"]})
        strict = OfflineEvaluator().evaluate_detailed(recs, truth, k_values=[5], view=STRICT)
        assert strict.users_evaluated == 2
        assert strict.metrics["recall@5"] == pytest.approx(0.5)

    def test_warm_excludes_cold_targets(self):
        truth = make_ground_truth({"u1": {"a": 1.0}, "u2": {"cold": 1.0}}, cold={"u2"})
        recs = RecommendationSet.from_mapping({"u1": ["a"], "u2": ["x", "y"]})
        warm = OfflineEvaluator().evaluate_detailed(recs, truth, k_values=[5], view=WARM)
        assert warm.users_evaluated == 1
        assert warm.metrics["recall@5"] == pytest.approx(1.0)

    def test_warm_is_never_lower_than_strict_for_the_same_model(self):
        """Removing unreachable targets can only help; that is why both ship."""
        truth = make_ground_truth({"u1": {"a": 1.0}, "u2": {"cold": 1.0}}, cold={"u2"})
        recs = RecommendationSet.from_mapping({"u1": ["a"], "u2": []})
        evaluator = OfflineEvaluator()
        strict = evaluator.evaluate_detailed(recs, truth, k_values=[5], view=STRICT)
        warm = evaluator.evaluate_detailed(recs, truth, k_values=[5], view=WARM)
        assert warm.metrics["recall@5"] >= strict.metrics["recall@5"]

    def test_reachable_fraction_is_reported(self):
        truth = make_ground_truth({"u1": {"a": 1.0}, "u2": {"cold": 1.0}}, cold={"u2"})
        assert truth.reachable_fraction == pytest.approx(0.5)


class TestContract:
    def test_phase_one_evaluate_signature_still_works(self):
        truth = GroundTruth(relevant={"u1": {"a": 1.0}})
        recs = RecommendationSet.from_mapping({"u1": ["a"]})
        metrics = OfflineEvaluator().evaluate(recs, truth, [1, 5])
        assert metrics["recall@1"] == pytest.approx(1.0)
        assert "ndcg@5" in metrics

    def test_evaluator_does_not_modify_recommendations(self):
        """A metric that can re-rank what it measures is not a measurement."""
        truth = make_ground_truth({"u1": {"c": 1.0}})
        recs = RecommendationSet.from_mapping({"u1": ["a", "b", "c"]})
        before = list(recs.items_for("u1"))
        OfflineEvaluator().evaluate_detailed(recs, truth, k_values=[1, 2, 3])
        assert list(recs.items_for("u1")) == before

    def test_unknown_metric_is_rejected(self):
        from omnirank.core.exceptions import DataError

        with pytest.raises(DataError):
            OfflineEvaluator(metrics=["not_a_metric"])

    def test_per_user_values_are_kept_for_bootstrap(self):
        truth = make_ground_truth({"u1": {"a": 1.0}, "u2": {"b": 1.0}})
        recs = RecommendationSet.from_mapping({"u1": ["a"], "u2": []})
        result = OfflineEvaluator().evaluate_detailed(recs, truth, k_values=[5])
        assert result.per_user["u1"]["recall@5"] == pytest.approx(1.0)
        assert result.per_user["u2"]["recall@5"] == 0.0


class TestBeyondAccuracy:
    def test_computed_when_catalogue_and_counts_supplied(self):
        truth = make_ground_truth({"u1": {"a": 1.0}, "u2": {"b": 1.0}})
        recs = RecommendationSet.from_mapping({"u1": ["a", "b"], "u2": ["a", "c"]})
        result = OfflineEvaluator().evaluate_detailed(
            recs,
            truth,
            k_values=[2],
            eligible_catalogue={"a", "b", "c", "d"},
            training_counts={"a": 10, "b": 2, "c": 1, "d": 1},
            category_by_item={"a": "x", "b": "x", "c": "y", "d": "y"},
        )
        flat = result.flat()
        assert flat["coverage@2"] == pytest.approx(0.75)
        assert "novelty@2" in flat
        assert "gini@2" in flat
        assert "category_diversity@2" in flat

    def test_intra_list_diversity_is_marked_unavailable_not_zero(self):
        truth = make_ground_truth({"u1": {"a": 1.0}})
        recs = RecommendationSet.from_mapping({"u1": ["a", "b"]})
        result = OfflineEvaluator().evaluate_detailed(
            recs,
            truth,
            k_values=[2],
            eligible_catalogue={"a", "b"},
            training_counts={"a": 1, "b": 1},
        )
        entry = result.beyond_accuracy[0]
        assert entry.intra_list_diversity is None
        assert "not downloaded" in entry.intra_list_diversity_unavailable_reason
        assert "intra_list_diversity@2" not in result.flat()


class TestFastPathAgreesWithReference:
    """The evaluator's single-scan path must equal the reference metric functions.

    The functions in ``omnirank.evaluation.metrics`` are the specification and
    are tested against hand-computed values. The evaluator computes the same
    numbers a faster way; this asserts the two never diverge, so the speedup
    cannot silently change a reported metric.
    """

    @staticmethod
    def _reference(recommended, relevant_grades, k_values, metrics) -> dict:
        from omnirank.evaluation.metrics import METRIC_FUNCTIONS, metric_name

        relevant = set(relevant_grades)
        out = {}
        for metric in metrics:
            function = METRIC_FUNCTIONS[metric]
            for k in k_values:
                if metric == "ndcg":
                    from omnirank.evaluation.metrics import ndcg_at_k

                    value = ndcg_at_k(recommended, relevant, k, relevance=dict(relevant_grades))
                else:
                    value = function(recommended, relevant, k)
                out[metric_name(metric, k)] = value
        return out

    @pytest.mark.parametrize("seed", range(25))
    def test_randomised_agreement(self, seed):
        import random

        from omnirank.evaluation.evaluator import _score_one_user
        from omnirank.evaluation.metrics import RANKING_METRICS

        rng = random.Random(seed)
        catalogue = [f"i{index}" for index in range(40)]
        recommended = rng.sample(catalogue, rng.randint(0, 30))
        relevant_grades = dict.fromkeys(rng.sample(catalogue, rng.randint(1, 5)), 1.0)
        k_values = (1, 3, 5, 10, 20)

        fast = _score_one_user(
            recommended, relevant_grades, k_values, max(k_values), RANKING_METRICS
        )
        reference = self._reference(recommended, relevant_grades, k_values, RANKING_METRICS)
        for key, value in reference.items():
            assert fast[key] == pytest.approx(value), key

    @pytest.mark.parametrize("seed", range(10))
    def test_graded_relevance_agreement(self, seed):
        import random

        from omnirank.evaluation.evaluator import _score_one_user

        rng = random.Random(seed + 100)
        catalogue = [f"i{index}" for index in range(20)]
        recommended = rng.sample(catalogue, 15)
        relevant_grades = {item: float(rng.randint(1, 5)) for item in rng.sample(catalogue, 4)}
        fast = _score_one_user(recommended, relevant_grades, (5, 10), 10, ("ndcg",))
        reference = self._reference(recommended, relevant_grades, (5, 10), ("ndcg",))
        for key, value in reference.items():
            assert fast[key] == pytest.approx(value), key

    def test_empty_recommendations_agree(self):
        from omnirank.evaluation.evaluator import _score_one_user
        from omnirank.evaluation.metrics import RANKING_METRICS

        fast = _score_one_user([], {"a": 1.0}, (5, 20), 20, RANKING_METRICS)
        assert set(fast.values()) == {0.0}

    def test_no_relevant_items_agree(self):
        from omnirank.evaluation.evaluator import _score_one_user
        from omnirank.evaluation.metrics import RANKING_METRICS

        fast = _score_one_user(["a", "b"], {}, (5,), 5, RANKING_METRICS)
        assert set(fast.values()) == {0.0}
