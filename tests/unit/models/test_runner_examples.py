"""Recommendation-example selection.

The value of an examples section depends entirely on it not being a highlight
reel, so the selection rule itself is tested.
"""

from __future__ import annotations

from omnirank.evaluation.base import GroundTruth
from omnirank.evaluation.ground_truth import EvaluationGroundTruth
from omnirank.evaluation.recommendations import RecommendationSet
from omnirank.models.baselines.runner import recommendation_examples


def make(users: int = 40, hit_every: int = 4, *, any_hits: bool = True):
    """Ground truth plus recommendations where every nth user is a hit.

    ``any_hits=False`` gives a population with no hits at all. A large modulus
    cannot express that, because ``0 % n == 0`` always makes user 0 a hit.
    """
    relevant = {f"u{index}": {f"target{index}": 1.0} for index in range(users)}
    mapping = {
        f"u{index}": (
            [f"target{index}", "x", "y"] if any_hits and index % hit_every == 0 else ["x", "y", "z"]
        )
        for index in range(users)
    }
    truth = EvaluationGroundTruth(
        truth=GroundTruth(relevant=relevant),
        target_split="validation",
        fit_splits=("train",),
        target_internal_items={f"u{index}": index for index in range(users)},
        cold_target_users=frozenset(),
    )
    return RecommendationSet.from_mapping(mapping), truth


class TestSelection:
    def test_returns_all_groups_and_the_base_rate(self):
        recommendations, truth = make()
        result = recommendation_examples(recommendations, truth, count=5)
        assert set(result) == {"sampled", "failures", "successes", "hit_rate_in_top_10"}
        assert len(result["sampled"]) == 5
        assert len(result["failures"]) == 5
        assert len(result["successes"]) == 5

    def test_successes_contain_only_hits(self):
        recommendations, truth = make()
        assert all(
            entry["target_in_top_10"]
            for entry in recommendation_examples(recommendations, truth, count=5)["successes"]
        )

    def test_base_rate_is_reported_so_the_sample_is_not_mistaken_for_the_population(self):
        recommendations, truth = make(users=40, hit_every=4)
        result = recommendation_examples(recommendations, truth, count=5)
        assert result["hit_rate_in_top_10"] == 0.25

    def test_deterministic(self):
        recommendations, truth = make()
        first = recommendation_examples(recommendations, truth, count=5)
        second = recommendation_examples(recommendations, truth, count=5)
        assert first == second

    def test_seed_changes_the_sample(self):
        recommendations, truth = make()
        first = recommendation_examples(recommendations, truth, count=5, seed=0)
        second = recommendation_examples(recommendations, truth, count=5, seed=99)
        assert first["sampled"] != second["sampled"]

    def test_sample_is_not_selected_on_success(self):
        """A neutral draw must contain misses; only 1 user in 4 is a hit here."""
        recommendations, truth = make(users=200, hit_every=4)
        sampled = recommendation_examples(recommendations, truth, count=20)["sampled"]
        hits = sum(1 for entry in sampled if entry["target_in_top_10"])
        assert 0 < hits < len(sampled)

    def test_failures_contain_only_misses(self):
        recommendations, truth = make()
        failures = recommendation_examples(recommendations, truth, count=5)["failures"]
        assert all(entry["target_in_top_10"] is False for entry in failures)

    def test_target_rank_is_reported_when_hit(self):
        recommendations, truth = make(users=4, hit_every=1)
        sampled = recommendation_examples(recommendations, truth, count=4)["sampled"]
        assert all(entry["target_rank_within_top_10"] == 1 for entry in sampled)


class TestAnonymisation:
    def test_real_user_ids_never_appear(self):
        """A public report does not need real identifiers to be useful."""
        recommendations, truth = make()
        result = recommendation_examples(recommendations, truth, count=5)
        rendered = str(result)
        for index in range(40):
            assert f'"u{index}"' not in rendered
        # `hit_rate_in_top_10` is a float, not a group of entries.
        groups = [value for value in result.values() if isinstance(value, list)]
        assert all(entry["user"].startswith("user_") for group in groups for entry in group)

    def test_the_same_user_gets_the_same_label_across_calls(self):
        """Stable labels are what make a cross-model comparison possible."""
        recommendations, truth = make()
        first = recommendation_examples(recommendations, truth, count=5)["sampled"]
        second = recommendation_examples(recommendations, truth, count=5)["sampled"]
        assert [entry["user"] for entry in first] == [entry["user"] for entry in second]


class TestEdgeCases:
    def test_empty_ground_truth(self):
        truth = EvaluationGroundTruth(
            truth=GroundTruth(relevant={}),
            target_split="validation",
            fit_splits=("train",),
            target_internal_items={},
            cold_target_users=frozenset(),
        )
        assert recommendation_examples(RecommendationSet(), truth) == {
            "sampled": [],
            "failures": [],
        }

    def test_no_failures_when_every_user_is_a_hit(self):
        recommendations, truth = make(users=10, hit_every=1)
        result = recommendation_examples(recommendations, truth, count=5)
        assert result["failures"] == []
        assert result["hit_rate_in_top_10"] == 1.0

    def test_no_successes_when_every_user_is_a_miss(self):
        recommendations, truth = make(users=10, any_hits=False)
        result = recommendation_examples(recommendations, truth, count=5)
        assert result["successes"] == []
        assert result["hit_rate_in_top_10"] == 0.0

    def test_count_larger_than_the_population(self):
        recommendations, truth = make(users=3, hit_every=1)
        assert len(recommendation_examples(recommendations, truth, count=50)["sampled"]) == 3
