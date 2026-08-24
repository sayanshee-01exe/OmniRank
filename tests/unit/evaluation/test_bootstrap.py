"""Bootstrap confidence intervals."""

from __future__ import annotations

import pytest

from omnirank.core.exceptions import DataError
from omnirank.evaluation.bootstrap import (
    bootstrap_metric,
    bootstrap_primary_metrics,
    paired_bootstrap_delta,
)


def per_user(values: list[float], metric: str = "ndcg@20") -> dict[str, dict[str, float]]:
    return {f"u{index}": {metric: value} for index, value in enumerate(values)}


class TestSingleMetric:
    def test_point_estimate_is_the_mean(self):
        interval = bootstrap_metric(per_user([0.0, 1.0, 0.5]), "ndcg@20", samples=200)
        assert interval.point_estimate == pytest.approx(0.5)

    def test_interval_brackets_the_estimate(self):
        interval = bootstrap_metric(per_user([0.0, 1.0] * 50), "ndcg@20", samples=500)
        assert interval.lower <= interval.point_estimate <= interval.upper

    def test_zero_variance_gives_a_degenerate_interval(self):
        interval = bootstrap_metric(per_user([0.4] * 30), "ndcg@20", samples=200)
        assert interval.lower == pytest.approx(0.4)
        assert interval.upper == pytest.approx(0.4)

    def test_deterministic_given_the_seed(self):
        values = per_user([0.0, 1.0] * 25)
        first = bootstrap_metric(values, "ndcg@20", samples=300, seed=42)
        second = bootstrap_metric(values, "ndcg@20", samples=300, seed=42)
        assert (first.lower, first.upper) == (second.lower, second.upper)

    def test_different_seeds_give_different_intervals(self):
        values = per_user([0.0, 1.0] * 25)
        first = bootstrap_metric(values, "ndcg@20", samples=300, seed=1)
        second = bootstrap_metric(values, "ndcg@20", samples=300, seed=2)
        assert (first.lower, first.upper) != (second.lower, second.upper)

    def test_wider_confidence_level_widens_the_interval(self):
        values = per_user([0.0, 1.0] * 50)
        narrow = bootstrap_metric(values, "ndcg@20", samples=500, confidence_level=0.80)
        wide = bootstrap_metric(values, "ndcg@20", samples=500, confidence_level=0.99)
        assert (wide.upper - wide.lower) >= (narrow.upper - narrow.lower)

    def test_unknown_metric_raises(self):
        with pytest.raises(DataError):
            bootstrap_metric(per_user([1.0]), "absent@20")

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.5])
    def test_invalid_confidence_level_rejected(self, level):
        with pytest.raises(DataError):
            bootstrap_metric(per_user([1.0, 0.0]), "ndcg@20", confidence_level=level)

    def test_invalid_sample_count_rejected(self):
        with pytest.raises(DataError):
            bootstrap_metric(per_user([1.0, 0.0]), "ndcg@20", samples=0)


class TestPairedDelta:
    def test_delta_is_the_mean_difference(self):
        better = per_user([1.0] * 20)
        worse = per_user([0.0] * 20)
        interval = paired_bootstrap_delta(better, worse, "ndcg@20", samples=200)
        assert interval.point_estimate == pytest.approx(1.0)

    def test_identical_models_give_a_zero_delta_including_zero(self):
        """The honest outcome when two models perform identically."""
        values = per_user([0.3, 0.7] * 20)
        interval = paired_bootstrap_delta(values, values, "ndcg@20", samples=300)
        assert interval.point_estimate == pytest.approx(0.0)
        assert not interval.excludes_zero

    def test_clear_difference_excludes_zero(self):
        better = per_user([1.0] * 100)
        worse = per_user([0.0] * 100)
        assert paired_bootstrap_delta(better, worse, "ndcg@20", samples=500).excludes_zero

    def test_only_shared_users_are_compared(self):
        first = {"u1": {"m": 1.0}, "u2": {"m": 1.0}}
        second = {"u1": {"m": 0.0}, "u3": {"m": 0.0}}
        interval = paired_bootstrap_delta(first, second, "m", samples=100)
        assert interval.users == 1

    def test_no_shared_users_raises(self):
        with pytest.raises(DataError):
            paired_bootstrap_delta({"u1": {"m": 1.0}}, {"u2": {"m": 1.0}}, "m")

    def test_deterministic(self):
        first, second = per_user([1.0, 0.0] * 25), per_user([0.0, 1.0] * 25)
        a = paired_bootstrap_delta(first, second, "ndcg@20", samples=300, seed=9)
        b = paired_bootstrap_delta(first, second, "ndcg@20", samples=300, seed=9)
        assert (a.lower, a.upper) == (b.lower, b.upper)

    def test_result_does_not_depend_on_dict_order(self):
        first = {"u1": {"m": 1.0}, "u2": {"m": 0.0}}
        second = {"u2": {"m": 0.0}, "u1": {"m": 1.0}}
        forward = paired_bootstrap_delta(first, second, "m", samples=200, seed=3)
        reverse = paired_bootstrap_delta(
            dict(reversed(list(first.items()))), second, "m", samples=200, seed=3
        )
        assert (forward.lower, forward.upper) == (reverse.lower, reverse.upper)


class TestReporting:
    def test_payload_has_the_expected_keys(self):
        payload = bootstrap_metric(per_user([0.0, 1.0]), "ndcg@20", samples=100).to_dict()
        assert {"metric", "point_estimate", "ci_lower", "ci_upper", "excludes_zero"} <= set(payload)

    def test_multiple_metrics_at_once(self):
        values = {"u1": {"recall@20": 1.0, "ndcg@20": 0.5}}
        intervals = bootstrap_primary_metrics(values, ["recall@20", "ndcg@20"], samples=50)
        assert set(intervals) == {"recall@20", "ndcg@20"}
