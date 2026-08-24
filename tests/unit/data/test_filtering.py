"""Iterative k-core filtering: convergence, cascading, and the audit trail."""

from __future__ import annotations

import pandas as pd
import pytest

from omnirank.core.exceptions import DataError
from omnirank.data.filtering import apply_iterative_filtering, snapshot_before_filtering


def frame_from(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Build an interaction frame from (user, item) pairs."""
    return pd.DataFrame(pairs, columns=["external_user_id", "external_item_id"])


class TestBasics:
    def test_disabled_filtering_returns_the_input(self):
        data = frame_from([("u1", "i1"), ("u2", "i2")])
        result = apply_iterative_filtering(data, enabled=False)
        assert len(result.interactions) == 2
        assert result.iterations == []
        assert result.enabled is False

    def test_zero_thresholds_remove_nothing(self):
        data = frame_from([("u1", "i1"), ("u2", "i2")])
        result = apply_iterative_filtering(data, min_user_interactions=0, min_item_interactions=0)
        assert len(result.interactions) == 2
        assert result.converged

    def test_singleton_items_are_removed(self):
        data = frame_from([("u1", "i1"), ("u2", "i1"), ("u1", "i2")])
        result = apply_iterative_filtering(data, min_user_interactions=0, min_item_interactions=2)
        assert set(result.interactions["external_item_id"]) == {"i1"}

    def test_sparse_users_are_removed(self):
        data = frame_from([("u1", "i1"), ("u1", "i2"), ("u2", "i1")])
        result = apply_iterative_filtering(data, min_user_interactions=2, min_item_interactions=0)
        assert set(result.interactions["external_user_id"]) == {"u1"}


class TestIteration:
    def test_removal_cascades_across_rounds(self):
        """u2 only survives via i2; removing i2 must then remove u2."""
        data = frame_from(
            [
                ("u1", "i1"),
                ("u2", "i1"),
                ("u3", "i1"),
                ("u1", "i2"),
                ("u2", "i2"),
                ("u2", "i3"),
            ]
        )
        result = apply_iterative_filtering(data, min_user_interactions=2, min_item_interactions=2)
        assert result.converged
        # i3 is a singleton and goes; u3 has one interaction and goes.
        assert "i3" not in set(result.interactions["external_item_id"])
        assert "u3" not in set(result.interactions["external_user_id"])

    def test_result_satisfies_both_thresholds_at_the_fixed_point(self):
        data = frame_from([(f"u{u}", f"i{i}") for u in range(6) for i in range(u % 4 + 1)])
        result = apply_iterative_filtering(data, min_user_interactions=2, min_item_interactions=2)
        surviving = result.interactions
        if not surviving.empty:
            assert surviving.groupby("external_user_id").size().min() >= 2
            assert surviving.groupby("external_item_id").size().min() >= 2

    def test_every_iteration_is_recorded(self):
        data = frame_from(
            [
                ("u1", "i1"),
                ("u2", "i1"),
                ("u1", "i2"),
                ("u2", "i2"),
                ("u3", "i3"),
            ]
        )
        result = apply_iterative_filtering(data, min_user_interactions=2, min_item_interactions=2)
        assert len(result.iterations) >= 1
        assert result.iterations[0].iteration == 1

    def test_removed_ids_are_recorded(self):
        data = frame_from([("u1", "i1"), ("u2", "i1"), ("u3", "i3")])
        result = apply_iterative_filtering(data, min_user_interactions=0, min_item_interactions=2)
        assert "i3" in result.removed_item_ids

    def test_over_aggressive_thresholds_fail_with_guidance(self):
        data = frame_from([("u1", "i1"), ("u2", "i2")])
        with pytest.raises(DataError) as exc:
            apply_iterative_filtering(data, min_user_interactions=50, min_item_interactions=50)
        assert "subset-users" in str(exc.value)

    def test_non_convergence_is_bounded(self):
        data = frame_from([("u1", "i1"), ("u1", "i2"), ("u2", "i1"), ("u2", "i2")])
        with pytest.raises(DataError) as exc:
            apply_iterative_filtering(
                data, min_user_interactions=2, min_item_interactions=2, max_iterations=0
            )
        assert "converge" in str(exc.value)


class TestSnapshot:
    def test_cold_start_population_is_captured_before_removal(self):
        """Filtering destroys the very population cold-start analysis needs."""
        data = frame_from([("u1", "i1"), ("u2", "i1"), ("u1", "i2")])
        snapshot = snapshot_before_filtering(data, min_user_interactions=2, min_item_interactions=2)
        assert snapshot.singleton_items == 1
        assert snapshot.total_items == 2
        assert snapshot.total_users == 2

    def test_snapshot_survives_into_the_report(self):
        data = frame_from([("u1", "i1"), ("u2", "i1"), ("u1", "i2")])
        result = apply_iterative_filtering(data, min_user_interactions=0, min_item_interactions=2)
        assert result.report()["before"]["singleton_items"] == 1


class TestDeterminism:
    def test_row_order_does_not_change_the_outcome(self):
        pairs = [("u1", "i1"), ("u2", "i1"), ("u1", "i2"), ("u2", "i2"), ("u3", "i3")]
        first = apply_iterative_filtering(
            frame_from(pairs), min_user_interactions=2, min_item_interactions=2
        )
        second = apply_iterative_filtering(
            frame_from(list(reversed(pairs))), min_user_interactions=2, min_item_interactions=2
        )
        assert set(map(tuple, first.interactions.to_numpy())) == set(
            map(tuple, second.interactions.to_numpy())
        )

    def test_report_is_complete(self):
        data = frame_from([("u1", "i1"), ("u2", "i1"), ("u1", "i2"), ("u2", "i2")])
        report = apply_iterative_filtering(
            data, min_user_interactions=2, min_item_interactions=2
        ).report()
        assert set(report) >= {
            "enabled",
            "converged",
            "configuration",
            "before",
            "iterations",
            "after",
            "totals",
        }
