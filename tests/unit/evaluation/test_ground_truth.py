"""Ground-truth construction, including the leakage guard."""

from __future__ import annotations

import pandas as pd
import pytest

from omnirank.core.exceptions import DataError
from omnirank.evaluation.ground_truth import build_ground_truth

I2E_ITEM = {10: "iA", 11: "iB", 12: "iC"}
I2E_USER = {0: "uA", 1: "uB"}


def targets(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["internal_user_id", "internal_item_id", "interaction_order"])


def build(target_rows, *, fit_items=frozenset({10, 11}), fit_rows=None):
    return build_ground_truth(
        targets(target_rows),
        target_split="validation",
        fit_splits=("train",),
        fit_item_ids=set(fit_items),
        internal_to_external_item=I2E_ITEM,
        internal_to_external_user=I2E_USER,
        fit_interactions=targets(fit_rows) if fit_rows is not None else None,
    )


class TestConstruction:
    def test_maps_internal_ids_to_external(self):
        truth = build([(0, 10, 5)])
        assert truth.truth.items_for("uA") == {"iA": 1.0}

    def test_records_provenance(self):
        truth = build([(0, 10, 5)])
        provenance = truth.provenance()
        assert provenance["target_split"] == "validation"
        assert provenance["fit_splits"] == ["train"]

    def test_empty_targets_rejected(self):
        with pytest.raises(DataError):
            build([])

    def test_relevance_is_binary(self):
        """PixelRec records one implicit signal; there is nothing to grade with."""
        truth = build([(0, 10, 5), (1, 11, 3)])
        assert set(truth.truth.items_for("uA").values()) == {1.0}


class TestWarmAndCold:
    def test_cold_targets_are_classified_not_removed(self):
        """Removing them would turn a real failure into an invisible one."""
        truth = build([(0, 10, 5), (1, 12, 3)])
        assert truth.users == {"uA", "uB"}
        assert truth.cold_target_users == {"uB"}

    def test_warm_users_exclude_cold_targets(self):
        truth = build([(0, 10, 5), (1, 12, 3)])
        assert truth.warm_users == {"uA"}

    def test_reachable_fraction(self):
        truth = build([(0, 10, 5), (1, 12, 3)])
        assert truth.reachable_fraction == pytest.approx(0.5)

    def test_all_warm(self):
        truth = build([(0, 10, 5), (1, 11, 3)])
        assert truth.cold_target_users == frozenset()
        assert truth.reachable_fraction == 1.0


class TestLeakageGuard:
    def test_accepts_history_that_precedes_targets(self):
        build([(0, 10, 5)], fit_rows=[(0, 11, 1), (0, 11, 2)])

    def test_rejects_history_at_or_after_a_target(self):
        """A mis-specified fit boundary would make every metric optimistic."""
        with pytest.raises(DataError) as exc:
            build([(0, 10, 5)], fit_rows=[(0, 11, 9)])
        assert "precede" in str(exc.value)

    def test_rejects_history_exactly_at_the_target_position(self):
        with pytest.raises(DataError):
            build([(0, 10, 5)], fit_rows=[(0, 11, 5)])

    def test_guard_is_skipped_when_no_fit_data_supplied(self):
        build([(0, 10, 5)], fit_rows=None)
