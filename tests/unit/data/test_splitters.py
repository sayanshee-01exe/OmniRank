"""Per-user leave-last-N splitting and interaction ordering."""

from __future__ import annotations

import pandas as pd
import pytest

from omnirank.core.config import SplittingConfig
from omnirank.core.exceptions import DataError
from omnirank.data.splitters import (
    TEST,
    TRAIN,
    VALIDATION,
    assign_interaction_order,
    split_leave_last_n,
)


def log(histories: dict[str, int], *, start: int = 1_640_995_200) -> pd.DataFrame:
    """Build an interaction log: user -> number of events."""
    rows = []
    row_id = 0
    for user, count in histories.items():
        for position in range(count):
            rows.append(
                {
                    "external_user_id": user,
                    "external_item_id": f"i{position}",
                    "timestamp": start + position * 3600,
                    "source_row_id": row_id,
                }
            )
            row_id += 1
    return pd.DataFrame(rows)


@pytest.fixture
def config() -> SplittingConfig:
    return SplittingConfig(strategy="per_user_leave_last_n")


class TestOrdering:
    def test_order_is_per_user_and_zero_based(self):
        ordered = assign_interaction_order(log({"a": 3, "b": 2}))
        assert ordered[ordered.external_user_id == "a"]["interaction_order"].tolist() == [0, 1, 2]
        assert ordered[ordered.external_user_id == "b"]["interaction_order"].tolist() == [0, 1]

    def test_order_follows_the_timestamp_not_the_file(self):
        frame = log({"a": 3}).iloc[::-1].reset_index(drop=True)
        ordered = assign_interaction_order(frame)
        assert ordered["timestamp"].is_monotonic_increasing

    def test_ties_break_deterministically_on_source_row_id(self):
        frame = pd.DataFrame(
            {
                "external_user_id": ["a", "a"],
                "external_item_id": ["i1", "i2"],
                "timestamp": [100, 100],
                "source_row_id": [7, 3],
            }
        )
        ordered = assign_interaction_order(frame)
        assert ordered["source_row_id"].tolist() == [3, 7]

    def test_missing_ordering_columns_fail(self):
        with pytest.raises(DataError):
            assign_interaction_order(pd.DataFrame({"external_user_id": ["a"]}))


class TestSplitAssignment:
    def test_last_event_is_test_and_second_last_is_validation(self, config):
        result = split_leave_last_n(log({"a": 5}), config)
        frame = result.interactions.sort_values("interaction_order")
        assert frame["split"].tolist() == [TRAIN, TRAIN, TRAIN, VALIDATION, TEST]

    def test_minimum_eligible_user_is_split(self, config):
        result = split_leave_last_n(log({"a": 3}), config)
        assert result.sizes == {"train": 1, "validation": 1, "test": 1}

    def test_ineligible_users_contribute_training_history(self, config):
        """Discarding them would shrink the catalogue for no benefit."""
        result = split_leave_last_n(log({"a": 2}), config)
        assert result.sizes == {"train": 2, "validation": 0, "test": 0}
        assert result.ineligible_users == 1

    def test_mixed_eligibility(self, config):
        result = split_leave_last_n(log({"a": 5, "b": 2}), config)
        assert result.eligible_users == 1
        assert result.ineligible_users == 1
        assert result.sizes == {"train": 5, "validation": 1, "test": 1}

    def test_multiple_held_out_events(self):
        config = SplittingConfig(
            strategy="per_user_leave_last_n", validation_interactions=2, test_interactions=2
        )
        result = split_leave_last_n(log({"a": 10}), config)
        assert result.sizes == {"train": 6, "validation": 2, "test": 2}

    def test_zero_validation_events_is_supported(self):
        config = SplittingConfig(
            strategy="per_user_leave_last_n", validation_interactions=0, test_interactions=1
        )
        result = split_leave_last_n(log({"a": 4}), config)
        assert result.sizes == {"train": 3, "validation": 0, "test": 1}

    def test_empty_log_is_rejected(self, config):
        with pytest.raises(DataError):
            split_leave_last_n(pd.DataFrame(), config)


class TestOrderingInvariants:
    def test_train_strictly_precedes_validation(self, config):
        frame = split_leave_last_n(log({"a": 6, "b": 4}), config).interactions
        for user, group in frame.groupby("external_user_id"):
            train_max = group[group.split == TRAIN]["interaction_order"].max()
            validation_min = group[group.split == VALIDATION]["interaction_order"].min()
            assert train_max < validation_min, user

    def test_validation_strictly_precedes_test(self, config):
        frame = split_leave_last_n(log({"a": 6, "b": 4}), config).interactions
        for user, group in frame.groupby("external_user_id"):
            validation_max = group[group.split == VALIDATION]["interaction_order"].max()
            test_min = group[group.split == TEST]["interaction_order"].min()
            assert validation_max < test_min, user

    def test_no_interaction_lands_in_two_splits(self, config):
        frame = split_leave_last_n(log({"a": 6}), config).interactions
        assert len(frame) == len(frame.drop_duplicates(subset=["source_row_id"]))


class TestDeterminism:
    def test_repeated_runs_agree(self, config):
        data = log({"a": 5, "b": 7})
        first = split_leave_last_n(data, config).interactions
        second = split_leave_last_n(data, config).interactions
        assert first.equals(second)

    def test_input_row_order_does_not_matter(self, config):
        data = log({"a": 5, "b": 7})
        forward = split_leave_last_n(data, config).interactions
        shuffled = split_leave_last_n(
            data.sample(frac=1.0, random_state=0).reset_index(drop=True), config
        ).interactions
        assert forward.equals(shuffled)


class TestStatistics:
    def test_reports_every_required_field(self, config):
        stats = split_leave_last_n(log({"a": 5, "b": 4}), config).statistics()
        for name in ("train", "validation", "test"):
            assert f"{name}_rows" in stats
            assert f"{name}_users" in stats
            assert f"{name}_items" in stats
        assert stats["ordering_field"] == "timestamp"
        assert stats["split_strategy"] == "per_user_leave_last_n"
