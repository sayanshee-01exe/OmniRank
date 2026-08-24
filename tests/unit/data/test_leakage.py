"""Leakage checks.

Every test here injects a *specific* leak and asserts the corresponding check
catches it. A leakage check that has never been shown to fail on bad data is not
evidence of anything - it is a check that might be vacuously passing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from omnirank.core.exceptions import DataError
from omnirank.data.leakage import (
    Severity,
    check_graph_is_training_only,
    check_mapping_covers_all_splits,
    check_no_duplicate_across_splits,
    check_no_future_items_in_training,
    check_popularity_is_training_only,
    check_sequences_contain_only_past,
    check_split_labels_absent_from_features,
    check_train_precedes_test,
    check_train_precedes_validation,
    check_user_statistics_are_training_only,
    check_validation_precedes_test,
    run_all_checks,
)
from omnirank.data.sequences import build_sequences
from omnirank.data.statistics import build_item_popularity, build_user_statistics


class TestCleanDataPasses:
    def test_all_critical_checks_pass_on_a_correct_split(self, split_frame):
        report = run_all_checks(split_frame)
        assert report.passed
        assert report.critical_failures == []

    def test_raise_if_failed_is_a_no_op_when_clean(self, split_frame):
        run_all_checks(split_frame).raise_if_failed()


class TestDuplicateAcrossSplits:
    def test_detects_an_interaction_in_two_splits(self, split_frame):
        leaked = pd.concat(
            [split_frame, split_frame.iloc[[0]].assign(split="test")], ignore_index=True
        )
        check = check_no_duplicate_across_splits(leaked)
        assert not check.passed
        assert check.severity is Severity.CRITICAL

    def test_names_the_offending_interaction(self, split_frame):
        leaked = pd.concat(
            [split_frame, split_frame.iloc[[0]].assign(split="test")], ignore_index=True
        )
        assert (
            split_frame.iloc[0]["interaction_id"]
            in (check_no_duplicate_across_splits(leaked).evidence["examples"])
        )


class TestOrderingChecks:
    def test_detects_a_training_event_after_a_validation_target(self, split_frame):
        """The classic leak: a future event left in the training history."""
        leaked = split_frame.copy()
        leaked.loc[leaked.index[0], "interaction_order"] = 99
        assert not check_train_precedes_validation(leaked).passed

    def test_detects_a_training_event_after_a_test_target(self, split_frame):
        leaked = split_frame.copy()
        leaked.loc[leaked.index[0], "interaction_order"] = 99
        assert not check_train_precedes_test(leaked).passed

    def test_detects_a_validation_event_after_a_test_target(self, split_frame):
        leaked = split_frame.copy()
        validation_rows = leaked.index[leaked.split == "validation"]
        leaked.loc[validation_rows[0], "interaction_order"] = 99
        assert not check_validation_precedes_test(leaked).passed

    def test_passes_on_correct_ordering(self, split_frame):
        assert check_train_precedes_validation(split_frame).passed
        assert check_validation_precedes_test(split_frame).passed
        assert check_train_precedes_test(split_frame).passed


class TestSequenceChecks:
    def test_detects_the_target_inside_the_input(self, split_frame):
        built, _ = build_sequences(split_frame, split="test", maximum_length=10, minimum_length=1)
        leaked = built.copy()
        leaked.at[leaked.index[0], "item_sequence"] = [
            *leaked.iloc[0]["item_sequence"],
            leaked.iloc[0]["target_item"],
        ]
        check = check_sequences_contain_only_past(leaked, split_name="test")
        assert not check.passed
        assert "include the target" in check.detail

    def test_detects_a_future_event_in_the_history(self, split_frame):
        built, _ = build_sequences(split_frame, split="test", maximum_length=10, minimum_length=1)
        leaked = built.copy()
        leaked.at[leaked.index[0], "interaction_order_sequence"] = [999]
        assert not check_sequences_contain_only_past(leaked, split_name="test").passed

    def test_passes_on_correctly_built_sequences(self, split_frame):
        for split in ("train", "validation", "test"):
            built, _ = build_sequences(
                split_frame, split=split, maximum_length=10, minimum_length=1
            )
            assert check_sequences_contain_only_past(built, split_name=split).passed

    def test_empty_sequences_pass_vacuously(self):
        assert check_sequences_contain_only_past(pd.DataFrame(), split_name="test").passed


class TestGraphCheck:
    def test_detects_a_held_out_edge_in_the_training_graph(self, split_frame):
        test_row = split_frame[split_frame.split == "test"].iloc[0]
        leaked = pd.DataFrame(
            {
                "internal_user_id": [test_row["internal_user_id"]],
                "internal_item_id": [test_row["internal_item_id"]],
                "edge_weight": [1.0],
                "interaction_order": [0],
            }
        )
        assert not check_graph_is_training_only(leaked, split_frame).passed

    def test_training_edges_pass(self, split_frame):
        train = split_frame[split_frame.split == "train"]
        edges = train[["internal_user_id", "internal_item_id", "interaction_order"]].assign(
            edge_weight=1.0
        )
        assert check_graph_is_training_only(edges, split_frame).passed

    def test_a_pair_present_in_both_train_and_test_is_not_a_leak(self, split_frame):
        """Re-interacting with a training item later is normal user behaviour."""
        frame = pd.concat(
            [
                split_frame,
                split_frame.iloc[[0]].assign(
                    split="test", interaction_order=99, interaction_id="dup"
                ),
            ],
            ignore_index=True,
        )
        edges = pd.DataFrame(
            {
                "internal_user_id": [split_frame.iloc[0]["internal_user_id"]],
                "internal_item_id": [split_frame.iloc[0]["internal_item_id"]],
                "edge_weight": [1.0],
                "interaction_order": [0],
            }
        )
        assert check_graph_is_training_only(edges, frame).passed


class TestStatisticChecks:
    def test_detects_popularity_counted_over_all_splits(self, split_frame):
        """The exact bug: counting the whole log instead of the training split."""
        leaked = (
            split_frame.groupby("internal_item_id")
            .size()
            .rename("training_interaction_count")
            .reset_index()
        )
        assert not check_popularity_is_training_only(leaked, split_frame).passed

    def test_training_only_popularity_passes(self, split_frame):
        assert check_popularity_is_training_only(
            build_item_popularity(split_frame), split_frame
        ).passed

    def test_detects_user_statistics_counted_over_all_splits(self, split_frame):
        leaked = (
            split_frame.groupby("internal_user_id")
            .size()
            .rename("training_interaction_count")
            .reset_index()
        )
        assert not check_user_statistics_are_training_only(leaked, split_frame).passed

    def test_training_only_user_statistics_pass(self, split_frame):
        statistics = build_user_statistics(split_frame, build_item_popularity(split_frame))
        assert check_user_statistics_are_training_only(statistics, split_frame).passed


class TestMappingCheck:
    def test_detects_unmapped_ids(self, split_frame):
        broken = split_frame.copy()
        broken.loc[broken.index[0], "internal_item_id"] = pd.NA
        assert not check_mapping_covers_all_splits(broken).passed

    def test_detects_sentinel_ids(self, split_frame):
        broken = split_frame.copy()
        broken.loc[broken.index[0], "internal_user_id"] = -1
        assert not check_mapping_covers_all_splits(broken).passed

    def test_consistent_mapping_passes(self, split_frame):
        assert check_mapping_covers_all_splits(split_frame).passed


class TestFeatureTableCheck:
    def test_detects_a_split_column_in_a_feature_table(self, split_frame):
        check = check_split_labels_absent_from_features({"leaky": split_frame})
        assert not check.passed
        assert "split" in check.evidence["offenders"]["leaky"]

    def test_detects_a_target_column(self):
        frame = pd.DataFrame({"internal_user_id": [0], "target_item": [1]})
        assert not check_split_labels_absent_from_features({"leaky": frame}).passed

    def test_clean_feature_tables_pass(self, split_frame):
        clean = build_item_popularity(split_frame)
        assert check_split_labels_absent_from_features({"popularity": clean}).passed


class TestColdItemWarning:
    def test_cold_items_are_a_warning_not_a_failure(self, split_frame):
        """An item first seen at test time is real cold start, not a bug."""
        check = check_no_future_items_in_training(split_frame)
        assert check.severity is Severity.WARNING
        assert check.evidence["cold_item_count"] > 0

    def test_warnings_do_not_fail_the_build(self, split_frame):
        report = run_all_checks(split_frame)
        assert report.warnings
        assert report.passed


class TestReport:
    def test_critical_failure_aborts_the_build(self, split_frame):
        leaked = split_frame.copy()
        leaked.loc[leaked.index[0], "interaction_order"] = 99
        report = run_all_checks(leaked)
        assert not report.passed
        with pytest.raises(DataError) as exc:
            report.raise_if_failed()
        assert "critical leakage check" in str(exc.value)

    def test_report_payload_is_complete(self, split_frame):
        payload = run_all_checks(split_frame).to_dict()
        assert set(payload) >= {
            "passed",
            "total_checks",
            "passed_checks",
            "critical_failures",
            "warnings",
            "checks",
        }

    def test_every_check_has_a_stable_identifier(self, split_frame):
        payload = run_all_checks(split_frame).to_dict()
        identifiers = [check["check_id"] for check in payload["checks"]]
        assert len(identifiers) == len(set(identifiers))
        assert all(identifier.startswith("L") for identifier in identifiers)
