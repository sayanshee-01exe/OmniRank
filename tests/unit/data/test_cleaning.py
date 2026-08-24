"""Cleaning: rejection rules, reconciliation, and the rejected-records trail."""

from __future__ import annotations

import pandas as pd
import pytest

from omnirank.core.exceptions import DataError
from omnirank.data.cleaning import (
    REJECTED_COLUMNS,
    CleaningStep,
    RejectedRecords,
    RejectionReason,
    clean_interactions,
    clean_items,
)

MIN_TS = pd.Timestamp("2010-01-01", tz="UTC")
MAX_TS = pd.Timestamp("2030-01-01", tz="UTC")


def make_interactions(rows: list[dict]) -> pd.DataFrame:
    """Build a canonical interaction frame from partial row dicts."""
    defaults = {
        "interaction_id": "e0",
        "external_user_id": "u1",
        "external_item_id": "i1",
        "event_type": "interaction",
        "timestamp": 1_640_995_200,
        "interaction_weight": 1.0,
        "source_row_id": 0,
    }
    frame = pd.DataFrame([{**defaults, **row} for row in rows])
    frame["timestamp"] = frame["timestamp"].astype("Int64")
    frame["event_timestamp_utc"] = pd.to_datetime(
        frame["timestamp"].astype("float"), unit="s", utc=True
    )
    return frame


def clean(rows, known=("i1",), **kwargs: object):
    sink = RejectedRecords()
    frame, step = clean_interactions(
        make_interactions(rows),
        set(known),
        sink,
        min_timestamp=MIN_TS,
        max_timestamp=MAX_TS,
        allowed_event_types={"interaction"},
        **kwargs,
    )
    return frame, step, sink


class TestReconciliation:
    def test_a_balanced_step_passes(self):
        CleaningStep("s", input_rows=10, output_rows=8, reason_counts={"a": 2}).check()

    def test_unexplained_removals_fail(self):
        """Rows vanishing without a recorded reason is the bug this catches."""
        with pytest.raises(DataError) as exc:
            CleaningStep("s", input_rows=10, output_rows=8).check()
        assert "every removal must be explained" in str(exc.value)

    def test_gaining_rows_fails(self):
        with pytest.raises(DataError) as exc:
            CleaningStep("s", input_rows=5, output_rows=9).check()
        assert "fanned out" in str(exc.value)

    def test_real_cleaning_always_reconciles(self):
        _, step, _ = clean(
            [
                {"source_row_id": 0},
                {"source_row_id": 1, "external_user_id": ""},
                {"source_row_id": 2, "external_item_id": "i999"},
                {"source_row_id": 3, "timestamp": -5},
            ]
        )
        step.check()
        assert step.removed_rows == 3


class TestInteractionRules:
    def test_clean_rows_survive(self):
        frame, _step, sink = clean([{"source_row_id": 0}])
        assert len(frame) == 1
        assert sink.count == 0

    @pytest.mark.parametrize(
        ("row", "reason"),
        [
            ({"external_user_id": ""}, RejectionReason.MISSING_USER_ID),
            ({"external_user_id": None}, RejectionReason.MISSING_USER_ID),
            ({"external_item_id": ""}, RejectionReason.MISSING_ITEM_ID),
            ({"timestamp": 0}, RejectionReason.INVALID_TIMESTAMP),
            ({"timestamp": -100}, RejectionReason.INVALID_TIMESTAMP),
            ({"timestamp": 100}, RejectionReason.TIMESTAMP_OUT_OF_RANGE),
            ({"timestamp": 2_000_000_000}, RejectionReason.FUTURE_TIMESTAMP),
            ({"event_type": "purchase"}, RejectionReason.UNKNOWN_EVENT_TYPE),
            ({"interaction_weight": -1.0}, RejectionReason.INVALID_WEIGHT),
            ({"interaction_weight": float("nan")}, RejectionReason.INVALID_WEIGHT),
            ({"interaction_weight": float("inf")}, RejectionReason.INVALID_WEIGHT),
            ({"external_item_id": "i999"}, RejectionReason.UNKNOWN_ITEM_REFERENCE),
        ],
    )
    def test_each_rule_rejects_and_names_itself(self, row, reason):
        frame, step, _ = clean([{"source_row_id": 0, **row}])
        assert frame.empty
        assert reason.value in step.reason_counts

    def test_duplicates_are_dropped_keeping_the_first(self):
        frame, step, _ = clean(
            [
                {"source_row_id": 0, "interaction_id": "a"},
                {"source_row_id": 1, "interaction_id": "b"},
            ]
        )
        assert len(frame) == 1
        assert frame.iloc[0]["interaction_id"] == "a"
        assert step.reason_counts[RejectionReason.DUPLICATE_INTERACTION.value] == 1

    def test_deduplication_can_be_disabled(self):
        frame, _, _ = clean([{"source_row_id": 0}, {"source_row_id": 1}], drop_duplicates=False)
        assert len(frame) == 2

    def test_dedup_key_ignores_the_interaction_id(self):
        """A re-emitted event with a fresh id must not be counted twice."""
        frame, _, _ = clean(
            [
                {"source_row_id": 0, "interaction_id": "x"},
                {"source_row_id": 1, "interaction_id": "y"},
            ]
        )
        assert len(frame) == 1

    def test_different_timestamps_are_not_duplicates(self):
        frame, _, _ = clean(
            [
                {"source_row_id": 0, "timestamp": 1_640_995_200},
                {"source_row_id": 1, "timestamp": 1_640_998_800},
            ]
        )
        assert len(frame) == 2

    def test_a_row_is_rejected_once_even_when_several_rules_apply(self):
        """Deduplication runs last so an already-rejected row is not recounted."""
        _, step, sink = clean([{"source_row_id": 0, "external_user_id": "", "timestamp": -1}])
        assert sink.count == 1
        assert sum(step.reason_counts.values()) == 1


class TestItemRules:
    def test_blank_item_ids_are_rejected(self):
        sink = RejectedRecords()
        frame, step = clean_items(
            pd.DataFrame({"external_item_id": ["i1", "", None], "title": ["a", "b", "c"]}), sink
        )
        assert len(frame) == 1
        assert step.reason_counts[RejectionReason.MISSING_ITEM_ID.value] == 2

    def test_duplicate_item_ids_keep_the_first(self):
        sink = RejectedRecords()
        frame, step = clean_items(
            pd.DataFrame({"external_item_id": ["i1", "i1"], "title": ["first", "second"]}), sink
        )
        assert len(frame) == 1
        assert frame.iloc[0]["title"] == "first"
        assert step.reason_counts[RejectionReason.DUPLICATE_ITEM_ID.value] == 1

    def test_missing_metadata_does_not_reject_an_item(self):
        """An item with an id is still recommendable from collaborative signal."""
        sink = RejectedRecords()
        frame, _ = clean_items(
            pd.DataFrame({"external_item_id": ["i1"], "title": [None], "description": [None]}), sink
        )
        assert len(frame) == 1


class TestRejectedRecords:
    def test_has_the_required_columns(self):
        _, _, sink = clean([{"source_row_id": 7, "external_user_id": ""}])
        assert list(sink.to_frame().columns) == list(REJECTED_COLUMNS)

    def test_points_back_at_the_source_row(self):
        _, _, sink = clean([{"source_row_id": 42, "external_item_id": "i999"}])
        row = sink.to_frame().iloc[0]
        assert row["source_row_identifier"] == "42"
        assert row["source_file"] == "interaction.csv"
        assert row["entity_type"] == "interaction"
        assert row["original_identifier"] == "i999"

    def test_empty_sink_yields_a_typed_empty_frame(self):
        assert list(RejectedRecords().to_frame().columns) == list(REJECTED_COLUMNS)

    def test_nothing_is_dropped_silently(self):
        """Every removed row must appear in the rejection trail."""
        _frame, step, sink = clean(
            [
                {"source_row_id": 0},
                {"source_row_id": 1, "external_user_id": ""},
                {"source_row_id": 2, "external_item_id": "i999"},
            ]
        )
        assert step.removed_rows == sink.count == 2
