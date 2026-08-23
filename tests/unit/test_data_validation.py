"""Batch validation: every rule in ValidationRule, plus report semantics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from omnirank.core.exceptions import SchemaValidationError
from omnirank.data.validation import (
    ValidationRule,
    validate_batch,
    validate_interactions,
    validate_items,
    validate_users,
)
from tests.conftest import FROZEN_NOW


@pytest.fixture
def data_config(config):
    return config.data


def _rules(report) -> set[str]:
    return set(report.counts_by_rule)


class TestUsers:
    def test_valid_batch_passes(self, data_config, user_row):
        valid, report = validate_users([user_row], data_config, now=FROZEN_NOW)
        assert len(valid) == 1
        assert report.ok

    def test_missing_id_is_reported_not_raised(self, data_config):
        valid, report = validate_users(
            [{"created_at": "2026-01-01T00:00:00Z"}], data_config, now=FROZEN_NOW
        )
        assert valid == []
        assert ValidationRule.MISSING_ID.value in _rules(report)

    def test_duplicate_ids_keep_the_first(self, data_config, user_row):
        valid, report = validate_users([user_row, user_row], data_config, now=FROZEN_NOW)
        assert len(valid) == 1
        assert ValidationRule.DUPLICATE_ENTITY_ID.value in _rules(report)

    def test_timestamp_before_the_configured_floor_is_rejected(self, data_config):
        valid, report = validate_users(
            [{"user_id": "u1", "created_at": "1970-01-01T00:00:00Z"}],
            data_config,
            now=FROZEN_NOW,
        )
        assert valid == []
        assert ValidationRule.INVALID_TIMESTAMP.value in _rules(report)

    def test_future_timestamp_is_rejected(self, data_config):
        future = (FROZEN_NOW + timedelta(days=365)).isoformat()
        valid, report = validate_users(
            [{"user_id": "u1", "created_at": future}], data_config, now=FROZEN_NOW
        )
        assert valid == []
        assert ValidationRule.FUTURE_TIMESTAMP.value in _rules(report)


class TestItems:
    def test_price_above_the_configured_ceiling_is_rejected(self, data_config, item_row):
        valid, report = validate_items(
            [{**item_row, "price": 10_000_000.0}], data_config, now=FROZEN_NOW
        )
        assert valid == []
        assert ValidationRule.INVALID_PRICE.value in _rules(report)

    def test_negative_price_is_rejected_by_the_record_schema(self, data_config, item_row):
        valid, report = validate_items([{**item_row, "price": -5.0}], data_config, now=FROZEN_NOW)
        assert valid == []
        assert ValidationRule.INVALID_PRICE.value in _rules(report)

    def test_invalid_availability_is_reported_as_such(self, data_config, item_row):
        valid, report = validate_items(
            [{**item_row, "available": "sometimes"}], data_config, now=FROZEN_NOW
        )
        assert valid == []
        assert ValidationRule.INVALID_AVAILABILITY.value in _rules(report)


class TestInteractions:
    def test_valid_batch_passes(self, data_config, interaction_row):
        valid, report = validate_interactions([interaction_row], data_config, now=FROZEN_NOW)
        assert len(valid) == 1
        assert report.ok

    def test_unknown_event_type_is_rejected(self, data_config, interaction_row):
        valid, report = validate_interactions(
            [{**interaction_row, "event_type": "levitate"}], data_config, now=FROZEN_NOW
        )
        assert valid == []
        assert ValidationRule.UNKNOWN_EVENT_TYPE.value in _rules(report)

    def test_duplicate_events_are_dropped(self, data_config, interaction_row):
        resent = {**interaction_row, "interaction_id": "e2"}
        valid, report = validate_interactions(
            [interaction_row, resent], data_config, now=FROZEN_NOW
        )
        assert len(valid) == 1
        assert ValidationRule.DUPLICATE_EVENT.value in _rules(report)

    def test_unknown_user_reference_is_rejected(self, data_config, interaction_row):
        valid, report = validate_interactions(
            [interaction_row],
            data_config,
            known_user_ids=set(),
            known_item_ids={"i1"},
            now=FROZEN_NOW,
        )
        assert valid == []
        assert ValidationRule.UNKNOWN_USER_REFERENCE.value in _rules(report)

    def test_unknown_item_reference_is_rejected(self, data_config, interaction_row):
        valid, report = validate_interactions(
            [interaction_row],
            data_config,
            known_user_ids={"u1"},
            known_item_ids=set(),
            now=FROZEN_NOW,
        )
        assert valid == []
        assert ValidationRule.UNKNOWN_ITEM_REFERENCE.value in _rules(report)

    def test_reference_checking_is_skipped_when_not_supplied(self, data_config, interaction_row):
        valid, _ = validate_interactions([interaction_row], data_config, now=FROZEN_NOW)
        assert len(valid) == 1

    def test_out_of_range_rating_is_rejected(self, data_config, interaction_row):
        valid, report = validate_interactions(
            [{**interaction_row, "event_type": "rating", "event_value": 99.0}],
            data_config,
            now=FROZEN_NOW,
        )
        assert valid == []
        assert ValidationRule.INVALID_RATING.value in _rules(report)

    def test_in_range_rating_is_accepted(self, data_config, interaction_row):
        valid, _ = validate_interactions(
            [{**interaction_row, "event_type": "rating", "event_value": 4.0}],
            data_config,
            now=FROZEN_NOW,
        )
        assert len(valid) == 1


class TestReport:
    def test_counts_and_rate(self, data_config, interaction_row):
        rows = [
            interaction_row,
            {**interaction_row, "interaction_id": "e2", "event_type": "levitate"},
        ]
        _, report = validate_interactions(rows, data_config, now=FROZEN_NOW)
        assert report.total == 2
        assert report.valid == 1
        assert report.rejected == 1
        assert report.rejection_rate == 0.5

    def test_empty_batch_has_zero_rate_not_a_division_error(self, data_config):
        _, report = validate_interactions([], data_config, now=FROZEN_NOW)
        assert report.rejection_rate == 0.0
        assert report.ok

    def test_summary_contains_counts_only(self, data_config, interaction_row):
        _, report = validate_interactions(
            [{**interaction_row, "event_type": "levitate"}], data_config, now=FROZEN_NOW
        )
        summary = report.summary()
        assert set(summary) == {
            "entity",
            "total",
            "valid",
            "rejected",
            "rejection_rate",
            "counts_by_rule",
        }
        # No record content, so a summary is always safe to log.
        assert "u1" not in str(summary)

    def test_raise_if_failed_is_a_no_op_when_clean(self, data_config, interaction_row):
        _, report = validate_interactions([interaction_row], data_config, now=FROZEN_NOW)
        report.raise_if_failed()

    def test_raise_if_failed_raises_with_rule_counts(self, data_config, interaction_row):
        _, report = validate_interactions(
            [{**interaction_row, "event_type": "levitate"}], data_config, now=FROZEN_NOW
        )
        with pytest.raises(SchemaValidationError) as exc:
            report.raise_if_failed()
        assert "unknown_event_type" in str(exc.value)


class TestValidateBatch:
    def test_resolves_references_across_entities(
        self, data_config, user_row, item_row, interaction_row
    ):
        batch = validate_batch(
            [user_row], [item_row], [interaction_row], data_config, now=FROZEN_NOW
        )
        assert batch.ok
        assert len(batch.interactions) == 1

    def test_interaction_to_a_rejected_item_is_dropped(
        self, data_config, user_row, item_row, interaction_row
    ):
        bad_item = {**item_row, "price": -1.0}
        batch = validate_batch(
            [user_row], [bad_item], [interaction_row], data_config, now=FROZEN_NOW
        )
        assert batch.items == ()
        assert batch.interactions == ()
        assert not batch.ok

    def test_strict_mode_raises(self, data_config, user_row, item_row):
        with pytest.raises(SchemaValidationError):
            validate_batch(
                [user_row],
                [item_row],
                [{"interaction_id": "e1"}],
                data_config,
                strict=True,
                now=FROZEN_NOW,
            )

    def test_reference_checking_can_be_disabled(self, data_config, item_row, interaction_row):
        batch = validate_batch(
            [],
            [item_row],
            [interaction_row],
            data_config,
            check_references=False,
            now=FROZEN_NOW,
        )
        assert len(batch.interactions) == 1

    def test_now_is_injectable_so_the_suite_never_reads_the_clock(
        self, data_config, user_row, item_row
    ):
        event = {
            "interaction_id": "e1",
            "user_id": "u1",
            "item_id": "i1",
            "event_type": "click",
            "timestamp": datetime(2030, 1, 1, tzinfo=UTC).isoformat(),
        }
        rejected = validate_batch([user_row], [item_row], [event], data_config, now=FROZEN_NOW)
        assert rejected.interactions == ()

        later = datetime(2031, 1, 1, tzinfo=UTC)
        accepted = validate_batch([user_row], [item_row], [event], data_config, now=later)
        assert len(accepted.interactions) == 1
