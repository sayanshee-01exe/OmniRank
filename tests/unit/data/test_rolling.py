"""Rolling-origin validation folds.

The single most important property is negative: **the official test target must
never enter a selection fold**. Everything else here supports that claim.
"""

from __future__ import annotations

import pandas as pd
import pytest

from omnirank.core.exceptions import DataError
from omnirank.data.rolling import (
    DEFAULT_TARGET_OFFSETS,
    FOLD_EXCLUDED,
    FOLD_HISTORY,
    FOLD_TARGET,
    RESERVED_TEST_OFFSET,
    build_fold,
    build_rolling_validation,
    check_fold_integrity,
    check_no_reserved_offset_used,
)


def log(histories: dict[str, int]) -> pd.DataFrame:
    """Build an interaction log: user -> number of events."""
    rows = []
    item = 0
    for user, count in histories.items():
        for order in range(count):
            rows.append(
                {"internal_user_id": user, "interaction_order": order, "internal_item_id": item}
            )
            item += 1
    return pd.DataFrame(rows)


def roles(fold, user: str) -> list[str]:
    """Fold roles for one user, in chronological order."""
    subset = fold.interactions[fold.interactions.internal_user_id == user]
    return subset.sort_values("interaction_order")["fold_role"].tolist()


class TestReservedTestOffset:
    def test_offset_one_is_refused(self):
        """Using it for selection would tune against the final benchmark."""
        with pytest.raises(DataError) as exc:
            build_fold(log({"a": 5}), offset=RESERVED_TEST_OFFSET)
        assert "reserved" in str(exc.value).lower()

    def test_rolling_validation_refuses_the_reserved_offset(self):
        with pytest.raises(DataError):
            build_rolling_validation(log({"a": 5}), target_offsets=(2, 1))

    def test_last_interaction_is_always_excluded(self):
        """The official test target must be invisible in every selection fold."""
        for offset in (2, 3, 4):
            fold = build_fold(log({"a": 8, "b": 6}), offset=offset)
            for user in ("a", "b"):
                assert roles(fold, user)[-1] == FOLD_EXCLUDED

    def test_protection_check_passes_on_valid_folds(self):
        validation = build_rolling_validation(log({"a": 8, "b": 6}), target_offsets=(3, 2))
        check_no_reserved_offset_used(validation)

    def test_default_offsets_exclude_the_reserved_one(self):
        assert RESERVED_TEST_OFFSET not in DEFAULT_TARGET_OFFSETS


class TestFoldConstruction:
    def test_offset_two_targets_the_second_to_last(self):
        fold = build_fold(log({"a": 6}), offset=2)
        assert roles(fold, "a") == [
            FOLD_HISTORY,
            FOLD_HISTORY,
            FOLD_HISTORY,
            FOLD_HISTORY,
            FOLD_TARGET,
            FOLD_EXCLUDED,
        ]

    def test_offset_three_targets_the_third_to_last(self):
        fold = build_fold(log({"a": 6}), offset=3)
        assert roles(fold, "a") == [
            FOLD_HISTORY,
            FOLD_HISTORY,
            FOLD_HISTORY,
            FOLD_TARGET,
            FOLD_EXCLUDED,
            FOLD_EXCLUDED,
        ]

    def test_exactly_one_target_per_eligible_user(self):
        fold = build_fold(log({"a": 6, "b": 5}), offset=2)
        counts = fold.targets.groupby("internal_user_id").size()
        assert set(counts.unique()) == {1}

    def test_history_strictly_precedes_the_target(self):
        fold = build_fold(log({"a": 8, "b": 6}), offset=3)
        check_fold_integrity(fold)

    def test_nothing_after_the_target_is_history(self):
        fold = build_fold(log({"a": 8}), offset=3)
        assignments = roles(fold, "a")
        target_index = assignments.index(FOLD_TARGET)
        assert all(role == FOLD_EXCLUDED for role in assignments[target_index + 1 :])

    def test_invalid_offset_rejected(self):
        with pytest.raises(DataError):
            build_fold(log({"a": 5}), offset=0)

    def test_empty_log_rejected(self):
        with pytest.raises(DataError):
            build_fold(pd.DataFrame(), offset=2)

    def test_missing_columns_rejected(self):
        with pytest.raises(DataError):
            build_fold(pd.DataFrame({"internal_user_id": ["a"]}), offset=2)


class TestEligibility:
    def test_user_without_enough_history_is_excluded_from_evaluation(self):
        """Two events cannot support an offset-2 fold: nothing precedes the target."""
        fold = build_fold(log({"a": 6, "short": 2}), offset=2)
        assert "short" not in set(fold.targets["internal_user_id"])
        assert fold.excluded_users == 1

    def test_minimum_history_is_configurable(self):
        relaxed = build_fold(log({"a": 3}), offset=2, minimum_history=1)
        strict = build_fold(log({"a": 3}), offset=2, minimum_history=2)
        assert relaxed.eligible_users == 1
        assert strict.eligible_users == 0

    def test_excluded_users_are_counted_not_silently_dropped(self):
        fold = build_fold(log({"a": 6, "b": 2, "c": 1}), offset=2)
        assert fold.eligible_users + fold.excluded_users == 3


class TestDeterminism:
    def test_repeated_builds_agree(self):
        data = log({"a": 7, "b": 5})
        assert build_fold(data, offset=2).checksum == build_fold(data, offset=2).checksum

    def test_row_order_does_not_matter(self):
        data = log({"a": 7, "b": 5})
        forward = build_fold(data, offset=2)
        shuffled = build_fold(data.sample(frac=1.0, random_state=3), offset=2)
        assert forward.checksum == shuffled.checksum

    def test_different_offsets_have_different_checksums(self):
        data = log({"a": 7, "b": 5})
        assert build_fold(data, offset=2).checksum != build_fold(data, offset=3).checksum


class TestManifest:
    def test_records_every_fold(self):
        manifest = build_rolling_validation(log({"a": 8}), target_offsets=(3, 2)).manifest()
        assert [item["target_offset"] for item in manifest["folds"]] == [3, 2]

    def test_records_the_reserved_offset(self):
        manifest = build_rolling_validation(log({"a": 8}), target_offsets=(2,)).manifest()
        assert manifest["reserved_test_offset"] == RESERVED_TEST_OFFSET

    def test_every_fold_carries_a_checksum(self):
        manifest = build_rolling_validation(log({"a": 8}), target_offsets=(3, 2)).manifest()
        assert all(item["checksum"] for item in manifest["folds"])

    def test_duplicate_offsets_rejected(self):
        with pytest.raises(DataError):
            build_rolling_validation(log({"a": 8}), target_offsets=(2, 2))

    def test_empty_offsets_rejected(self):
        with pytest.raises(DataError):
            build_rolling_validation(log({"a": 8}), target_offsets=())

    def test_fold_lookup_by_offset(self):
        validation = build_rolling_validation(log({"a": 8}), target_offsets=(3, 2))
        assert validation.fold(3).offset == 3
        with pytest.raises(DataError):
            validation.fold(9)


class TestNoFutureLeakage:
    def test_fold_history_never_contains_a_later_interaction(self):
        fold = build_fold(log({"a": 10, "b": 8}), offset=3)
        for user in ("a", "b"):
            subset = fold.interactions[fold.interactions.internal_user_id == user]
            history_max = subset.loc[subset.fold_role == FOLD_HISTORY, "interaction_order"].max()
            target = subset.loc[subset.fold_role == FOLD_TARGET, "interaction_order"].min()
            assert history_max < target

    def test_target_item_is_not_in_the_fold_history(self):
        """PixelRec has no repeated pairs, so any overlap means a build error."""
        check_fold_integrity(build_fold(log({"a": 10}), offset=2))

    def test_integrity_check_catches_a_corrupted_fold(self):
        fold = build_fold(log({"a": 6}), offset=2)
        corrupted = fold.interactions.copy()
        # Promote a post-target row to history: the leak the check exists for.
        last = corrupted.index[-1]
        corrupted.loc[last, "fold_role"] = FOLD_HISTORY
        fold.interactions = corrupted
        with pytest.raises(DataError) as exc:
            check_fold_integrity(fold)
        assert "precede" in str(exc.value)
