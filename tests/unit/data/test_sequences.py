"""Sequential example construction and its leakage guarantees."""

from __future__ import annotations

import pandas as pd

from omnirank.data.sequences import SEQUENCE_COLUMNS, build_all_sequences, build_sequences


class TestConstruction:
    def test_history_is_everything_strictly_before_the_target(self, split_frame):
        built, _ = build_sequences(split_frame, split="test", maximum_length=10, minimum_length=1)
        user_zero = built[built.internal_user_id == 0].iloc[0]
        assert user_zero["target_item"] == 14
        assert user_zero["item_sequence"] == [10, 11, 12, 13]

    def test_history_crosses_split_boundaries(self, split_frame):
        """A test target's history legitimately includes train and validation rows."""
        built, _ = build_sequences(split_frame, split="test", maximum_length=10, minimum_length=1)
        user_zero = built[built.internal_user_id == 0].iloc[0]
        assert 13 in user_zero["item_sequence"]  # the validation item

    def test_target_is_never_inside_the_input(self, split_frame):
        for split in ("train", "validation", "test"):
            built, _ = build_sequences(
                split_frame, split=split, maximum_length=10, minimum_length=1
            )
            for _, row in built.iterrows():
                assert row["target_item"] not in row["item_sequence"]

    def test_orders_are_all_before_the_target(self, split_frame):
        built, _ = build_sequences(split_frame, split="test", maximum_length=10, minimum_length=1)
        for _, row in built.iterrows():
            assert all(order < row["target_order"] for order in row["interaction_order_sequence"])

    def test_sequence_length_matches_the_sequence(self, split_frame):
        built, _ = build_sequences(split_frame, split="test", maximum_length=10, minimum_length=1)
        assert (built["sequence_length"] == built["item_sequence"].apply(len)).all()

    def test_item_and_order_sequences_are_aligned(self, split_frame):
        built, _ = build_sequences(split_frame, split="test", maximum_length=10, minimum_length=1)
        for _, row in built.iterrows():
            assert len(row["item_sequence"]) == len(row["interaction_order_sequence"])

    def test_columns_match_the_contract(self, split_frame):
        built, _ = build_sequences(split_frame, split="test", maximum_length=10, minimum_length=1)
        assert list(built.columns) == list(SEQUENCE_COLUMNS)


class TestTruncation:
    def test_oldest_events_are_dropped_first(self):
        """A self-attentive model attends to the recent tail, so keep it."""
        frame = pd.DataFrame(
            {
                "internal_user_id": [0] * 6,
                "internal_item_id": [10, 11, 12, 13, 14, 15],
                "interaction_order": [0, 1, 2, 3, 4, 5],
                "split": ["train"] * 5 + ["test"],
            }
        )
        built, stats = build_sequences(frame, split="test", maximum_length=3, minimum_length=1)
        assert built.iloc[0]["item_sequence"] == [12, 13, 14]
        assert stats.truncated == 1

    def test_short_histories_are_not_truncated(self, split_frame):
        _, stats = build_sequences(split_frame, split="test", maximum_length=100, minimum_length=1)
        assert stats.truncated == 0


class TestMinimumLength:
    def test_users_below_the_minimum_are_skipped(self, split_frame):
        built, stats = build_sequences(
            split_frame, split="test", maximum_length=10, minimum_length=3
        )
        # user 1 has a 2-event history before its test target; user 0 has 4.
        assert set(built["internal_user_id"]) == {0}
        assert stats.skipped_short_history == 1

    def test_skips_are_counted_not_hidden(self, split_frame):
        _, stats = build_sequences(split_frame, split="train", maximum_length=10, minimum_length=2)
        assert stats.skipped_short_history > 0


class TestAllSplits:
    def test_builds_all_three(self, split_frame):
        frames, stats = build_all_sequences(split_frame, maximum_length=10, minimum_length=1)
        assert set(frames) == {"train", "validation", "test"}
        assert len(stats) == 3

    def test_one_example_per_target(self, split_frame):
        frames, _ = build_all_sequences(split_frame, maximum_length=10, minimum_length=1)
        expected = (split_frame["split"] == "test").sum()
        assert len(frames["test"]) == expected

    def test_empty_input_yields_typed_empty_frames(self):
        frames, stats = build_all_sequences(pd.DataFrame(), maximum_length=10, minimum_length=1)
        assert all(frame.empty for frame in frames.values())
        assert all(stat.examples == 0 for stat in stats)


class TestDeterminism:
    def test_repeated_builds_agree(self, split_frame):
        first, _ = build_sequences(split_frame, split="test", maximum_length=10, minimum_length=1)
        second, _ = build_sequences(split_frame, split="test", maximum_length=10, minimum_length=1)
        assert first.equals(second)

    def test_row_order_does_not_matter(self, split_frame):
        forward, _ = build_sequences(split_frame, split="test", maximum_length=10, minimum_length=1)
        shuffled_input = split_frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
        shuffled, _ = build_sequences(
            shuffled_input, split="test", maximum_length=10, minimum_length=1
        )
        assert (
            forward.sort_values("internal_user_id")
            .reset_index(drop=True)
            .equals(shuffled.sort_values("internal_user_id").reset_index(drop=True))
        )
