"""Two-tower training examples built from a rolling fold.

The property that matters is that a fold's examples come from that fold's
history and nowhere else. If ``sequences_from_fold`` silently reached past the
fold origin, every fold would train on the same data and the rolling summary
would report agreement it never measured.
"""

from __future__ import annotations

import pandas as pd
import pytest

from omnirank.data.rolling import build_fold
from omnirank.retrieval.runner import sequences_from_fold


def log(histories: dict[int, int]) -> pd.DataFrame:
    """Build an interaction log: user -> number of events, items unique."""
    rows = []
    item = 0
    for user, count in histories.items():
        for order in range(count):
            rows.append(
                {"internal_user_id": user, "interaction_order": order, "internal_item_id": item}
            )
            item += 1
    return pd.DataFrame(rows)


class TestFoldSequences:
    def test_target_is_the_fold_target(self) -> None:
        fold = build_fold(log({1: 6}), offset=3)
        frame = sequences_from_fold(fold)
        expected = fold.targets.loc[fold.targets.internal_user_id == 1, "internal_item_id"].item()
        assert frame.loc[0, "target_item"] == expected

    def test_history_stops_at_the_fold_origin(self) -> None:
        """Nothing at or after the target may appear in the input sequence."""
        fold = build_fold(log({1: 8}), offset=3)
        frame = sequences_from_fold(fold)
        sequence = frame.loc[0, "item_sequence"]
        target = frame.loc[0, "target_item"]
        # Items are numbered in chronological order, so "before the origin"
        # is checkable arithmetically rather than by trusting the fold.
        assert max(sequence) < target
        assert len(sequence) == 5

    def test_offsets_produce_different_training_data(self) -> None:
        """A later origin must see strictly more history than an earlier one."""
        interactions = log({1: 8, 2: 7})
        shallow = sequences_from_fold(build_fold(interactions, offset=3))
        deep = sequences_from_fold(build_fold(interactions, offset=2))
        shallow_lengths = [len(row) for row in shallow["item_sequence"]]
        deep_lengths = [len(row) for row in deep["item_sequence"]]
        assert deep_lengths == [length + 1 for length in shallow_lengths]
        assert list(deep["target_item"]) != list(shallow["target_item"])

    def test_history_is_truncated_to_the_most_recent_events(self) -> None:
        fold = build_fold(log({1: 30}), offset=3)
        frame = sequences_from_fold(fold, maximum_history_length=5)
        sequence = frame.loc[0, "item_sequence"]
        assert len(sequence) == 5
        # The *recent* five, not the first five.
        assert sequence == sorted(sequence)
        assert min(sequence) > 20

    def test_users_without_history_are_dropped(self) -> None:
        """A user whose whole log sits at or after the origin has no example."""
        fold = build_fold(log({1: 6, 2: 3}), offset=3)
        frame = sequences_from_fold(fold)
        assert list(frame["internal_user_id"]) == [1]

    @pytest.mark.parametrize("offset", [2, 3])
    def test_every_row_is_a_usable_example(self, offset: int) -> None:
        frame = sequences_from_fold(build_fold(log({1: 9, 2: 8, 3: 7}), offset=offset))
        assert not frame.empty
        for sequence, target in zip(frame["item_sequence"], frame["target_item"], strict=True):
            assert sequence, "an empty history is not a trainable example"
            assert target not in sequence, "target leaked into its own input"
