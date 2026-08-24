"""Sequential example construction - component 7.

One example per (user, evaluation target): the user's chronologically ordered
history *strictly before* the target, plus the target itself.

Three rules make these examples trustworthy, and each corresponds to a leakage
check in :mod:`omnirank.data.leakage`:

* **History is strictly past.** Every element has ``interaction_order`` less
  than the target's.
* **The target is never inside the input.** Otherwise the model learns to copy.
* **Truncation drops the oldest events.** When a history exceeds
  ``maximum_length``, the most recent events are kept - they are what a
  self-attentive model actually attends to.

Sequences are stored variable-length. Padding is a training-time concern and
belongs to the collate function that knows the model's expected shape; baking a
padding value into the dataset would fix a decision no model has made yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from omnirank.core.logging import get_logger
from omnirank.data.splitters import TEST, TRAIN, VALIDATION

logger = get_logger(__name__)

SEQUENCE_COLUMNS: Final = (
    "internal_user_id",
    "item_sequence",
    "interaction_order_sequence",
    "sequence_length",
    "target_item",
    "target_order",
    "split",
)


@dataclass(slots=True)
class SequenceBuildStats:
    """What sequence construction produced and skipped."""

    split: str
    examples: int
    users: int
    skipped_short_history: int
    truncated: int
    mean_length: float
    max_length_configured: int
    min_length_configured: int

    def to_dict(self) -> dict[str, Any]:
        """Report-ready description."""
        return {
            "split": self.split,
            "examples": self.examples,
            "users": self.users,
            "skipped_short_history": self.skipped_short_history,
            "truncated": self.truncated,
            "mean_length": round(self.mean_length, 3),
            "max_length_configured": self.max_length_configured,
            "min_length_configured": self.min_length_configured,
        }


def _histories(frame: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-user (items, orders) arrays, chronologically sorted."""
    ordered = frame.sort_values(["internal_user_id", "interaction_order"], kind="mergesort")
    grouped = ordered.groupby("internal_user_id", observed=True)
    return {
        int(cast("int", user)): (
            group["internal_item_id"].to_numpy(dtype="int64"),
            group["interaction_order"].to_numpy(dtype="int64"),
        )
        for user, group in grouped
    }


def build_sequences(
    frame: pd.DataFrame,
    *,
    split: str,
    maximum_length: int,
    minimum_length: int,
) -> tuple[pd.DataFrame, SequenceBuildStats]:
    """Build sequential examples for one split.

    Args:
        frame: The full split-labelled interaction log, with internal ids and
            ``interaction_order``. The *whole* log is required, not just the
            target split, because a validation target's history lives in train.
        split: Which split's targets to build examples for.
        maximum_length: Longest history to keep; older events are dropped.
        minimum_length: Shortest history that yields an example. A user with
            less contributes no example rather than a degenerate one.

    Returns:
        The examples frame and its build statistics.

    For ``split="train"`` the targets are the training events themselves, each
    predicted from the events before it - the standard next-item objective. For
    validation and test the targets are the held-out events, and the history is
    everything strictly earlier regardless of which split that history sits in.
    """
    if frame.empty:
        return pd.DataFrame(columns=list(SEQUENCE_COLUMNS)), SequenceBuildStats(
            split=split,
            examples=0,
            users=0,
            skipped_short_history=0,
            truncated=0,
            mean_length=0.0,
            max_length_configured=maximum_length,
            min_length_configured=minimum_length,
        )

    histories = _histories(frame)
    targets = frame[frame["split"] == split]

    users: list[int] = []
    item_sequences: list[list[int]] = []
    order_sequences: list[list[int]] = []
    lengths: list[int] = []
    target_items: list[int] = []
    target_orders: list[int] = []
    skipped = 0
    truncated = 0

    for user, target_item, target_order in zip(
        targets["internal_user_id"].to_numpy(dtype="int64"),
        targets["internal_item_id"].to_numpy(dtype="int64"),
        targets["interaction_order"].to_numpy(dtype="int64"),
        strict=True,
    ):
        items, orders = histories[int(user)]
        # Strictly-less-than is the whole leakage guarantee for this stage.
        past = orders < target_order
        history_items = items[past]
        history_orders = orders[past]

        if len(history_items) < minimum_length:
            skipped += 1
            continue
        if len(history_items) > maximum_length:
            history_items = history_items[-maximum_length:]
            history_orders = history_orders[-maximum_length:]
            truncated += 1

        users.append(int(user))
        item_sequences.append(history_items.tolist())
        order_sequences.append(history_orders.tolist())
        lengths.append(len(history_items))
        target_items.append(int(target_item))
        target_orders.append(int(target_order))

    built = pd.DataFrame(
        {
            "internal_user_id": pd.Series(users, dtype="int64"),
            "item_sequence": item_sequences,
            "interaction_order_sequence": order_sequences,
            "sequence_length": pd.Series(lengths, dtype="int64"),
            "target_item": pd.Series(target_items, dtype="int64"),
            "target_order": pd.Series(target_orders, dtype="int64"),
            "split": split,
        }
    )
    stats = SequenceBuildStats(
        split=split,
        examples=len(built),
        users=int(built["internal_user_id"].nunique()) if len(built) else 0,
        skipped_short_history=skipped,
        truncated=truncated,
        mean_length=float(np.mean(lengths)) if lengths else 0.0,
        max_length_configured=maximum_length,
        min_length_configured=minimum_length,
    )
    logger.info("sequences.built", **stats.to_dict())
    return built.loc[:, list(SEQUENCE_COLUMNS)], stats


def build_all_sequences(
    frame: pd.DataFrame, *, maximum_length: int, minimum_length: int
) -> tuple[dict[str, pd.DataFrame], list[SequenceBuildStats]]:
    """Build sequential examples for train, validation, and test."""
    frames: dict[str, pd.DataFrame] = {}
    stats: list[SequenceBuildStats] = []
    for split in (TRAIN, VALIDATION, TEST):
        built, split_stats = build_sequences(
            frame, split=split, maximum_length=maximum_length, minimum_length=minimum_length
        )
        frames[split] = built
        stats.append(split_stats)
    return frames, stats


__all__ = ["SEQUENCE_COLUMNS", "SequenceBuildStats", "build_all_sequences", "build_sequences"]
