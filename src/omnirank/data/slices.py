"""Cold-start and evaluation slices.

A single aggregate metric hides the failures that matter. A model can look
strong overall while being useless for users with three interactions, or for the
80% of the catalogue nobody has clicked. These slices exist so Phase 3 reports
per-population numbers from its very first baseline, rather than discovering the
gap after a model is chosen.

Every slice is derived from **training-only** statistics, so slice membership
never depends on the labels being predicted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd

from omnirank.core.logging import get_logger
from omnirank.data.splitters import TRAIN

logger = get_logger(__name__)

#: Sparse-user buckets, by training interaction count. Half-open, ascending,
#: with the last bucket unbounded. Chosen to separate genuinely cold users
#: (1-3) from merely light ones, because they fail for different reasons.
USER_ACTIVITY_BUCKETS: Final = (
    ("1-3", 1, 3),
    ("4-10", 4, 10),
    ("11-30", 11, 30),
    ("31+", 31, None),
)

SLICE_COLUMNS: Final = ("slice_name", "entity_type", "entity_id")


@dataclass(slots=True)
class SliceDefinition:
    """One evaluation slice and the rule that produced it."""

    name: str
    entity_type: str
    description: str
    rule: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        """Manifest-ready description."""
        return {
            "slice_name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "rule": self.rule,
            "size": self.size,
        }


def _frame(name: str, entity_type: str, ids: pd.Series) -> pd.DataFrame:
    """Build a slice membership frame."""
    return pd.DataFrame(
        {
            "slice_name": name,
            "entity_type": entity_type,
            "entity_id": pd.Series(ids, dtype="int64").reset_index(drop=True),
        }
    )


def build_user_activity_slices(
    user_statistics: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], list[SliceDefinition]]:
    """Bucket users by training interaction count."""
    frames: dict[str, pd.DataFrame] = {}
    definitions: list[SliceDefinition] = []
    if user_statistics.empty:
        return frames, definitions

    counts = user_statistics["training_interaction_count"]
    for label, low, high in USER_ACTIVITY_BUCKETS:
        mask = counts >= low
        if high is not None:
            mask &= counts <= high
        name = f"users_activity_{label}"
        members = user_statistics.loc[mask, "internal_user_id"]
        frames[name] = _frame(name, "user", members)
        definitions.append(
            SliceDefinition(
                name=name,
                entity_type="user",
                description=f"Users with {label} training interactions.",
                rule=(
                    f"training_interaction_count >= {low}"
                    + (f" and <= {high}" if high is not None else "")
                ),
                size=int(mask.sum()),
            )
        )
    return frames, definitions


def build_item_popularity_slices(
    item_popularity: pd.DataFrame, *, long_tail_quantile: float
) -> tuple[dict[str, pd.DataFrame], list[SliceDefinition]]:
    """Split the catalogue into head and long tail."""
    frames: dict[str, pd.DataFrame] = {}
    definitions: list[SliceDefinition] = []
    if item_popularity.empty:
        return frames, definitions

    for name, mask, description in (
        ("items_long_tail", item_popularity["long_tail_flag"], "Long-tail items."),
        ("items_head", ~item_popularity["long_tail_flag"], "Head (popular) items."),
    ):
        frames[name] = _frame(name, "item", item_popularity.loc[mask, "internal_item_id"])
        definitions.append(
            SliceDefinition(
                name=name,
                entity_type="item",
                description=description,
                rule=(
                    f"items outside the head accounting for {long_tail_quantile:.0%} of "
                    "training interactions, ranked by training interaction count"
                ),
                size=int(mask.sum()),
            )
        )
    return frames, definitions


def build_cold_item_slice(
    frame: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], list[SliceDefinition]]:
    """Items evaluated in validation or test but never seen in training.

    Genuine new-item cold start, produced naturally by the split. This is the
    population content and multimodal features exist to serve, and the ceiling
    on what any purely collaborative model can achieve.
    """
    train_items = set(frame.loc[frame["split"] == TRAIN, "internal_item_id"].unique())
    held_items = set(frame.loc[frame["split"] != TRAIN, "internal_item_id"].unique())
    cold = sorted(held_items - train_items)
    name = "items_cold_start"
    return (
        {name: _frame(name, "item", pd.Series(cold, dtype="int64"))},
        [
            SliceDefinition(
                name=name,
                entity_type="item",
                description="Items appearing only in validation/test, never in training.",
                rule="internal_item_id in held-out splits and not in train",
                size=len(cold),
            )
        ],
    )


def build_cold_user_slice(
    frame: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], list[SliceDefinition]]:
    """Users evaluated but absent from training.

    Under per-user leave-last-N every eligible user keeps training history by
    construction, so this slice is normally **empty** - and it is emitted empty
    rather than omitted, so a future split strategy that does produce cold users
    surfaces them without a schema change. No new-user scenario is fabricated.
    """
    train_users = set(frame.loc[frame["split"] == TRAIN, "internal_user_id"].unique())
    held_users = set(frame.loc[frame["split"] != TRAIN, "internal_user_id"].unique())
    cold = sorted(held_users - train_users)
    name = "users_cold_start"
    return (
        {name: _frame(name, "user", pd.Series(cold, dtype="int64"))},
        [
            SliceDefinition(
                name=name,
                entity_type="user",
                description=(
                    "Users appearing only in validation/test. Empty under "
                    "per-user leave-last-N by construction; not fabricated."
                ),
                rule="internal_user_id in held-out splits and not in train",
                size=len(cold),
            )
        ],
    )


def build_modality_slices(
    item_metadata: pd.DataFrame,
    text_index: pd.DataFrame,
    image_index: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], list[SliceDefinition]]:
    """Partition the catalogue by which multimodal features it actually has.

    When the feature files were not downloaded, every item lands in
    ``items_missing_both_modalities`` and the slice honestly reports zero
    coverage rather than pretending the modality exists.
    """
    frames: dict[str, pd.DataFrame] = {}
    definitions: list[SliceDefinition] = []
    if item_metadata.empty:
        return frames, definitions

    def _covered(index: pd.DataFrame, column: str) -> set[int]:
        if index.empty or column not in index.columns:
            return set()
        return set(index.loc[index[column], "internal_item_id"].astype("int64"))

    with_text = _covered(text_index, "has_text_feature")
    with_image = _covered(image_index, "has_image_feature")
    all_items = set(item_metadata["internal_item_id"].astype("int64"))

    for name, members, description in (
        (
            "items_missing_text_features",
            all_items - with_text,
            "Items with no text feature vector.",
        ),
        (
            "items_missing_image_features",
            all_items - with_image,
            "Items with no image feature vector.",
        ),
        (
            "items_missing_both_modalities",
            all_items - with_text - with_image,
            "Items with neither text nor image feature vector.",
        ),
        (
            "items_both_modalities",
            with_text & with_image,
            "Items with both text and image feature vectors.",
        ),
    ):
        frames[name] = _frame(name, "item", pd.Series(sorted(members), dtype="int64"))
        definitions.append(
            SliceDefinition(
                name=name,
                entity_type="item",
                description=description,
                rule="membership in the aligned feature index tables",
                size=len(members),
            )
        )
    return frames, definitions


def build_all_slices(
    frame: pd.DataFrame,
    *,
    user_statistics: pd.DataFrame,
    item_popularity: pd.DataFrame,
    item_metadata: pd.DataFrame,
    text_index: pd.DataFrame,
    image_index: pd.DataFrame,
    long_tail_quantile: float,
) -> tuple[dict[str, pd.DataFrame], list[SliceDefinition]]:
    """Build every evaluation slice and its manifest entry."""
    frames: dict[str, pd.DataFrame] = {}
    definitions: list[SliceDefinition] = []
    builders: tuple[Callable[[], tuple[dict[str, pd.DataFrame], list[SliceDefinition]]], ...] = (
        lambda: build_user_activity_slices(user_statistics),
        lambda: build_item_popularity_slices(
            item_popularity, long_tail_quantile=long_tail_quantile
        ),
        lambda: build_cold_item_slice(frame),
        lambda: build_cold_user_slice(frame),
        lambda: build_modality_slices(item_metadata, text_index, image_index),
    )
    for builder in builders:
        built_frames, built_definitions = builder()
        frames.update(built_frames)
        definitions.extend(built_definitions)

    logger.info(
        "slices.built",
        slices=len(frames),
        sizes={definition.name: definition.size for definition in definitions},
    )
    return frames, definitions


__all__ = [
    "SLICE_COLUMNS",
    "USER_ACTIVITY_BUCKETS",
    "SliceDefinition",
    "build_all_slices",
    "build_cold_item_slice",
    "build_cold_user_slice",
    "build_item_popularity_slices",
    "build_modality_slices",
    "build_user_activity_slices",
]
