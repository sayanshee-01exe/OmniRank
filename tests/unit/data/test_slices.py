"""Evaluation slices."""

from __future__ import annotations

from itertools import pairwise

import pandas as pd

from omnirank.data.slices import (
    USER_ACTIVITY_BUCKETS,
    build_all_slices,
    build_cold_item_slice,
    build_cold_user_slice,
    build_item_popularity_slices,
    build_modality_slices,
    build_user_activity_slices,
)
from omnirank.data.statistics import build_item_popularity, build_user_statistics


class TestUserActivity:
    def test_buckets_partition_users_exactly_once(self, split_frame):
        statistics = build_user_statistics(split_frame, build_item_popularity(split_frame))
        frames, _ = build_user_activity_slices(statistics)
        members = [tuple(frame["entity_id"]) for frame in frames.values()]
        flat = [entity for group in members for entity in group]
        assert len(flat) == len(set(flat)) == len(statistics)

    def test_bucket_boundaries_are_contiguous(self):
        """Gaps between buckets would silently drop users from every slice."""
        for (_, _, high), (_, next_low, _) in pairwise(USER_ACTIVITY_BUCKETS):
            assert high is not None and next_low == high + 1

    def test_users_land_in_the_right_bucket(self, split_frame):
        statistics = build_user_statistics(split_frame, build_item_popularity(split_frame))
        frames, _ = build_user_activity_slices(statistics)
        # users 0 (3 train), 1 (1 train), 2 (2 train) are all in the 1-3 bucket.
        assert set(frames["users_activity_1-3"]["entity_id"]) == {0, 1, 2}

    def test_empty_statistics_yield_no_slices(self):
        frames, definitions = build_user_activity_slices(pd.DataFrame())
        assert frames == {}
        assert definitions == []


class TestItemPopularitySlices:
    def test_head_and_tail_partition_the_catalogue(self, split_frame):
        popularity = build_item_popularity(split_frame)
        frames, _ = build_item_popularity_slices(popularity, long_tail_quantile=0.8)
        head = set(frames["items_head"]["entity_id"])
        tail = set(frames["items_long_tail"]["entity_id"])
        assert head & tail == set()
        assert head | tail == set(popularity["internal_item_id"])

    def test_the_rule_is_recorded(self, split_frame):
        popularity = build_item_popularity(split_frame)
        _, definitions = build_item_popularity_slices(popularity, long_tail_quantile=0.8)
        assert "80%" in definitions[0].rule


class TestColdStart:
    def test_cold_items_are_those_absent_from_training(self, split_frame):
        frames, definitions = build_cold_item_slice(split_frame)
        cold = set(frames["items_cold_start"]["entity_id"])
        train_items = set(split_frame[split_frame.split == "train"]["internal_item_id"])
        assert cold & train_items == set()
        assert definitions[0].size == len(cold)

    def test_cold_users_are_empty_under_leave_last_n(self, split_frame):
        """Every eligible user keeps training history by construction."""
        frames, definitions = build_cold_user_slice(split_frame)
        assert len(frames["users_cold_start"]) == 0
        assert definitions[0].size == 0

    def test_empty_cold_user_slice_is_still_emitted(self, split_frame):
        """Emitted empty so a future split strategy surfaces them without a schema change."""
        frames, _ = build_cold_user_slice(split_frame)
        assert "users_cold_start" in frames
        assert list(frames["users_cold_start"].columns) == [
            "slice_name",
            "entity_type",
            "entity_id",
        ]


class TestModalitySlices:
    def test_all_items_are_missing_when_no_features_exist(self):
        metadata = pd.DataFrame({"internal_item_id": [0, 1, 2]})
        empty_text = pd.DataFrame({"internal_item_id": [0, 1, 2], "has_text_feature": [False] * 3})
        empty_image = pd.DataFrame(
            {"internal_item_id": [0, 1, 2], "has_image_feature": [False] * 3}
        )
        frames, _ = build_modality_slices(metadata, empty_text, empty_image)
        assert len(frames["items_missing_both_modalities"]) == 3
        assert len(frames["items_both_modalities"]) == 0

    def test_partial_coverage_is_partitioned_correctly(self):
        metadata = pd.DataFrame({"internal_item_id": [0, 1, 2]})
        text = pd.DataFrame(
            {"internal_item_id": [0, 1, 2], "has_text_feature": [True, True, False]}
        )
        image = pd.DataFrame(
            {"internal_item_id": [0, 1, 2], "has_image_feature": [True, False, False]}
        )
        frames, _ = build_modality_slices(metadata, text, image)
        assert set(frames["items_both_modalities"]["entity_id"]) == {0}
        assert set(frames["items_missing_image_features"]["entity_id"]) == {1, 2}
        assert set(frames["items_missing_both_modalities"]["entity_id"]) == {2}


class TestAllSlices:
    def test_builds_every_expected_slice(self, split_frame):
        popularity = build_item_popularity(split_frame)
        statistics = build_user_statistics(split_frame, popularity)
        metadata = pd.DataFrame(
            {"internal_item_id": sorted(split_frame["internal_item_id"].unique())}
        )
        empty_index = pd.DataFrame(
            {
                "internal_item_id": metadata["internal_item_id"],
                "has_text_feature": False,
                "has_image_feature": False,
            }
        )
        frames, definitions = build_all_slices(
            split_frame,
            user_statistics=statistics,
            item_popularity=popularity,
            item_metadata=metadata,
            text_index=empty_index,
            image_index=empty_index,
            long_tail_quantile=0.8,
        )
        expected = {
            "users_activity_1-3",
            "users_activity_4-10",
            "users_activity_11-30",
            "users_activity_31+",
            "items_long_tail",
            "items_head",
            "items_cold_start",
            "users_cold_start",
            "items_missing_text_features",
            "items_missing_image_features",
            "items_missing_both_modalities",
            "items_both_modalities",
        }
        assert set(frames) == expected
        assert len(definitions) == len(expected)

    def test_every_definition_records_its_rule(self, split_frame):
        popularity = build_item_popularity(split_frame)
        statistics = build_user_statistics(split_frame, popularity)
        metadata = pd.DataFrame({"internal_item_id": [10, 11]})
        empty = pd.DataFrame(
            {"internal_item_id": [10, 11], "has_text_feature": False, "has_image_feature": False}
        )
        _, definitions = build_all_slices(
            split_frame,
            user_statistics=statistics,
            item_popularity=popularity,
            item_metadata=metadata,
            text_index=empty,
            image_index=empty,
            long_tail_quantile=0.8,
        )
        assert all(definition.rule for definition in definitions)
        assert all(definition.description for definition in definitions)
