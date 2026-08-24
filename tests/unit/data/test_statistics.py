"""Training-only user and item statistics."""

from __future__ import annotations

import pandas as pd

from omnirank.data.statistics import (
    ITEM_POPULARITY_COLUMNS,
    USER_STATISTIC_COLUMNS,
    build_item_popularity,
    build_user_statistics,
)


class TestItemPopularity:
    def test_counts_training_rows_only(self, split_frame):
        """Item 14 appears only as a test target and must not be counted."""
        popularity = build_item_popularity(split_frame)
        assert 14 not in set(popularity["internal_item_id"])

    def test_counts_match_a_manual_recount(self, split_frame):
        popularity = build_item_popularity(split_frame)
        expected = (
            split_frame[split_frame.split == "train"].groupby("internal_item_id").size().to_dict()
        )
        actual = dict(
            zip(
                popularity["internal_item_id"],
                popularity["training_interaction_count"],
                strict=True,
            )
        )
        assert actual == expected

    def test_columns_match_the_contract(self, split_frame):
        assert list(build_item_popularity(split_frame).columns) == list(ITEM_POPULARITY_COLUMNS)

    def test_ranks_are_unique_and_start_at_one(self, split_frame):
        popularity = build_item_popularity(split_frame)
        assert popularity["training_popularity_rank"].min() == 1
        assert popularity["training_popularity_rank"].is_unique

    def test_most_popular_item_ranks_first(self, split_frame):
        popularity = build_item_popularity(split_frame).sort_values("training_popularity_rank")
        assert popularity.iloc[0]["internal_item_id"] == 10  # appears in 3 training rows

    def test_long_tail_flag_is_assigned(self, split_frame):
        popularity = build_item_popularity(split_frame)
        assert popularity["long_tail_flag"].dtype == bool

    def test_head_is_never_empty(self, split_frame):
        """A uniform catalogue must not make every item long tail."""
        popularity = build_item_popularity(split_frame, long_tail_quantile=0.01)
        assert (~popularity["long_tail_flag"]).sum() >= 1

    def test_cold_items_are_omitted_not_zero_filled(self, split_frame):
        """A zero row would make a cold item look merely unpopular."""
        popularity = build_item_popularity(split_frame)
        assert (popularity["training_interaction_count"] > 0).all()

    def test_empty_training_split_yields_an_empty_frame(self):
        frame = pd.DataFrame(
            {
                "internal_user_id": [0],
                "internal_item_id": [1],
                "interaction_order": [0],
                "split": ["test"],
            }
        )
        assert build_item_popularity(frame).empty


class TestUserStatistics:
    def test_counts_training_rows_only(self, split_frame):
        statistics = build_user_statistics(split_frame, build_item_popularity(split_frame))
        counts = dict(
            zip(
                statistics["internal_user_id"],
                statistics["training_interaction_count"],
                strict=True,
            )
        )
        assert counts == {0: 3, 1: 1, 2: 2}

    def test_columns_match_the_contract(self, split_frame):
        statistics = build_user_statistics(split_frame, build_item_popularity(split_frame))
        assert list(statistics.columns) == list(USER_STATISTIC_COLUMNS)

    def test_last_training_order_excludes_held_out_events(self, split_frame):
        statistics = build_user_statistics(split_frame, build_item_popularity(split_frame))
        user_zero = statistics[statistics.internal_user_id == 0].iloc[0]
        assert user_zero["last_training_interaction_order"] == 2

    def test_mean_item_popularity_uses_training_popularity(self, split_frame):
        statistics = build_user_statistics(split_frame, build_item_popularity(split_frame))
        assert statistics["mean_item_popularity"].notna().all()

    def test_ineligible_users_still_get_statistics(self, split_frame):
        statistics = build_user_statistics(split_frame, build_item_popularity(split_frame))
        assert 2 in set(statistics["internal_user_id"])

    def test_empty_training_split_yields_an_empty_frame(self):
        frame = pd.DataFrame(
            {
                "internal_user_id": [0],
                "internal_item_id": [1],
                "interaction_order": [0],
                "split": ["test"],
            }
        )
        assert build_user_statistics(frame, pd.DataFrame()).empty
