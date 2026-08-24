"""Ranking metrics, verified against hand-computed values.

Every expected number here was calculated by hand from the metric definition,
not captured from a previous run. A test that asserts whatever the code happened
to produce documents a bug as readily as a fix.
"""

from __future__ import annotations

import math

import pytest

from omnirank.core.exceptions import DataError
from omnirank.evaluation.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    validate_k,
)

RECS = ["a", "b", "c", "d", "e"]


class TestValidateK:
    @pytest.mark.parametrize("k", [1, 5, 20, 1000])
    def test_accepts_positive_integers(self, k):
        assert validate_k(k) == k

    @pytest.mark.parametrize("k", [0, -1, -100])
    def test_rejects_non_positive(self, k):
        with pytest.raises(DataError):
            validate_k(k)

    @pytest.mark.parametrize("k", [1.5, "5", None, True])
    def test_rejects_non_integers(self, k):
        """`True` is an int subclass; a boolean cut-off is a caller bug."""
        with pytest.raises(DataError):
            validate_k(k)


class TestRecall:
    def test_perfect_retrieval(self):
        assert recall_at_k(RECS, {"a"}, 5) == 1.0

    def test_miss(self):
        assert recall_at_k(RECS, {"z"}, 5) == 0.0

    def test_partial_with_two_relevant(self):
        # "a" retrieved, "z" not -> 1 of 2.
        assert recall_at_k(RECS, {"a", "z"}, 5) == 0.5

    def test_cutoff_excludes_later_hit(self):
        assert recall_at_k(RECS, {"e"}, 4) == 0.0
        assert recall_at_k(RECS, {"e"}, 5) == 1.0

    def test_empty_recommendations_score_zero(self):
        assert recall_at_k([], {"a"}, 20) == 0.0

    def test_short_list_is_not_padded(self):
        assert recall_at_k(["a"], {"a"}, 20) == 1.0

    def test_no_relevant_items_scores_zero(self):
        assert recall_at_k(RECS, set(), 5) == 0.0


class TestPrecision:
    def test_denominator_is_k_not_list_length(self):
        """A short list is penalised for the positions it did not fill."""
        assert precision_at_k(["a"], {"a"}, 5) == pytest.approx(0.2)

    def test_two_hits_in_five(self):
        assert precision_at_k(RECS, {"a", "c"}, 5) == pytest.approx(0.4)

    def test_perfect_at_one(self):
        assert precision_at_k(RECS, {"a"}, 1) == 1.0

    def test_empty_recommendations(self):
        assert precision_at_k([], {"a"}, 10) == 0.0


class TestHitRate:
    def test_hit(self):
        assert hit_rate_at_k(RECS, {"c"}, 5) == 1.0

    def test_miss_outside_cutoff(self):
        assert hit_rate_at_k(RECS, {"e"}, 3) == 0.0

    def test_binary_regardless_of_hit_count(self):
        assert hit_rate_at_k(RECS, {"a", "b", "c"}, 5) == 1.0


class TestReciprocalRank:
    @pytest.mark.parametrize(
        ("item", "expected"),
        [("a", 1.0), ("b", 0.5), ("c", 1 / 3), ("d", 0.25), ("e", 0.2)],
    )
    def test_rank_positions(self, item, expected):
        assert reciprocal_rank_at_k(RECS, {item}, 5) == pytest.approx(expected)

    def test_uses_first_relevant_only(self):
        assert reciprocal_rank_at_k(RECS, {"c", "e"}, 5) == pytest.approx(1 / 3)

    def test_miss(self):
        assert reciprocal_rank_at_k(RECS, {"z"}, 5) == 0.0


class TestAveragePrecision:
    def test_single_relevant_equals_reciprocal_rank(self):
        assert average_precision_at_k(RECS, {"c"}, 5) == pytest.approx(1 / 3)

    def test_two_relevant_hand_computed(self):
        # hits at positions 1 and 3 -> (1/1 + 2/3) / min(2, 5) = 0.8333...
        assert average_precision_at_k(RECS, {"a", "c"}, 5) == pytest.approx(5 / 6)

    def test_perfect_ranking(self):
        assert average_precision_at_k(RECS, {"a", "b"}, 5) == pytest.approx(1.0)

    def test_normalised_by_achievable_hits(self):
        """More relevant items than positions must still allow 1.0."""
        assert average_precision_at_k(RECS, {"a", "b", "c"}, 2) == pytest.approx(1.0)


class TestNDCG:
    def test_relevant_at_rank_one(self):
        assert ndcg_at_k(RECS, {"a"}, 5) == pytest.approx(1.0)

    def test_relevant_at_rank_two(self):
        assert ndcg_at_k(RECS, {"b"}, 5) == pytest.approx(1 / math.log2(3))

    def test_relevant_at_rank_k(self):
        assert ndcg_at_k(RECS, {"e"}, 5) == pytest.approx(1 / math.log2(6))

    def test_relevant_outside_k(self):
        assert ndcg_at_k(RECS, {"e"}, 4) == 0.0

    def test_perfect_multi_item_ranking(self):
        assert ndcg_at_k(RECS, {"a", "b"}, 5) == pytest.approx(1.0)

    def test_imperfect_multi_item_hand_computed(self):
        # gains at ranks 1 and 3: 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
        # ideal at ranks 1 and 2:  1 + 1/log2(3) = 1.63093
        expected = 1.5 / (1.0 + 1 / math.log2(3))
        assert ndcg_at_k(RECS, {"a", "c"}, 5) == pytest.approx(expected)

    def test_graded_relevance_orders_by_grade(self):
        high = ndcg_at_k(["a", "b"], {"a", "b"}, 2, relevance={"a": 3.0, "b": 1.0})
        low = ndcg_at_k(["b", "a"], {"a", "b"}, 2, relevance={"a": 3.0, "b": 1.0})
        assert high == pytest.approx(1.0)
        assert low < high

    def test_empty_recommendations(self):
        assert ndcg_at_k([], {"a"}, 20) == 0.0

    def test_no_relevant_items(self):
        assert ndcg_at_k(RECS, set(), 5) == 0.0


class TestOnePositiveEquivalences:
    """PixelRec holds out one item per user, collapsing three metric pairs.

    These are documented so a report never presents them as independent
    corroboration of one another.
    """

    @pytest.mark.parametrize("k", [1, 3, 5, 20])
    @pytest.mark.parametrize("target", ["a", "c", "e", "z"])
    def test_recall_equals_hit_rate(self, k, target):
        assert recall_at_k(RECS, {target}, k) == hit_rate_at_k(RECS, {target}, k)

    @pytest.mark.parametrize("k", [1, 3, 5, 20])
    @pytest.mark.parametrize("target", ["a", "c", "e", "z"])
    def test_map_equals_mrr(self, k, target):
        assert average_precision_at_k(RECS, {target}, k) == pytest.approx(
            reciprocal_rank_at_k(RECS, {target}, k)
        )

    @pytest.mark.parametrize("k", [1, 3, 5, 20])
    @pytest.mark.parametrize("target", ["a", "c", "e"])
    def test_precision_equals_recall_over_k(self, k, target):
        assert precision_at_k(RECS, {target}, k) == pytest.approx(
            recall_at_k(RECS, {target}, k) / k
        )
