"""Retrieval diagnostics.

The numbers here are checked against hand arithmetic on tiny inputs. Both
metrics are summary statistics over set operations, which is exactly the kind of
code that produces a plausible number while being wrong -- and a plausible
overlap figure would be believed, because there is nothing to compare it to.
"""

from __future__ import annotations

import pytest

from omnirank.core.exceptions import DataError
from omnirank.retrieval.diagnostics import candidate_recall, source_overlap


class TestCandidateRecall:
    def test_counts_users_whose_target_is_in_the_pool(self) -> None:
        result = candidate_recall(
            {"u1": ["a", "b", "c"], "u2": ["x", "y", "z"]},
            {"u1": {"b"}, "u2": {"q"}},
            depth=3,
        )
        assert result.users_evaluated == 2
        assert result.users_with_target == 1
        assert result.recall == pytest.approx(0.5)

    def test_depth_truncates_the_pool(self) -> None:
        """A target at rank 3 is not retrieved at depth 2."""
        assert candidate_recall({"u1": ["a", "b", "c"]}, {"u1": {"c"}}, depth=2).recall == 0.0
        assert candidate_recall({"u1": ["a", "b", "c"]}, {"u1": {"c"}}, depth=3).recall == 1.0

    def test_users_without_targets_are_not_evaluated(self) -> None:
        """A user with no held-out target cannot succeed or fail."""
        result = candidate_recall({"u1": ["a"], "u2": ["b"]}, {"u1": {"a"}, "u2": set()}, depth=5)
        assert result.users_evaluated == 1

    def test_a_user_with_no_candidates_is_a_miss_not_a_skip(self) -> None:
        result = candidate_recall({}, {"u1": {"a"}}, depth=5)
        assert result.users_evaluated == 1
        assert result.recall == 0.0

    def test_unreachable_targets_are_counted_separately(self) -> None:
        """A cold target is a legitimate miss, not a retrieval failure."""
        result = candidate_recall(
            {"u1": ["a"], "u2": ["a"]},
            {"u1": {"a"}, "u2": {"cold"}},
            depth=5,
            reachable_items={"a", "b"},
        )
        assert result.users_with_unreachable_target == 1
        assert result.recall == pytest.approx(0.5)
        # Over the users a retriever could have served, it got them all.
        assert result.reachable_recall == pytest.approx(1.0)

    def test_reachable_recall_equals_recall_without_a_catalogue(self) -> None:
        result = candidate_recall({"u1": ["a"]}, {"u1": {"a"}}, depth=5)
        assert result.reachable_recall == result.recall

    def test_empty_input_does_not_divide_by_zero(self) -> None:
        result = candidate_recall({}, {}, depth=5)
        assert result.recall == 0.0
        assert result.reachable_recall == 0.0

    def test_rejects_a_non_positive_depth(self) -> None:
        with pytest.raises(DataError):
            candidate_recall({}, {}, depth=0)


class TestSourceOverlap:
    def test_jaccard_matches_hand_arithmetic(self) -> None:
        """{a,b,c} vs {b,c,d}: intersection 2, union 4."""
        result = source_overlap(
            {"one": {"u1": ["a", "b", "c"]}, "two": {"u1": ["b", "c", "d"]}}, depth=3
        )
        assert result.pairwise_jaccard["one|two"] == pytest.approx(0.5)

    def test_identical_sources_score_one(self) -> None:
        result = source_overlap({"one": {"u1": ["a", "b"]}, "two": {"u1": ["a", "b"]}}, depth=2)
        assert result.pairwise_jaccard["one|two"] == pytest.approx(1.0)
        assert result.unique_contribution == {"one": 0.0, "two": 0.0}

    def test_disjoint_sources_score_zero(self) -> None:
        result = source_overlap({"one": {"u1": ["a", "b"]}, "two": {"u1": ["c", "d"]}}, depth=2)
        assert result.pairwise_jaccard["one|two"] == 0.0
        assert result.unique_contribution == {"one": 1.0, "two": 1.0}

    def test_two_empty_sources_are_not_treated_as_identical(self) -> None:
        """Agreement by vacancy is not similarity."""
        result = source_overlap({"one": {"u1": []}, "two": {"u1": []}}, depth=3)
        assert result.pairwise_jaccard.get("one|two", 0.0) == 0.0

    def test_mean_sources_per_item_matches_hand_arithmetic(self) -> None:
        """a is in both, b and c in one each: (2+1+1)/3 distinct items."""
        result = source_overlap({"one": {"u1": ["a", "b"]}, "two": {"u1": ["a", "c"]}}, depth=2)
        assert result.mean_sources_per_item == pytest.approx(4 / 3)

    def test_unique_contribution_is_per_source(self) -> None:
        """one contributes b uniquely (1 of 2); two contributes nothing unique."""
        result = source_overlap({"one": {"u1": ["a", "b"]}, "two": {"u1": ["a"]}}, depth=2)
        assert result.unique_contribution["one"] == pytest.approx(0.5)
        assert result.unique_contribution["two"] == pytest.approx(0.0)

    def test_depth_truncates_before_comparison(self) -> None:
        result = source_overlap(
            {"one": {"u1": ["a", "b", "c"]}, "two": {"u1": ["c", "d", "e"]}}, depth=2
        )
        assert result.pairwise_jaccard["one|two"] == 0.0  # c is beyond depth 2

    def test_averages_across_users(self) -> None:
        result = source_overlap(
            {
                "one": {"u1": ["a"], "u2": ["a"]},
                "two": {"u1": ["a"], "u2": ["b"]},
            },
            depth=1,
        )
        assert result.pairwise_jaccard["one|two"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2

    def test_rejects_a_single_source(self) -> None:
        with pytest.raises(DataError):
            source_overlap({"one": {"u1": ["a"]}}, depth=3)

    def test_rejects_a_non_positive_depth(self) -> None:
        with pytest.raises(DataError):
            source_overlap({"one": {}, "two": {}}, depth=0)
