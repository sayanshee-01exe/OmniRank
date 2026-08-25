"""The fold scorer and its summariser.

These definitions are what every rolling-fold number means, so they are tested
against hand-computable cases rather than against themselves.
"""

from __future__ import annotations

import math

import pytest

from omnirank.core.exceptions import OmniRankError
from omnirank.retrieval.fold_evaluation import (
    ABLATION_OVERRIDES,
    overrides_for,
    score_recommendations,
    summarise_folds,
)


class TestScoring:
    def test_hit_at_rank_one_is_perfect(self) -> None:
        scores = score_recommendations({"u": ["a", "b"]}, {"u": "a"}, set())
        assert scores["strict_recall@20"] == 1.0
        assert scores["strict_ndcg@20"] == 1.0

    def test_ndcg_discounts_by_rank(self) -> None:
        scores = score_recommendations({"u": ["x", "y", "a"]}, {"u": "a"}, set())
        assert scores["strict_recall@20"] == 1.0
        assert scores["strict_ndcg@20"] == pytest.approx(1 / math.log2(4), abs=1e-8)

    def test_cutoffs_are_respected(self) -> None:
        """A hit at rank 8 counts at 10 and above, not at 5."""
        items = [f"i{n}" for n in range(20)]
        scores = score_recommendations({"u": items}, {"u": "i7"}, set())
        assert scores["strict_recall@5"] == 0.0
        assert scores["strict_recall@10"] == 1.0
        assert scores["strict_ndcg@5"] == 0.0

    def test_miss_scores_zero_but_still_counts_in_the_denominator(self) -> None:
        scores = score_recommendations(
            {"a": ["x"], "b": ["y"]}, {"a": "x", "b": "unreachable"}, set()
        )
        assert scores["users_evaluated"] == 2
        assert scores["strict_recall@20"] == 0.5

    def test_users_with_no_candidates_leave_the_denominator(self) -> None:
        """An unanswerable user is excluded and the count says so.

        Scoring them as misses would conflate "ranked badly" with "could not be
        served at all", which are different failures with different fixes.
        """
        scores = score_recommendations({"a": ["x"], "b": []}, {"a": "x", "b": "z"}, set())
        assert scores["users_evaluated"] == 1
        assert scores["strict_recall@20"] == 1.0

    def test_cold_recall_uses_the_cold_denominator(self) -> None:
        """Cold recall is over cold-target users only, not over everyone."""
        recommended = {"a": ["cold1"], "b": ["warm1"], "c": ["x"]}
        targets = {"a": "cold1", "b": "warm1", "c": "cold2"}
        scores = score_recommendations(recommended, targets, {"cold1", "cold2"})
        assert scores["cold_users_evaluated"] == 2
        # One of two cold users was served their target.
        assert scores["cold_recall@20"] == 0.5
        assert scores["strict_recall@20"] == pytest.approx(2 / 3)

    def test_no_cold_targets_reports_undefined_not_zero(self) -> None:
        """A rate over zero users is undefined, and 0.0 reads as a failure.

        This matters in fold evaluation, where every target is warm by
        construction: reporting 0.0 would claim cold retrieval was measured and
        failed, when it was never measurable.
        """
        scores = score_recommendations({"a": ["x"]}, {"a": "x"}, set())
        assert scores["cold_users_evaluated"] == 0
        assert scores["cold_recall@20"] is None

    def test_candidate_recall_mirrors_the_deepest_cutoff(self) -> None:
        items = [f"i{n}" for n in range(300)]
        scores = score_recommendations({"u": items}, {"u": "i150"}, set())
        assert scores["candidate_recall@200"] == scores["strict_recall@200"]
        assert scores["candidate_recall@200"] == 1.0
        assert scores["strict_recall@100"] == 0.0


class TestOverrides:
    def test_unknown_label_is_refused(self) -> None:
        with pytest.raises(OmniRankError, match="Unknown ablation label"):
            overrides_for("no_such_variant")

    def test_overrides_are_copies(self) -> None:
        """A caller mutating its overrides must not edit the grid."""
        first = overrides_for("text_only")
        first["use_text"] = False
        assert ABLATION_OVERRIDES["text_only"]["use_text"] is True

    @pytest.mark.parametrize("label", sorted(ABLATION_OVERRIDES))
    def test_every_variant_enables_at_least_one_input(self, label: str) -> None:
        overrides = overrides_for(label)
        inputs = ("use_text", "use_image", "use_tag", "use_item_id_residual")
        assert any(overrides.get(name) for name in inputs), label


class TestSummary:
    @staticmethod
    def row(label: str, fold: str, seed: int, ndcg: float, cold: float | None = 0.0) -> dict:
        return {
            "label": label,
            "fold": fold,
            "seed": seed,
            "strict_ndcg@20": ndcg,
            "strict_recall@20": ndcg * 2,
            "cold_recall@20": cold,
            "candidate_recall@200": 0.5,
            "train_seconds": 10.0,
            "peak_memory_mb": 100.0,
        }

    def test_reports_worst_fold_beside_the_mean(self) -> None:
        rows = [self.row("a", "f3", 42, 0.10), self.row("a", "f2", 42, 0.02)]
        summary = summarise_folds(rows)[0]
        assert summary["mean_strict_ndcg@20"] == pytest.approx(0.06)
        assert summary["worst_fold_strict_ndcg@20"] == 0.02
        assert summary["stdev_strict_ndcg@20"] > 0

    def test_single_run_has_no_spread(self) -> None:
        summary = summarise_folds([self.row("a", "f3", 42, 0.1)])[0]
        assert summary["stdev_strict_ndcg@20"] == 0.0
        assert summary["runs"] == 1

    def test_configurations_are_summarised_separately(self) -> None:
        rows = [self.row("a", "f3", 42, 0.1), self.row("b", "f3", 42, 0.2)]
        summary = summarise_folds(rows)
        assert [entry["label"] for entry in summary] == ["a", "b"]

    def test_folds_and_seeds_are_recorded(self) -> None:
        rows = [
            self.row("a", "f3", 42, 0.1),
            self.row("a", "f2", 43, 0.2),
            self.row("a", "f3", 43, 0.15),
        ]
        summary = summarise_folds(rows)[0]
        assert summary["folds"] == "f2+f3"
        assert summary["seeds"] == "42+43"
        assert summary["runs"] == 3


class TestUnmeasuredColdRate:
    """A summary must not average away a metric that was never measured."""

    @staticmethod
    def row(cold: float | None) -> dict:
        return {
            "label": "a",
            "fold": "f3",
            "seed": 42,
            "strict_ndcg@20": 0.1,
            "strict_recall@20": 0.2,
            "cold_recall@20": cold,
            "candidate_recall@200": 0.5,
            "train_seconds": 10.0,
            "peak_memory_mb": 100.0,
        }

    def test_all_unmeasured_reports_none(self) -> None:
        summary = summarise_folds([self.row(None), self.row(None)])[0]
        assert summary["cold_runs_measured"] == 0
        assert summary["mean_cold_recall@20"] is None
        assert summary["worst_cold_recall@20"] is None

    def test_unmeasured_runs_do_not_drag_the_mean_down(self) -> None:
        """The bug this guards: `None` treated as 0.0 halves a real rate."""
        summary = summarise_folds([self.row(0.4), self.row(None)])[0]
        assert summary["cold_runs_measured"] == 1
        assert summary["mean_cold_recall@20"] == 0.4
