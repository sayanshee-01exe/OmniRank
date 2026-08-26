"""Five-source fusion arithmetic and the shape of its evidence files.

The RRF tests here are deliberately hand-computable. Fusion is the step where a
scoring mistake is least visible: every source returns plausible lists, the
blend returns a plausible list, and only the arithmetic says whether the blend
means anything.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from omnirank.models.base import Candidate
from omnirank.retrieval.aggregation import build_aggregator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE_ROOT = PROJECT_ROOT / "reports/metrics/phase_05"


def _load_fusion_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_compare_five_source_fusion", PROJECT_ROOT / "scripts" / "compare_five_source_fusion.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FUSION = _load_fusion_script()


def candidates(*item_ids: str) -> list[Candidate]:
    """A source's ranked list. Scores descend but are deliberately arbitrary."""
    return [
        Candidate(item_id=item, score=1.0 / (rank + 1), sources=("s",))
        for rank, item in enumerate(item_ids)
    ]


class TestReciprocalRankArithmetic:
    def test_the_formula_is_what_it_claims(self) -> None:
        """RRF(i) = sum over sources of w_s / (c + rank_s(i)), 1-based rank."""
        aggregator = build_aggregator("reciprocal_rank_fusion", rrf_constant=60.0)
        result = aggregator.aggregate({"a": candidates("x")}, limit=1)
        assert result.candidates[0].score == pytest.approx(1.0 / 61.0)

    def test_two_sources_agreeing_beat_one_source_alone(self) -> None:
        """The property that makes fusion worth doing at all."""
        aggregator = build_aggregator("reciprocal_rank_fusion")
        result = aggregator.aggregate(
            {"a": candidates("shared", "solo_a"), "b": candidates("shared", "solo_b")},
            limit=3,
        )
        assert result.candidates[0].item_id == "shared"

    def test_an_item_absent_from_a_source_contributes_nothing_from_it(self) -> None:
        """Not a zero score -- no term at all. A zero would be a vote."""
        aggregator = build_aggregator("reciprocal_rank_fusion", rrf_constant=60.0)
        result = aggregator.aggregate(
            {"a": candidates("only_in_a"), "b": candidates("only_in_b")}, limit=2
        )
        for candidate in result.candidates:
            assert candidate.score == pytest.approx(1.0 / 61.0)

    def test_ranks_are_fused_not_scores(self) -> None:
        """A source with huge raw scores must not dominate.

        This is why fusion is rank-based: the two-tower produces cosine
        similarities and LightGCN produces graph-propagated dot products, and
        no calibration puts those on a common scale.
        """
        aggregator = build_aggregator("reciprocal_rank_fusion")
        huge = [Candidate(item_id="big", score=1e9, sources=("a",))]
        modest = [Candidate(item_id="small", score=1e-9, sources=("b",))]
        result = aggregator.aggregate({"a": huge, "b": modest}, limit=2)
        scores = {c.item_id: c.score for c in result.candidates}
        assert scores["big"] == pytest.approx(scores["small"])

    def test_weights_shift_the_ordering_in_the_direction_given(self) -> None:
        weighted = build_aggregator("reciprocal_rank_fusion", source_weights={"a": 10.0, "b": 1.0})
        result = weighted.aggregate({"a": candidates("from_a"), "b": candidates("from_b")}, limit=2)
        assert result.candidates[0].item_id == "from_a"

    def test_a_source_returning_nothing_does_not_break_the_blend(self) -> None:
        aggregator = build_aggregator("reciprocal_rank_fusion")
        result = aggregator.aggregate({"a": candidates("x", "y"), "b": []}, limit=2)
        assert [c.item_id for c in result.candidates] == ["x", "y"]


class TestSourceWeightsAreFixedInAdvance:
    def test_every_fused_source_has_a_weight(self) -> None:
        """A missing weight silently defaults, which is a different experiment."""
        members = {*FUSION.COLLABORATIVE, FUSION.TWO_TOWER}
        assert members <= set(FUSION.SOURCE_WEIGHTS)

    def test_weights_are_non_negative(self) -> None:
        assert all(weight >= 0 for weight in FUSION.SOURCE_WEIGHTS.values())


class TestBootstrapConfiguration:
    def test_the_required_comparisons_are_all_present(self) -> None:
        pairs = set(FUSION.BOOTSTRAP_PAIRS)
        for required in (
            ("two_tower", "popularity"),
            ("two_tower", "matrix_factorization"),
            ("two_tower", "lightgcn"),
            ("five_source_rrf", "four_source_rrf"),
            ("five_source_rrf", "lightgcn"),
        ):
            assert required in pairs, f"missing comparison: {required}"

    def test_cold_recall_is_among_the_bootstrap_metrics(self) -> None:
        """The metric Phase 5 exists to move must be compared, not assumed."""
        assert "cold_recall@20" in FUSION.BOOTSTRAP_METRICS

    def test_the_resample_count_is_fixed(self) -> None:
        assert FUSION.BOOTSTRAP_SAMPLES >= 1000


needs_evidence = pytest.mark.skipif(
    not (PHASE_ROOT / "bootstrap_deltas.csv").is_file(),
    reason="no real fusion evidence in this checkout",
)


class TestBootstrapOutputSchema:
    @needs_evidence
    def test_every_row_carries_the_required_columns(self) -> None:
        import csv

        with (PHASE_ROOT / "bootstrap_deltas.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        required = {
            "challenger",
            "baseline",
            "metric",
            "delta",
            "ci_lower",
            "ci_upper",
            "confidence_level",
            "samples",
            "users",
            "excludes_zero",
        }
        assert rows
        for row in rows:
            assert required <= set(row), f"missing columns in {row}"

    @needs_evidence
    def test_the_point_estimate_lies_inside_its_own_interval(self) -> None:
        """A delta outside its CI means the two were computed from different data."""
        import csv

        with (PHASE_ROOT / "bootstrap_deltas.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                low, high = float(row["ci_lower"]), float(row["ci_upper"])
                assert low <= float(row["delta"]) <= high, row

    @needs_evidence
    def test_significance_is_exactly_whether_the_interval_excludes_zero(self) -> None:
        """Significance must never be asserted independently of the interval."""
        import csv

        with (PHASE_ROOT / "bootstrap_deltas.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                low, high = float(row["ci_lower"]), float(row["ci_upper"])
                expected = low > 0 or high < 0
                assert (str(row["excludes_zero"]).lower() == "true") is expected, row


class TestUniqueContributionEvidence:
    @needs_evidence
    def test_the_unique_contribution_file_records_a_measurement(self) -> None:
        payload = json.loads((PHASE_ROOT / "two_tower_unique_contribution.json").read_text())
        assert payload.get("targets_reached_only_by_two_tower") is not None
        assert payload.get("pairwise_jaccard")
        assert payload.get("unique_contribution")

    @needs_evidence
    def test_jaccard_values_are_proper_fractions(self) -> None:
        payload = json.loads((PHASE_ROOT / "two_tower_unique_contribution.json").read_text())
        for pair, value in payload["pairwise_jaccard"].items():
            assert 0.0 <= float(value) <= 1.0, pair

    @needs_evidence
    def test_the_two_tower_appears_in_every_measured_pairing(self) -> None:
        payload = json.loads((PHASE_ROOT / "two_tower_unique_contribution.json").read_text())
        partners = {
            name
            for pair in payload["pairwise_jaccard"]
            for name in pair.split("|")
            if name != "two_tower"
        }
        assert {"lightgcn", "sasrec", "popularity", "matrix_factorization"} <= partners


def _load_compare_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_compare_multimodal_retrievers",
        PROJECT_ROOT / "scripts" / "compare_multimodal_retrievers.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMPARE = _load_compare_script()


class _Retriever:
    """Just enough of a retriever for the population guard."""

    def __init__(self, servable: int, total: int) -> None:
        self._histories = {user: ([1, 2, 3] if user < servable else []) for user in range(total)}


class TestFinalPopulationGuard:
    """A final model must answer for the population it will be asked about.

    Not hypothetical: the first registered Phase 5 final model was fitted with
    the development `--subset-users` default and could serve 5,000 of 50,000
    users. It loaded cleanly, its checksums matched, and every metric it
    produced was depressed by a missing flag rather than by the model.
    """

    def test_a_complete_population_passes(self) -> None:
        assert COMPARE.population_shortfall(_Retriever(50_000, 50_000), 50_000) is None

    def test_a_handful_of_unservable_users_is_tolerated(self) -> None:
        """Not every user is fittable, and that is a data property.

        A user whose entire log is the single interaction that became the
        held-out target has no history left to build a query from. The fold
        builder excludes them for the same reason. Demanding 100% would refuse
        a correct model.
        """
        assert COMPARE.population_shortfall(_Retriever(49_999, 50_000), 50_000) is None

    def test_the_threshold_sits_between_the_two_cases(self) -> None:
        """A few excluded users and a 10% subset are orders of magnitude apart,
        so any threshold between them separates the cases cleanly."""
        assert 0.1 < COMPARE.MINIMUM_SERVABLE_SHARE < 1.0

    def test_a_subset_fitted_model_is_refused(self) -> None:
        shortfall = COMPARE.population_shortfall(_Retriever(5_000, 50_000), 50_000)
        assert shortfall is not None
        assert shortfall["servable_users"] == 5_000
        assert shortfall["expected_users"] == 50_000
        assert shortfall["share"] == 0.1

    def test_users_present_but_historyless_do_not_count_as_servable(self) -> None:
        """An empty history returns an empty list. Present is not servable."""
        shortfall = COMPARE.population_shortfall(_Retriever(0, 50_000), 50_000)
        assert shortfall is not None
        assert shortfall["servable_users"] == 0

    def test_the_message_names_the_flag_to_change(self) -> None:
        shortfall = COMPARE.population_shortfall(_Retriever(1, 10), 10)
        assert shortfall is not None
        assert "--subset-users" in shortfall["detail"]
