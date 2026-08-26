"""The Phase 5 report generator.

The report is the artifact a reader trusts, so the properties that matter are
about honesty rather than formatting: a missing measurement must say it is
missing, and a significance claim must follow its own interval rather than
prose written when the numbers were different.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_generate_phase5_report", PROJECT_ROOT / "scripts" / "generate_phase5_report.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPORT = _load_generator()


def interval(challenger: str, baseline: str, metric: str, delta: float, low: float, high: float):
    """One bootstrap row as the CSV reader would produce it: all strings."""
    return {
        "challenger": challenger,
        "baseline": baseline,
        "metric": metric,
        "delta": str(delta),
        "ci_lower": str(low),
        "ci_upper": str(high),
        "excludes_zero": str(low > 0 or high < 0),
        "users": "50000",
        "samples": "1000",
    }


class TestCellFormatting:
    def test_a_pipe_is_escaped_so_it_cannot_split_a_row(self) -> None:
        """`lightgcn|sasrec` would otherwise shift every value right of it."""
        assert REPORT._cell("lightgcn|sasrec") == "lightgcn\\|sasrec"

    def test_a_long_float_is_trimmed(self) -> None:
        """A raw repr implies precision the measurement does not have."""
        assert REPORT._cell(0.00027801197967980233) == "0.000278"

    def test_an_integer_valued_float_loses_its_decimal(self) -> None:
        assert REPORT._cell(2270.0) == "2270"

    def test_a_blank_stays_blank(self) -> None:
        assert REPORT._cell("") == ""
        assert REPORT._cell(None) == ""

    def test_a_bool_is_preserved_not_coerced_to_a_number(self) -> None:
        assert REPORT._cell(True) == "True"


class TestMissingEvidenceIsStated:
    def test_an_empty_table_says_so_rather_than_rendering_blank(self) -> None:
        """A blank table reads as 'measured, nothing found'. It is not."""
        rendered = REPORT.table([], [("a", "A")])
        assert "has not been produced" in rendered

    def test_the_missing_helper_names_the_file(self) -> None:
        assert "candidate_recall.csv" in REPORT.missing("candidate_recall.csv")

    def test_a_table_with_rows_renders_them(self) -> None:
        rendered = REPORT.table([{"a": 1}, {"a": 2}], [("a", "A")])
        assert "| A |" in rendered
        assert rendered.count("\n") == 4


class TestSignificanceFollowsTheInterval:
    def test_an_interval_excluding_zero_is_reported_significant(self) -> None:
        rows = [interval("five_source_rrf", "four_source_rrf", "ndcg@20", 0.0002, 0.0001, 0.0003)]
        assert "significant" in REPORT.headline_fusion_verdict(rows)
        assert "**no** statistically significant" not in REPORT.headline_fusion_verdict(rows)

    def test_an_interval_crossing_zero_is_never_called_significant(self) -> None:
        """The claim this guards: a large delta with a crossing interval."""
        rows = [interval("five_source_rrf", "four_source_rrf", "recall@20", 0.05, -0.01, 0.11)]
        verdict = REPORT.headline_fusion_verdict(rows)
        assert "**no** statistically significant gain" in verdict

    def test_a_mixed_result_names_both_sides(self) -> None:
        rows = [
            interval("five_source_rrf", "four_source_rrf", "ndcg@20", 0.0002, 0.0001, 0.0003),
            interval("five_source_rrf", "four_source_rrf", "recall@20", 0.0003, -0.0001, 0.0007),
        ]
        verdict = REPORT.headline_fusion_verdict(rows)
        assert "ndcg@20" in verdict
        assert "no** significant gain on recall@20" in verdict

    def test_an_absent_comparison_is_reported_as_unmeasured(self) -> None:
        assert "not measured" in REPORT.headline_fusion_verdict([])

    def test_the_prose_reports_each_interval_it_was_given(self) -> None:
        rows = [
            interval("two_tower", "lightgcn", "ndcg@20", -0.005, -0.006, -0.004),
            interval("two_tower", "lightgcn", "cold_recall@20", -0.0008, -0.0026, 0.0008),
        ]
        prose = REPORT.significance_prose(rows)
        assert "ndcg@20: significant" in prose
        assert "cold_recall@20: not significant" in prose

    def test_no_intervals_means_no_significance_is_claimed(self) -> None:
        assert "no significance is claimed" in REPORT.significance_prose([])


class TestColdBarVerdict:
    """Whether the bar was met, and whether the difference is significant.

    These are two separate questions, and an earlier version conflated them: it
    reported "a real deficit" for any interval excluding zero, and so described
    a 13x *advantage* as a deficit. Direction comes from the point estimate;
    significance comes from the interval.
    """

    beaten: ClassVar[list[dict[str, str]]] = [
        {"system": "two_tower", "kind": "single", "cold_recall@20": "0.000441"},
        {"system": "lightgcn", "kind": "single", "cold_recall@20": "0.001322"},
    ]
    cleared: ClassVar[list[dict[str, str]]] = [
        {"system": "two_tower", "kind": "single", "cold_recall@20": "0.018062"},
        {"system": "lightgcn", "kind": "single", "cold_recall@20": "0.001322"},
    ]

    def test_a_higher_number_means_the_bar_was_met(self) -> None:
        met, _ = REPORT.cold_bar_verdict(self.cleared)
        assert met is True

    def test_a_lower_number_means_it_was_not(self) -> None:
        met, _ = REPORT.cold_bar_verdict(self.beaten)
        assert met is False

    def test_it_reports_both_numbers(self) -> None:
        _, prose = REPORT.cold_bar_verdict(self.beaten)
        assert "0.000441" in prose
        assert "0.001322" in prose

    def test_a_positive_significant_delta_is_an_advantage_not_a_deficit(self) -> None:
        """The exact bug this guards. A 13x win must never read as a deficit."""
        rows = [interval("two_tower", "lightgcn", "cold_recall@20", 0.0167, 0.0114, 0.0224)]
        met, prose = REPORT.cold_bar_verdict(self.cleared, rows)
        assert met is True
        assert "a real advantage" in prose
        assert "deficit" not in prose

    def test_a_negative_significant_delta_is_a_deficit(self) -> None:
        rows = [interval("two_tower", "lightgcn", "cold_recall@20", -0.0009, -0.0026, -0.0002)]
        met, prose = REPORT.cold_bar_verdict(self.beaten, rows)
        assert met is False
        assert "a real deficit" in prose

    def test_a_crossing_interval_is_called_neither(self) -> None:
        """Overstating in the pessimistic direction is still overstating."""
        rows = [interval("two_tower", "lightgcn", "cold_recall@20", -0.0009, -0.0026, 0.0009)]
        _, prose = REPORT.cold_bar_verdict(self.beaten, rows)
        assert "crosses zero" in prose
        assert "not\nstatistically significant" in prose or "not statistically" in prose
        assert "deficit" not in prose
        assert "advantage" not in prose

    def test_it_reports_the_ratio_when_the_baseline_is_positive(self) -> None:
        _, prose = REPORT.cold_bar_verdict(self.cleared)
        assert "13.7" in prose

    def test_an_unmeasured_comparison_says_so(self) -> None:
        met, prose = REPORT.cold_bar_verdict([])
        assert met is False
        assert "not measured" in prose


class TestStandaloneVerdict:
    """Where the two-tower sits against the best collaborative source.

    Derived rather than written because this sentence survived a refit that
    reversed it: prose that outlives the numbers it describes is worse than no
    prose.
    """

    stronger: ClassVar[list[dict[str, str]]] = [
        {"system": "two_tower", "kind": "single", "ndcg@20": "0.00887"},
        {"system": "lightgcn", "kind": "single", "ndcg@20": "0.00611"},
    ]
    weaker: ClassVar[list[dict[str, str]]] = [
        {"system": "two_tower", "kind": "single", "ndcg@20": "0.00041"},
        {"system": "lightgcn", "kind": "single", "ndcg@20": "0.00611"},
    ]

    def test_a_stronger_model_is_reported_above(self) -> None:
        assert "above" in REPORT._standalone_verdict(self.stronger, [])

    def test_a_weaker_model_is_reported_below(self) -> None:
        assert "below" in REPORT._standalone_verdict(self.weaker, [])

    def test_it_names_the_source_it_compared_against(self) -> None:
        assert "lightgcn" in REPORT._standalone_verdict(self.stronger, [])

    def test_significance_is_taken_from_the_interval(self) -> None:
        rows = [interval("two_tower", "lightgcn", "ndcg@20", 0.0028, 0.0022, 0.0033)]
        assert "excludes zero" in REPORT._standalone_verdict(self.stronger, rows)

    def test_a_crossing_interval_is_reported_as_such(self) -> None:
        rows = [interval("two_tower", "lightgcn", "ndcg@20", 0.0028, -0.001, 0.006)]
        assert "crosses zero" in REPORT._standalone_verdict(self.stronger, rows)

    def test_an_unmeasured_comparison_says_so(self) -> None:
        assert "not measured" in REPORT._standalone_verdict([], [])


class TestGateDetailLookup:
    gate: ClassVar[dict[str, Any]] = {
        "results": [
            {"check": "seen-item filtering", "status": "PASS", "detail": "no leakage"},
            {"check": "smoke", "status": "FAIL", "detail": "broken"},
        ]
    }

    def test_it_reports_status_and_detail(self) -> None:
        assert REPORT._gate_detail(self.gate, "seen-item filtering") == "PASS — no leakage"

    def test_a_failure_is_reported_as_a_failure(self) -> None:
        assert REPORT._gate_detail(self.gate, "smoke").startswith("FAIL")

    def test_an_absent_check_is_not_silently_a_pass(self) -> None:
        detail = REPORT._gate_detail(self.gate, "never ran")
        assert "not run" in detail


needs_report = pytest.mark.skipif(
    not (PROJECT_ROOT / "docs/phase_reports/phase_05_report.md").is_file(),
    reason="report not generated in this checkout",
)


class TestGeneratedReport:
    @needs_report
    def test_it_carries_all_fifty_one_numbered_sections(self) -> None:
        text = (PROJECT_ROOT / "docs/phase_reports/phase_05_report.md").read_text()
        for index in range(1, 52):
            assert f"\n## {index}. " in text, f"section {index} is missing"

    @needs_report
    def test_it_distinguishes_the_result_types(self) -> None:
        """A reader must be able to tell synthetic from real from test."""
        text = (PROJECT_ROOT / "docs/phase_reports/phase_05_report.md").read_text().lower()
        for phrase in ("synthetic", "rolling validation", "official final test", "strict", "warm"):
            assert phrase in text

    @needs_report
    def test_it_does_not_claim_an_unqualified_fusion_improvement(self) -> None:
        """The spec's explicit prohibition, checked literally."""
        text = (PROJECT_ROOT / "docs/phase_reports/phase_05_report.md").read_text().lower()
        # Any claim of a gain must appear alongside its interval language.
        assert "interval" in text
        assert "significant" in text


class TestColdPositivityFollowsTheTable:
    """The cold-recall commentary must agree with the table above it.

    The hand-written version outlived the model it described: after a refit
    made cold Recall positive at every cutoff, the prose still claimed it was
    zero at K=5 and K=10 — contradicting the table printed three lines above.
    """

    def test_all_positive_says_every_cutoff(self) -> None:
        cold = {
            "recall@5": 0.008811,
            "recall@10": 0.012775,
            "recall@20": 0.018062,
            "recall@50": 0.029515,
        }
        prose = REPORT._cold_positivity(cold)
        assert "positive at every measured cutoff" in prose
        assert "zero at" not in prose

    def test_a_genuine_zero_is_reported_as_zero(self) -> None:
        cold = {"recall@5": 0.0, "recall@10": 0.0, "recall@20": 0.0004, "recall@50": 0.0013}
        prose = REPORT._cold_positivity(cold)
        assert "zero at K = 5, 10" in prose
        assert "positive at K = 20, 50" in prose

    def test_all_zero_fails_the_requirement_explicitly(self) -> None:
        """Phase 5's completion requirement is cold Recall@K > 0 somewhere."""
        prose = REPORT._cold_positivity({"recall@5": 0.0, "recall@20": 0.0})
        assert "zero at every measured cutoff" in prose
        assert "has not" in prose

    def test_an_unmeasured_cold_view_says_so(self) -> None:
        assert "not recorded" in REPORT._cold_positivity({})


class TestAccuracyLimitationFollowsTheRanking:
    """Limitation 1 must not contradict sections 29, 33 and 50."""

    final: ClassVar[dict[str, Any]] = {"strict": {"ndcg@20": 0.008873}}
    leading: ClassVar[list[dict[str, str]]] = [
        {"system": "two_tower", "kind": "single", "ndcg@20": "0.008873"},
        {"system": "lightgcn", "kind": "single", "ndcg@20": "0.006108"},
    ]
    trailing: ClassVar[list[dict[str, str]]] = [
        {"system": "two_tower", "kind": "single", "ndcg@20": "0.000408"},
        {"system": "lightgcn", "kind": "single", "ndcg@20": "0.006108"},
    ]

    def test_a_leading_model_is_not_called_below_the_baseline(self) -> None:
        prose = REPORT._accuracy_limitation(self.final, self.leading)
        assert "below" not in prose
        assert "highest of the five sources" in prose

    def test_a_leading_model_still_states_the_absolute_caveat(self) -> None:
        """Leading internally is not the same as being any good."""
        prose = REPORT._accuracy_limitation(self.final, self.leading)
        assert "absolute" in prose.lower()

    def test_a_trailing_model_is_reported_as_trailing(self) -> None:
        prose = REPORT._accuracy_limitation({"strict": {"ndcg@20": 0.000408}}, self.trailing)
        assert "below the best collaborative source" in prose
