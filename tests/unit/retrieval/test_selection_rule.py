"""The noise-aware selection rule.

Ranking on the mean alone picks a winner even when the gap is smaller than the
run-to-run spread, producing something that looks like a decision but is a coin
flip. These cases pin the boundary.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "_compare_multimodal",
    Path(__file__).resolve().parents[3] / "scripts" / "compare_multimodal_retrievers.py",
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_choose = _MODULE._choose


class _Logger:
    """Records the event name the rule reports, which is part of its contract."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def info(self, event: str, **_: Any) -> None:
        self.events.append(event)


def entry(label: str, mean: float, worst: float, stdev: float, runs: int = 6) -> dict[str, Any]:
    return {
        "label": label,
        "runs": runs,
        "mean_strict_ndcg@20": mean,
        "worst_fold_strict_ndcg@20": worst,
        "worst_fold_mean_strict_ndcg@20": worst,
        "stdev_strict_ndcg@20": stdev,
    }


class TestSelectionRule:
    def test_a_clear_lead_wins_on_the_mean(self) -> None:
        log = _Logger()
        chosen = _choose([entry("a", 0.10, 0.09, 0.001), entry("b", 0.02, 0.02, 0.001)], log, "run")
        assert chosen["label"] == "a"
        assert log.events == ["phase5.selection_decisive"]

    def test_a_gap_inside_the_noise_falls_back_to_the_worst_run(self) -> None:
        """The case this rule exists for.

        `b` has the lower mean but the higher floor, and the means differ by
        less than the spread. Ranking on the mean would take `a` on noise.
        """
        log = _Logger()
        chosen = _choose(
            [entry("a", 0.020, 0.005, 0.010), entry("b", 0.018, 0.017, 0.004)], log, "run"
        )
        assert chosen["label"] == "b"
        assert log.events == ["phase5.selection_within_noise"]

    @pytest.mark.parametrize(
        ("gap", "expected"),
        [
            (0.0049, "b"),  # gap below the noise: not distinguishable
            (0.0051, "a"),  # gap above the noise: a real lead
        ],
    )
    def test_the_noise_band_is_where_the_rule_switches(self, gap: float, expected: str) -> None:
        """Either side of the band, not on it.

        A gap *exactly* equal to the noise is a measure-zero case whose outcome
        is decided by float representation, so it is not pinned here. What is
        pinned is that the rule switches behaviour across the band.
        """
        log = _Logger()
        chosen = _choose(
            [entry("a", 0.020, 0.001, 0.005), entry("b", 0.020 - gap, 0.014, 0.005)],
            log,
            "run",
        )
        assert chosen["label"] == expected

    def test_only_configurations_inside_the_noise_band_can_tie_break(self) -> None:
        """A distant third must not win on its floor."""
        log = _Logger()
        chosen = _choose(
            [
                entry("a", 0.020, 0.010, 0.010),
                entry("b", 0.018, 0.015, 0.004),
                entry("far", 0.001, 0.0009, 0.0001),
            ],
            log,
            "run",
        )
        assert chosen["label"] == "b"

    def test_a_single_configuration_is_returned_unchanged(self) -> None:
        log = _Logger()
        assert _choose([entry("only", 0.1, 0.1, 0.0)], log, "run")["label"] == "only"
        assert log.events == []

    @pytest.mark.parametrize("worst_a", [0.001, 0.017])
    def test_the_highest_mean_still_wins_when_it_also_has_the_best_floor(
        self, worst_a: float
    ) -> None:
        log = _Logger()
        chosen = _choose(
            [entry("a", 0.020, worst_a, 0.010), entry("b", 0.019, 0.0005, 0.004)], log, "run"
        )
        assert chosen["label"] == "a"


class TestRunCountFairness:
    """The tie-break must not reward being measured less.

    A minimum over runs favours whichever contender had fewest draws -- more
    runs mean more chances to sample a low one. That is a property of the
    sampling, not of the model, and it produced a wrong selection in practice
    before the statistic was changed to the worst *fold mean*.
    """

    @staticmethod
    def contender(label: str, runs: int, worst_run: float, worst_fold_mean: float) -> dict:
        return {
            "label": label,
            "runs": runs,
            "mean_strict_ndcg@20": 0.020,
            # A 2-run contender never drew as low as the 6-run one...
            "worst_fold_strict_ndcg@20": worst_run,
            # ...but averaged within folds, the 6-run one is stronger.
            "worst_fold_mean_strict_ndcg@20": worst_fold_mean,
            "stdev_strict_ndcg@20": 0.009,
        }

    def test_the_less_measured_contender_does_not_win_on_a_lucky_minimum(self) -> None:
        log = _Logger()
        chosen = _choose(
            [
                self.contender(
                    "measured_six_times", runs=6, worst_run=0.010, worst_fold_mean=0.018
                ),
                self.contender("measured_twice", runs=2, worst_run=0.0106, worst_fold_mean=0.011),
            ],
            log,
            "run",
        )
        assert chosen["label"] == "measured_six_times"
        assert log.events == ["phase5.selection_within_noise"]
