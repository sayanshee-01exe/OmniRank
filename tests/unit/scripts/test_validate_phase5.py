"""The Phase 5 gate's own behaviour.

The gate is the thing that decides whether Phase 6 may begin, so a gate that
passes for the wrong reason is worse than no gate: it converts an unfinished
phase into a documented one. These tests pin the two properties that matter --
that CI mode cannot certify real completion, and that a critical failure
actually fails.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_gate() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_validate_phase5", PROJECT_ROOT / "scripts" / "validate_phase5.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load_gate()


class TestArgumentParsing:
    def test_ci_defaults_off(self) -> None:
        """The unflagged command must be the full local gate."""
        assert GATE.parse_args([]).ci is False

    def test_ci_flag_is_accepted(self) -> None:
        assert GATE.parse_args(["--ci"]).ci is True

    def test_quiet_and_json_survive_alongside_ci(self) -> None:
        args = GATE.parse_args(["--ci", "--quiet", "--json", "out.json"])
        assert (args.ci, args.quiet, args.json) == (True, True, "out.json")

    def test_no_run_tests_is_available_for_structural_inspection(self) -> None:
        assert GATE.parse_args(["--ci", "--no-run-tests"]).no_run_tests is True

    def test_an_unknown_flag_is_rejected(self) -> None:
        """Silently ignoring a typo'd flag would run the wrong mode."""
        with pytest.raises(SystemExit):
            GATE.parse_args(["--nonexistent"])


class TestCheckStatus:
    def test_a_passing_check_reads_pass(self) -> None:
        assert GATE.Check("x", passed=True, critical=True).status == "PASS"

    def test_a_failing_critical_check_reads_fail(self) -> None:
        assert GATE.Check("x", passed=False, critical=True).status == "FAIL"

    def test_a_failing_non_critical_check_reads_warn(self) -> None:
        assert GATE.Check("x", passed=False, critical=False).status == "WARN"

    def test_a_skipped_check_reads_skip(self) -> None:
        assert GATE.Check("x", passed=False, critical=True, skipped=True).status == "SKIP"


class TestSkipSemantics:
    """A skip is not a pass, and not a failure. It is 'not looked at'."""

    def test_a_skip_does_not_block_the_gate(self) -> None:
        result = GATE.GateResult()
        result.skip("real-data check", "no PixelRec in CI")
        assert result.passed is True

    def test_a_skip_is_not_counted_as_passed(self) -> None:
        result = GATE.GateResult()
        result.skip("real-data check", "no PixelRec in CI")
        assert result.to_dict()["checks_passed"] == 0
        assert result.to_dict()["skipped"] == 1

    def test_a_skip_is_not_counted_as_a_warning(self) -> None:
        """Otherwise CI's output would read as 13 unresolved problems."""
        result = GATE.GateResult()
        result.skip("real-data check", "no PixelRec in CI")
        assert result.to_dict()["warnings"] == 0

    def test_a_critical_failure_still_blocks(self) -> None:
        result = GATE.GateResult()
        result.skip("skipped", "n/a")
        result.add("broken", False, detail="genuinely broken")
        assert result.passed is False
        assert [check.name for check in result.critical_failures] == ["broken"]


class TestCiModeCannotCertifyCompletion:
    """The property that keeps a green CI badge honest."""

    def test_ci_mode_skips_every_real_data_check(self) -> None:
        result = GATE.GateResult(mode="ci")
        GATE.check_real_completion_not_claimed(result)
        skipped = {check.name for check in result.skipped}
        for name in (
            "registered two-tower model",
            "cold Recall@K is positive on real data",
            "five-source fusion evidence",
            "paired bootstrap evidence",
            "README records Phase 5 accurately",
        ):
            assert name in skipped, f"{name} must be skipped, not silently passed"

    def test_skipped_real_data_checks_are_never_marked_passed(self) -> None:
        result = GATE.GateResult(mode="ci")
        GATE.check_real_completion_not_claimed(result)
        assert not any(check.passed for check in result.checks)

    def test_the_mode_is_recorded_in_the_json_report(self) -> None:
        """A consumer must be able to tell the two runs apart."""
        assert GATE.GateResult(mode="ci").to_dict()["mode"] == "ci"
        assert GATE.GateResult(mode="full").to_dict()["mode"] == "full"


class TestCiJobInvocation:
    """`python ... | tee` reports tee's status, so a failing gate looks green."""

    @staticmethod
    def _workflow(tmp_path: Path, body: str, monkeypatch: Any) -> GATE.GateResult:
        workflow = tmp_path / ".github" / "workflows"
        workflow.mkdir(parents=True)
        (workflow / "ci.yml").write_text(body)
        monkeypatch.setattr(GATE, "PROJECT_ROOT", tmp_path)
        result = GATE.GateResult()
        GATE.check_ci_job(result)
        return result

    @staticmethod
    def _verdict(result: GATE.GateResult) -> bool:
        return next(
            check.passed
            for check in result.checks
            if check.name == "CI invokes the gate without swallowing its exit code"
        )

    def test_a_bare_invocation_passes(self, tmp_path: Path, monkeypatch: Any) -> None:
        body = (
            "jobs:\n  multimodal-retrieval:\n    steps:\n"
            "      - run: python scripts/validate_phase5.py --ci\n"
        )
        assert self._verdict(self._workflow(tmp_path, body, monkeypatch)) is True

    def test_a_pipe_guarded_by_pipefail_passes(self, tmp_path: Path, monkeypatch: Any) -> None:
        body = (
            "jobs:\n  multimodal-retrieval:\n    steps:\n      - run: |\n"
            "          set -o pipefail\n"
            "          python scripts/validate_phase5.py --ci | tee phase5-validation.log\n"
        )
        assert self._verdict(self._workflow(tmp_path, body, monkeypatch)) is True

    def test_an_unguarded_pipe_fails(self, tmp_path: Path, monkeypatch: Any) -> None:
        """The defect: tee exits 0, so a failing validator looks like success."""
        body = (
            "jobs:\n  multimodal-retrieval:\n    steps:\n      - run: |\n"
            "          python scripts/validate_phase5.py --ci | tee phase5-validation.log\n"
        )
        assert self._verdict(self._workflow(tmp_path, body, monkeypatch)) is False

    def test_pipefail_far_above_the_invocation_does_not_count(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """It must be in the same run block, not anywhere in the file."""
        body = (
            "jobs:\n  other:\n    steps:\n      - run: set -o pipefail\n"
            + "\n".join(f"      # filler {n}" for n in range(12))
            + "\n  multimodal-retrieval:\n    steps:\n      - run: |\n"
            "          python scripts/validate_phase5.py --ci | tee log\n"
        )
        assert self._verdict(self._workflow(tmp_path, body, monkeypatch)) is False

    def test_never_invoking_the_validator_fails(self, tmp_path: Path, monkeypatch: Any) -> None:
        body = "jobs:\n  multimodal-retrieval:\n    steps:\n      - run: pytest -q\n"
        assert self._verdict(self._workflow(tmp_path, body, monkeypatch)) is False


class TestReadmeCheck:
    @staticmethod
    def _verdict(tmp_path: Path, text: str, monkeypatch: Any) -> tuple[bool, str]:
        (tmp_path / "README.md").write_text(text)
        monkeypatch.setattr(GATE, "PROJECT_ROOT", tmp_path)
        result = GATE.GateResult()
        GATE.check_readme(result)
        check = result.checks[0]
        return check.passed, check.detail

    def test_a_correct_readme_passes(self, tmp_path: Path, monkeypatch: Any) -> None:
        passed, _ = self._verdict(tmp_path, "> **Status: Phase 5 ... complete.**\n", monkeypatch)
        assert passed is True

    def test_a_stale_phase_3_claim_fails(self, tmp_path: Path, monkeypatch: Any) -> None:
        passed, detail = self._verdict(
            tmp_path, "> **Status: Phase 3 complete.** Phase 5 is complete too.\n", monkeypatch
        )
        assert passed is False
        assert "status: phase 3" in detail

    def test_a_no_neural_model_claim_fails(self, tmp_path: Path, monkeypatch: Any) -> None:
        passed, _ = self._verdict(
            tmp_path, "Phase 5 complete. No neural retrieval model exists.\n", monkeypatch
        )
        assert passed is False

    def test_a_readme_silent_on_phase_5_fails(self, tmp_path: Path, monkeypatch: Any) -> None:
        passed, detail = self._verdict(tmp_path, "# OmniRank\n", monkeypatch)
        assert passed is False
        assert "does not record Phase 5" in detail


class TestPytestRunner:
    def test_a_missing_test_path_is_reported_not_raised(self) -> None:
        passed, detail = GATE._run_pytest("tests/does_not_exist.py")
        assert passed is False
        assert "missing test path" in detail

    def test_a_real_passing_target_reports_success(self) -> None:
        passed, detail = GATE._run_pytest("tests/unit/retrieval/test_fold_evaluation.py")
        assert passed is True
        assert "passed" in detail


class TestExitCodes:
    def test_the_documented_codes_are_what_the_module_uses(self) -> None:
        assert GATE.GATE_FAILED_EXIT == 1
        assert GATE.INSPECTION_ERROR_EXIT == 2

    def test_ci_mode_exits_zero_and_writes_a_ci_report(self, tmp_path: Path) -> None:
        report = tmp_path / "gate.json"
        code = GATE.main(["--ci", "--no-run-tests", "--quiet", "--json", str(report)])
        assert code == 0
        payload = json.loads(report.read_text())
        assert payload["mode"] == "ci"
        assert payload["skipped"] > 0
