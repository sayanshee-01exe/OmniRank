"""Repository smoke test: the project installs, imports, and holds together.

The point of these is to catch structural breakage - a module that stops
importing, a package that vanishes from the wheel, a docstring-free public
module - in one cheap run, before anything more specific is checked.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import omnirank

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALL_MODULES = sorted(
    module.name for module in pkgutil.walk_packages(omnirank.__path__, "omnirank.")
)


class TestImportIntegrity:
    def test_the_package_imports(self):
        assert omnirank.__version__

    def test_module_discovery_found_the_expected_surface(self):
        """A collapsed package tree would make every other test here vacuous."""
        assert len(ALL_MODULES) > 40

    @pytest.mark.parametrize("module_name", ALL_MODULES)
    def test_every_module_imports(self, module_name):
        importlib.import_module(module_name)

    def test_no_module_requires_the_ml_extra(self):
        """Phase 1 must import with none of torch/faiss/lightgbm installed."""
        heavy = {"torch", "faiss", "lightgbm", "sentence_transformers", "mlflow", "dvc"}
        for module_name in ALL_MODULES:
            importlib.import_module(module_name)
        assert heavy.isdisjoint(sys.modules)

    def test_every_public_module_has_a_docstring(self):
        missing = [
            name
            for name in ALL_MODULES
            if not (importlib.import_module(name).__doc__ or "").strip()
        ]
        assert missing == []


class TestLayering:
    """`core` is the base of the dependency graph and must import nothing above it."""

    def test_core_does_not_import_from_other_subpackages(self):
        forbidden = {
            "data",
            "models",
            "retrieval",
            "ranking",
            "reranking",
            "evaluation",
            "artifacts",
            "database",
            "cache",
            "monitoring",
            "api",
        }
        offenders: list[str] = []
        for path in (PROJECT_ROOT / "src/omnirank/core").glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                module = getattr(node, "module", None) or ""
                if isinstance(node, ast.ImportFrom) and module.startswith("omnirank."):
                    subpackage = module.split(".")[1]
                    if subpackage in forbidden:
                        offenders.append(f"{path.name} -> {module}")
        assert offenders == []

    def test_data_does_not_import_models_or_api(self):
        offenders: list[str] = []
        for path in (PROJECT_ROOT / "src/omnirank/data").glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                module = getattr(node, "module", None) or ""
                if isinstance(node, ast.ImportFrom) and module.startswith(
                    ("omnirank.models", "omnirank.api", "omnirank.retrieval")
                ):
                    offenders.append(f"{path.name} -> {module}")
        assert offenders == []


class TestNoPrintStatements:
    """Library code logs; only CLI entrypoints write to stdout."""

    def test_src_contains_no_print_calls(self):
        offenders: list[str] = []
        for path in (PROJECT_ROOT / "src").rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ):
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
        assert offenders == []


class TestProjectLayout:
    @pytest.mark.parametrize(
        "relative",
        [
            "pyproject.toml",
            "Makefile",
            "README.md",
            ".env.example",
            ".gitignore",
            "docker-compose.yml",
            "configs/base.yaml",
            "src/omnirank/database/schema.sql",
            "docs/architecture/system_architecture.md",
            "docs/data/data_contracts.md",
            "docs/api/api_contracts.md",
            "docs/phase_reports/phase_01_report.md",
        ],
    )
    def test_required_file_exists(self, relative):
        assert (PROJECT_ROOT / relative).is_file()

    @pytest.mark.parametrize("number", range(1, 9))
    def test_every_adr_exists_and_has_the_required_sections(self, number):
        matches = list((PROJECT_ROOT / "docs/adr").glob(f"ADR-{number:03d}-*.md"))
        assert len(matches) == 1, f"expected exactly one ADR-{number:03d}"
        text = matches[0].read_text()
        for section in (
            "## Status",
            "## Context",
            "## Decision",
            "## Alternatives",
            "## Consequences",
        ):
            assert section in text, f"{matches[0].name} is missing {section}"

    def test_env_example_contains_no_real_secret(self):
        text = (PROJECT_ROOT / ".env.example").read_text()
        assert "OMNIRANK__DATABASE__PASSWORD" in text
        assert "change-me-locally" in text

    def test_dotenv_is_gitignored(self):
        ignored = (PROJECT_ROOT / ".gitignore").read_text()
        assert "\n.env\n" in ignored
        assert "!.env.example" in ignored


class TestScripts:
    """Entrypoints must run, and must not pretend to do work they cannot."""

    def _invoke(self, script: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / script), *args],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=60,
            check=False,
        )

    def test_prepare_data_check_only_succeeds(self):
        result = self._invoke("prepare_data.py", "--check-only")
        assert result.returncode == 0

    def test_prepare_data_exits_nonzero_rather_than_claiming_success(self):
        result = self._invoke("prepare_data.py")
        assert result.returncode == 3
        assert "not_implemented" in result.stderr

    def test_train_rejects_an_unknown_model(self):
        assert self._invoke("train.py", "--model", "nonsense").returncode == 2

    def test_train_reports_the_phase_for_a_known_model(self):
        result = self._invoke("train.py", "--model", "lightgcn")
        assert result.returncode == 3
        assert "planned_phase=3" in result.stderr

    def test_evaluate_reports_a_missing_artifact(self):
        result = self._invoke("evaluate.py", "--model", "absent")
        assert result.returncode == 4
        assert "artifact_unavailable" in result.stderr

    def test_no_script_prints_a_metric(self):
        """Fabricated benchmark numbers are the one thing these must never emit."""
        for script in ("prepare_data.py", "train.py", "evaluate.py"):
            args = ("--model", "popularity") if script != "prepare_data.py" else ()
            output = self._invoke(script, *args)
            combined = (output.stdout + output.stderr).lower()
            for forbidden in ("ndcg@", "recall@0.", "precision@0."):
                assert forbidden not in combined
