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

    def test_importing_the_package_pulls_no_heavy_dependency(self):
        """Importing any module must not drag in faiss, lightgbm, or friends.

        Popularity is the terminal stage of the serving fallback chain and the
        evaluator scores every model, so a lightweight install has to work. The
        torch-backed models (`baselines.bpr`, `lightgcn`, `sasrec`) are the
        documented exceptions and are imported lazily by the runner and CLIs;
        they are skipped here rather than being allowed to pull torch in.

        FAISS is checked too, and it is the interesting case: the vector index
        module must import fine without it, because `_require_faiss` defers the
        import to first use.

        Run in a subprocess, for the same reason the next test is: this pytest
        session has already imported faiss and torch for other suites, so
        asserting against the in-process `sys.modules` would test the run order
        rather than the package.
        """
        skip = {
            "omnirank.models.baselines.bpr",
            "omnirank.models.lightgcn",
            "omnirank.models.lightgcn.model",
            "omnirank.models.sasrec",
            "omnirank.models.sasrec.model",
            "omnirank.retrieval.blended",
            "omnirank.retrieval.runner",
        }
        modules = [name for name in ALL_MODULES if name not in skip]
        script = (
            "import importlib, sys\n"
            f"for name in {modules!r}:\n"
            "    importlib.import_module(name)\n"
            "heavy = {'faiss', 'torch', 'lightgbm', 'sentence_transformers', 'mlflow', 'dvc'}\n"
            "print(sorted(heavy & set(sys.modules)))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "[]", result.stdout

    def test_evaluation_and_popularity_do_not_pull_torch(self):
        """Asserted in a subprocess, because this session may already hold torch."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys;"
                "import omnirank.evaluation;"
                "from omnirank.models.baselines import PopularityRecommender;"
                "print('torch' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=90,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False", "torch leaked into the core import path"

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
            "docs/phase_reports/phase_02_report.md",
            "configs/data/pixelrec50k.yaml",
            "scripts/download_pixelrec50k.py",
            "docs/data/pixelrec50k_overview.md",
            "docs/data/pixelrec50k_raw_schema.md",
            "docs/data/source_to_canonical_mapping.md",
            "docs/data/cleaning_rules.md",
            "docs/data/interaction_ordering.md",
            "docs/data/filtering_policy.md",
            "docs/data/temporal_splitting.md",
            "docs/data/leakage_prevention.md",
            "docs/data/processed_schemas.md",
            "docs/data/multimodal_feature_alignment.md",
            "docs/data/cold_start_evaluation.md",
            "docs/data/data_versioning.md",
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

    def test_prepare_data_validate_only_succeeds(self):
        """Phase 2 replaced the placeholder: --validate-only checks real files."""
        result = self._invoke(
            "prepare_data.py", "--config", "configs/data/pixelrec50k.yaml", "--validate-only"
        )
        # 0 when the dataset is downloaded, 3 when it is not. Both are correct;
        # what must never happen is a silent success without the source present.
        assert result.returncode in (0, 3)
        if result.returncode == 3:
            assert "download_pixelrec50k" in result.stderr

    def test_prepare_data_rejects_an_unknown_config(self):
        result = self._invoke("prepare_data.py", "--config", "data/nonexistent.yaml")
        assert result.returncode == 2
        assert "Configuration error" in result.stderr

    def test_download_script_dry_run_downloads_nothing(self):
        result = self._invoke("download_pixelrec50k.py", "--dry-run")
        assert result.returncode == 0
        assert "Dry run" in result.stdout
        # The licence must be visible before anyone downloads anything.
        assert "non-commercial" in result.stdout

    def test_download_script_never_offers_the_full_dataset(self):
        """Only the four PixelRec50K/feature ids are reachable from the script."""
        source = (PROJECT_ROOT / "scripts" / "download_pixelrec50k.py").read_text()
        assert source.count("RemoteFile(") == 4

    def test_train_rejects_an_unknown_model(self):
        """Phase 3 implements two models; argparse rejects anything else."""
        assert self._invoke("train.py", "--model", "nonsense", "--version", "x").returncode == 2

    def test_train_requires_a_version(self):
        assert self._invoke("train.py", "--model", "popularity").returncode == 2

    def test_evaluate_reports_a_missing_artifact(self):
        result = self._invoke("evaluate.py", "--model", "popularity", "--version", "does-not-exist")
        assert result.returncode == 2
        assert "artifact_not_found" in result.stderr

    def test_evaluate_rejects_a_non_full_protocol(self):
        """Sampled results must never be produced by the reporting path."""
        result = self._invoke(
            "evaluate.py", "--model", "popularity", "--version", "v", "--protocol", "sampled"
        )
        assert result.returncode == 2

    def test_final_stage_refuses_without_a_locked_configuration(self, tmp_path):
        """Test data must not be read before a configuration is locked."""
        import shutil

        selection = PROJECT_ROOT / "reports/metrics/phase_03/selected_configuration.json"
        backup = tmp_path / "selected_configuration.json"
        existed = selection.is_file()
        if existed:
            shutil.move(selection, backup)
        try:
            result = self._invoke("compare_baselines.py", "--stage", "final", "--skip-bpr")
            assert result.returncode == 3
            assert "no_selection" in result.stderr
        finally:
            if existed:
                shutil.move(backup, selection)

    def test_failing_runs_never_emit_a_metric(self):
        """A run that produced no measurement must not print one.

        Phase 3 scripts legitimately report real metrics on success - they come
        from the evaluator, computed from real recommendations. What must never
        happen is a metric appearing from a run that computed nothing, which is
        how a placeholder number ends up in a status update.
        """
        for script, args in (
            # A dataset that does not exist: cannot compute anything.
            ("prepare_data.py", ("--config", "data/nonexistent.yaml")),
            # An unregistered artifact: nothing to evaluate.
            ("evaluate.py", ("--model", "popularity", "--version", "does-not-exist")),
        ):
            output = self._invoke(script, *args)
            assert output.returncode != 0, script
            combined = (output.stdout + output.stderr).lower()
            for forbidden in ("ndcg@", "recall@2", "precision@2", "hit_rate@"):
                assert forbidden not in combined, (script, forbidden)
