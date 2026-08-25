#!/usr/bin/env python
"""Verify that Phase 5 is genuinely complete before Phase 6 builds on it.

    python scripts/validate_phase5.py
    python scripts/validate_phase5.py --json reports/metrics/phase_06/phase5_gate_report.json

Phase 6 puts a ranker, a reranker and an online service on top of Phase 5's
retrievers. Every one of those inherits Phase 5's candidate pool, so a Phase 5
that is half-built does not announce itself in Phase 6 -- it shows up as a
ranker that mysteriously cannot reach cold items, or a bundle that loads an
index nothing can query. This gate exists so that failure is loud and early
rather than diffuse and late.

Checks are graded. A **critical** failure means Phase 6 must not start. A
**warning** means something is missing that Phase 6 can tolerate but the Phase 5
report should not claim.

Exit codes:
  0  every critical check passed
  1  at least one critical check failed
  2  the repository could not be inspected at all
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GATE_FAILED_EXIT = 1
INSPECTION_ERROR_EXIT = 2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Source files Phase 5 must provide. `dataset`, `training` and `persistence`
#: may legitimately live inside `model.py`, so they are checked as a group.
REQUIRED_SOURCES = (
    "src/omnirank/models/two_tower/__init__.py",
    "src/omnirank/models/two_tower/config.py",
    "src/omnirank/models/two_tower/model.py",
    "src/omnirank/models/two_tower/losses.py",
)
OPTIONAL_SOURCES = (
    "src/omnirank/models/two_tower/dataset.py",
    "src/omnirank/models/two_tower/training.py",
    "src/omnirank/models/two_tower/persistence.py",
    "src/omnirank/models/two_tower/features.py",
)

REQUIRED_CONFIGS = (
    "configs/models/two_tower.yaml",
    "configs/models/phase5_selected.yaml",
    "configs/features/pixelrec_published.yaml",
)

REQUIRED_COMMANDS = (
    "scripts/prepare_multimodal_features.py",
    "scripts/compare_multimodal_retrievers.py",
    "scripts/build_index.py",
)

REQUIRED_EVIDENCE = (
    "docs/phase_reports/phase_05_report.md",
    "reports/metrics/phase_05/selected_configuration.json",
)

FEATURE_MANIFEST = "data/processed/pixelrec50k/features/multimodal_feature_manifest.json"


@dataclass
class Check:
    """One gate check and what it found."""

    name: str
    passed: bool
    critical: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Report-ready payload."""
        return {
            "check": self.name,
            "passed": self.passed,
            "severity": "critical" if self.critical else "warning",
            "detail": self.detail,
        }


@dataclass
class GateResult:
    """Every check, plus the verdict they add up to."""

    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool, *, critical: bool = True, detail: str = "") -> None:
        """Record one check."""
        self.checks.append(Check(name=name, passed=passed, critical=critical, detail=detail))

    @property
    def critical_failures(self) -> list[Check]:
        """Checks that block Phase 6."""
        return [check for check in self.checks if check.critical and not check.passed]

    @property
    def warnings(self) -> list[Check]:
        """Checks that do not block but must not be claimed as done."""
        return [check for check in self.checks if not check.critical and not check.passed]

    @property
    def passed(self) -> bool:
        """Whether Phase 6 may begin."""
        return not self.critical_failures

    def to_dict(self) -> dict[str, Any]:
        """Report-ready payload."""
        return {
            "gate": "phase_05_completion",
            "passed": self.passed,
            "checks_run": len(self.checks),
            "checks_passed": sum(1 for check in self.checks if check.passed),
            "critical_failures": len(self.critical_failures),
            "warnings": len(self.warnings),
            "results": [check.to_dict() for check in self.checks],
        }


def _check_paths(result: GateResult, label: str, paths: tuple[str, ...], *, critical: bool) -> None:
    """Assert a group of paths exists, naming the ones that do not."""
    missing = [path for path in paths if not (PROJECT_ROOT / path).exists()]
    result.add(
        label,
        not missing,
        critical=critical,
        detail="all present" if not missing else f"missing: {', '.join(missing)}",
    )


def check_sources(result: GateResult) -> None:
    """The two-tower package exists and carries the required modules."""
    _check_paths(result, "two_tower source modules", REQUIRED_SOURCES, critical=True)
    present = [path for path in OPTIONAL_SOURCES if (PROJECT_ROOT / path).exists()]
    result.add(
        "two_tower supporting modules",
        bool(present),
        critical=False,
        detail=(
            f"present: {', '.join(Path(p).name for p in present)}"
            if present
            else "none of dataset/training/persistence/features exist as separate modules"
        ),
    )


def check_model_class(result: GateResult) -> None:
    """The package exposes a fitted-capable CandidateGenerator."""
    try:
        from omnirank.models.base import CandidateGenerator
        from omnirank.models.two_tower import MultimodalTwoTower  # type: ignore[attr-defined]
    except Exception as exc:  # any import failure is the same verdict
        result.add(
            "MultimodalTwoTower importable",
            False,
            detail=f"{type(exc).__name__}: {str(exc)[:160]}",
        )
        return
    result.add(
        "MultimodalTwoTower importable",
        True,
        detail="exposed from omnirank.models.two_tower",
    )
    result.add(
        "MultimodalTwoTower implements CandidateGenerator",
        issubclass(MultimodalTwoTower, CandidateGenerator),
        detail="required so the ranker and fusion treat it like every other source",
    )
    required_methods = (
        "fit",
        "recommend",
        "recommend_batch",
        "score",
        "save",
        "load",
        "item_embeddings",
        "metadata",
    )
    missing = [name for name in required_methods if not hasattr(MultimodalTwoTower, name)]
    result.add(
        "MultimodalTwoTower interface surface",
        not missing,
        detail="all present" if not missing else f"missing: {', '.join(missing)}",
    )


def check_configs(result: GateResult) -> None:
    """Phase 5 configuration and the locked selection exist."""
    _check_paths(result, "Phase 5 configurations", REQUIRED_CONFIGS, critical=True)


def check_commands(result: GateResult) -> None:
    """The Phase 5 CLIs exist."""
    _check_paths(result, "Phase 5 commands", REQUIRED_COMMANDS, critical=True)


def check_evidence(result: GateResult) -> None:
    """The report and locked configuration exist."""
    _check_paths(result, "Phase 5 evidence", REQUIRED_EVIDENCE, critical=True)


def check_features(result: GateResult) -> None:
    """Aligned multimodal features exist and describe real coverage."""
    manifest_path = PROJECT_ROOT / FEATURE_MANIFEST
    if not manifest_path.is_file():
        result.add(
            "multimodal feature manifest",
            False,
            detail=f"missing: {FEATURE_MANIFEST}",
        )
        return
    manifest = json.loads(manifest_path.read_text())
    modalities = manifest.get("modalities", {})
    available = {name for name, block in modalities.items() if block.get("available")}
    result.add(
        "multimodal feature manifest",
        bool(available),
        detail=f"available modalities: {sorted(available) or 'none'}",
    )
    coverage = {
        name: round(float(block.get("coverage", 0.0)), 6)
        for name, block in modalities.items()
        if block.get("available")
    }
    result.add(
        "feature coverage is non-zero",
        any(value > 0 for value in coverage.values()),
        detail=f"coverage: {coverage or 'none'}",
    )
    result.add(
        "features carry mapping identity",
        bool(manifest.get("item_mapping_checksum")),
        detail=(
            "item_mapping_checksum present"
            if manifest.get("item_mapping_checksum")
            else "no mapping checksum; a store cannot prove which items it describes"
        ),
    )


def _registered(kind: str, model: str) -> list[Path]:
    """Registered artifact metadata files for one model."""
    root = PROJECT_ROOT / "artifacts" / kind / model
    return sorted(root.glob("*.json")) if root.is_dir() else []


def check_artifacts(result: GateResult) -> dict[str, Any] | None:
    """A final two-tower model and its index are registered."""
    metadata_files = _registered("metadata", "two_tower")
    result.add(
        "registered two-tower model",
        bool(metadata_files),
        detail=(
            f"found: {[p.name for p in metadata_files]}"
            if metadata_files
            else "no artifacts/metadata/two_tower/*.json"
        ),
    )
    index_root = PROJECT_ROOT / "artifacts" / "indexes"
    index_dirs = sorted(index_root.glob("*/two_tower/*")) if index_root.is_dir() else []
    result.add(
        "registered two-tower FAISS index",
        bool(index_dirs),
        detail=(
            f"found: {[str(p.relative_to(PROJECT_ROOT)) for p in index_dirs]}"
            if index_dirs
            else "no artifacts/indexes/*/two_tower/*"
        ),
    )
    if not metadata_files:
        return None
    metadata: dict[str, Any] = json.loads(metadata_files[-1].read_text())
    return metadata


def check_smoke_recommendation(result: GateResult, metadata: dict[str, Any] | None) -> None:
    """Load the registered model and ask it for a recommendation.

    A model that loads but cannot answer is the failure this catches: every
    static check above can pass while the artifact is unusable.
    """
    if metadata is None:
        result.add(
            "saved-model smoke recommendation",
            False,
            detail="skipped: no registered two-tower model to load",
        )
        result.add(
            "cold item present in the index",
            False,
            detail="skipped: no registered two-tower model to load",
        )
        return

    artifact_path = PROJECT_ROOT / str(metadata.get("artifact_path", ""))
    try:
        from omnirank.models.two_tower import MultimodalTwoTower  # type: ignore[attr-defined]

        model = MultimodalTwoTower.load(artifact_path, device="cpu")
        users = list(getattr(model, "known_users", []) or [])[:1]
        recommendations = model.recommend(users[0], 5) if users else []
        result.add(
            "saved-model smoke recommendation",
            bool(recommendations),
            detail=f"returned {len(recommendations)} candidates",
        )
    except Exception as exc:  # any failure is the same verdict
        result.add(
            "saved-model smoke recommendation",
            False,
            detail=f"{type(exc).__name__}: {str(exc)[:160]}",
        )
        result.add(
            "cold item present in the index",
            False,
            detail="skipped: model did not load",
        )
        return

    _check_cold_item(result, model)


def _check_cold_item(result: GateResult, model: Any) -> None:
    """A content-representable cold item must be retrievable.

    This is Phase 5's whole purpose. A two-tower model whose catalogue excludes
    cold items is a slower LightGCN, and every cold metric downstream would
    read zero for a reason no warm number reveals.
    """
    cold_path = (
        PROJECT_ROOT / "data/processed/pixelrec50k/evaluation_slices/items_cold_start.parquet"
    )
    if not cold_path.is_file():
        result.add(
            "cold item present in the index",
            False,
            critical=False,
            detail="cold-start slice not found; cannot verify",
        )
        return
    try:
        import pandas as pd

        cold = pd.read_parquet(cold_path)
        cold_ids = {int(value) for value in cold["entity_id"]}
        catalogue = set(getattr(model, "fit_item_catalogue", set()))
        overlap = cold_ids & catalogue
        result.add(
            "cold item present in the index",
            bool(overlap),
            detail=(
                f"{len(overlap)} of {len(cold_ids)} cold items are in the model catalogue"
                if overlap
                else "no cold item is retrievable; the content path is not doing its job"
            ),
        )
    except Exception as exc:
        result.add(
            "cold item present in the index",
            False,
            detail=f"{type(exc).__name__}: {str(exc)[:160]}",
        )


def check_compatibility(result: GateResult, metadata: dict[str, Any] | None) -> None:
    """Model, index, feature and mapping identities agree."""
    if metadata is None:
        result.add(
            "model/index/feature/mapping compatibility",
            False,
            detail="skipped: no registered two-tower model",
        )
        return
    manifest_path = PROJECT_ROOT / FEATURE_MANIFEST
    problems: list[str] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        recorded = (metadata.get("id_mapping_fingerprints") or {}).get("item")
        expected = manifest.get("item_mapping_checksum")
        if recorded and expected and recorded != expected:
            problems.append("model mapping checksum differs from the feature manifest")
    else:
        problems.append("feature manifest missing")
    result.add(
        "model/index/feature/mapping compatibility",
        not problems,
        detail="consistent" if not problems else "; ".join(problems),
    )


def check_report_claims(result: GateResult) -> None:
    """The Phase 5 report exists and records a cold-item result."""
    report = PROJECT_ROOT / "docs/phase_reports/phase_05_report.md"
    if not report.is_file():
        result.add(
            "Phase 5 report records cold-item recall",
            False,
            detail="report does not exist",
        )
        return
    text = report.read_text().lower()
    result.add(
        "Phase 5 report records cold-item recall",
        "cold" in text and "recall" in text,
        detail="cold recall discussed" if "cold" in text else "no cold-item discussion found",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--json",
        default="reports/metrics/phase_06/phase5_gate_report.json",
        help="Where to write the machine-readable gate report.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the verdict and failures.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    result = GateResult()

    try:
        check_sources(result)
        check_model_class(result)
        check_configs(result)
        check_commands(result)
        check_evidence(result)
        check_features(result)
        metadata = check_artifacts(result)
        check_compatibility(result, metadata)
        check_smoke_recommendation(result, metadata)
        check_report_claims(result)
    except Exception as exc:  # the gate itself failing is distinct
        print(f"Gate could not run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return INSPECTION_ERROR_EXIT

    payload = result.to_dict()
    destination = Path(args.json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    width = max(len(check.name) for check in result.checks)
    if not args.quiet:
        print("\nPhase 5 completion gate\n")
        for check in result.checks:
            mark = "PASS" if check.passed else ("FAIL" if check.critical else "WARN")
            print(f"  [{mark}] {check.name.ljust(width)}  {check.detail}")

    print(
        f"\n{payload['checks_passed']}/{payload['checks_run']} checks passed  "
        f"({payload['critical_failures']} critical failures, {payload['warnings']} warnings)"
    )
    print(f"Report written to {destination}")

    if result.passed:
        print("\nGATE PASSED - Phase 6 may begin.")
        return 0

    print("\nGATE FAILED - Phase 6 must not begin. Critical failures:")
    for check in result.critical_failures:
        print(f"  - {check.name}: {check.detail}")
    return GATE_FAILED_EXIT


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        raise SystemExit(main())
    raise SystemExit(INSPECTION_ERROR_EXIT)
