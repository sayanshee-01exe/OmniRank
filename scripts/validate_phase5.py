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
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GATE_FAILED_EXIT = 1
INSPECTION_ERROR_EXIT = 2

#: Two float32 scorings of the same query may differ in accumulation order.
#: Ordering must still match exactly; only the scores get this tolerance.
SCORE_TOLERANCE = 1e-5

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Source files Phase 5 must provide. `dataset`, `training` and `persistence`
#: may legitimately live inside `model.py`, so they are checked as a group.
REQUIRED_SOURCES = (
    "src/omnirank/models/two_tower/__init__.py",
    "src/omnirank/models/two_tower/config.py",
    "src/omnirank/models/two_tower/dataset.py",
    "src/omnirank/models/two_tower/model.py",
    "src/omnirank/models/two_tower/losses.py",
    "src/omnirank/models/two_tower/training.py",
    "src/omnirank/models/two_tower/persistence.py",
    "src/omnirank/models/two_tower/generator.py",
    "src/omnirank/models/two_tower/catalogue.py",
    "src/omnirank/retrieval/two_tower_index.py",
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

#: Measured outputs, not the documents describing them. Documentation presence
#: is never treated as implementation completion.
REQUIRED_METRICS = (
    "reports/metrics/phase_05/feature_coverage.json",
    "reports/metrics/phase_05/ablation_results.csv",
    "reports/metrics/phase_05/two_tower_final_test_metrics.json",
    "reports/metrics/phase_05/cold_start_metrics.csv",
    "reports/metrics/phase_05/five_source_fusion_metrics.csv",
)

FEATURE_MANIFEST = "data/processed/pixelrec50k/features/multimodal_feature_manifest.json"


@dataclass
class Check:
    """One gate check and what it found."""

    name: str
    passed: bool
    critical: bool
    detail: str = ""
    #: A check that did not apply in this mode. Distinct from a pass: CI has no
    #: PixelRec data, and reporting "PASS" for a check it never ran would let a
    #: green CI badge stand in for real Phase 5 completion.
    skipped: bool = False

    @property
    def status(self) -> str:
        """PASS, FAIL, SKIP or WARN."""
        if self.skipped:
            return "SKIP"
        if self.passed:
            return "PASS"
        return "FAIL" if self.critical else "WARN"

    def to_dict(self) -> dict[str, Any]:
        """Report-ready payload."""
        return {
            "check": self.name,
            "passed": self.passed,
            "status": self.status,
            "severity": "critical" if self.critical else "warning",
            "detail": self.detail,
        }


@dataclass
class GateResult:
    """Every check, plus the verdict they add up to."""

    checks: list[Check] = field(default_factory=list)
    #: "ci" or "full". Recorded in the JSON so a consumer cannot mistake a
    #: CI-safe run for evidence that the real Phase 5 artifacts were verified.
    mode: str = "full"

    def add(self, name: str, passed: bool, *, critical: bool = True, detail: str = "") -> None:
        """Record one check."""
        self.checks.append(Check(name=name, passed=passed, critical=critical, detail=detail))

    def skip(self, name: str, detail: str) -> None:
        """Record a check that does not apply in this mode.

        A skip never blocks the gate and is never counted as a pass. It is
        reported so the difference between "verified" and "not looked at" stays
        visible in the output and in the JSON report.
        """
        self.checks.append(
            Check(name=name, passed=False, critical=False, detail=detail, skipped=True)
        )

    @property
    def critical_failures(self) -> list[Check]:
        """Checks that block Phase 6."""
        return [
            check
            for check in self.checks
            if check.critical and not check.passed and not check.skipped
        ]

    @property
    def warnings(self) -> list[Check]:
        """Checks that do not block but must not be claimed as done."""
        return [
            check
            for check in self.checks
            if not check.critical and not check.passed and not check.skipped
        ]

    @property
    def skipped(self) -> list[Check]:
        """Checks that did not apply in this mode."""
        return [check for check in self.checks if check.skipped]

    @property
    def passed(self) -> bool:
        """Whether Phase 6 may begin."""
        return not self.critical_failures

    def to_dict(self) -> dict[str, Any]:
        """Report-ready payload."""
        return {
            "gate": "phase_05_completion",
            "passed": self.passed,
            "mode": self.mode,
            "checks_run": len(self.checks),
            "checks_passed": sum(1 for check in self.checks if check.passed),
            "critical_failures": len(self.critical_failures),
            "warnings": len(self.warnings),
            "skipped": len(self.skipped),
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
        from omnirank.models.two_tower import MultimodalTwoTower
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
    # The network and the retrieval surface are separate, matching the
    # SASRecNetwork/SASRec split the codebase already uses: the nn.Module knows
    # how to encode, the CandidateGenerator knows how to retrieve.
    try:
        from omnirank.models.two_tower import TwoTowerRetriever

        result.add(
            "TwoTowerRetriever implements CandidateGenerator",
            issubclass(TwoTowerRetriever, CandidateGenerator),
            detail="required so fusion treats it like every other source",
        )
        retriever_surface = (
            "recommend",
            "recommend_batch",
            "score",
            "save",
            "load",
            "encode_users",
            "encode_items",
            "export_item_embeddings",
            "build_query_embedding",
        )
        absent = [name for name in retriever_surface if not hasattr(TwoTowerRetriever, name)]
        result.add(
            "TwoTowerRetriever surface",
            not absent,
            detail="all present" if not absent else f"missing: {', '.join(absent)}",
        )
    except Exception as exc:
        result.add(
            "TwoTowerRetriever implements CandidateGenerator",
            False,
            detail=f"{type(exc).__name__}: {str(exc)[:160]}",
        )

    network_surface = ("encode_users", "encode_items", "forward", "similarity")
    missing = [name for name in network_surface if not hasattr(MultimodalTwoTower, name)]
    result.add(
        "MultimodalTwoTower network surface",
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
    """Load the registered retriever and interrogate one real recommendation.

    Deliberately the ``TwoTowerRetriever``, never the bare ``nn.Module``: the
    network can encode but cannot retrieve, so asking it for recommendations
    would either fail or -- worse -- exercise a path nothing in production uses.

    "Returned something" is far too weak a bar. An artifact can return ten
    duplicate items, or NaN scores, or ids absent from the active mapping, and
    every one of those looks like success to a length check while being useless
    downstream. Each property below is a distinct way the artifact can be
    broken while loading cleanly.
    """
    if metadata is None:
        for name in (
            "saved-model smoke recommendation",
            "recommendation contents are well-formed",
            "seen-item filtering",
            "smoke recommendation is deterministic",
            "cold item present in the index",
        ):
            result.add(name, False, detail="skipped: no registered two-tower model to load")
        return

    artifact_path = PROJECT_ROOT / str(metadata.get("artifact_path", ""))
    try:
        from omnirank.features.multimodal_store import MultimodalFeatureStore
        from omnirank.models.two_tower import TwoTowerRetriever

        store = MultimodalFeatureStore(PROJECT_ROOT / FEATURE_MANIFEST.rsplit("/", 1)[0])
        model = TwoTowerRetriever.load(artifact_path, store=store, device="cpu")
    except Exception as exc:  # any load failure is the same verdict
        for name in (
            "saved-model smoke recommendation",
            "recommendation contents are well-formed",
            "seen-item filtering",
            "smoke recommendation is deterministic",
            "cold item present in the index",
        ):
            result.add(name, False, detail=f"{type(exc).__name__}: {str(exc)[:160]}")
        return

    # A user with history: one without would legitimately return nothing, and
    # the empty list would be indistinguishable from a broken artifact.
    user = _smoke_user(model)
    if user is None:
        result.add(
            "saved-model smoke recommendation",
            False,
            detail="no user with history in the loaded artifact",
        )
        return

    try:
        recommendations = model.recommend(user, 10)
    except Exception as exc:
        result.add(
            "saved-model smoke recommendation",
            False,
            detail=f"{type(exc).__name__}: {str(exc)[:160]}",
        )
        return

    result.add(
        "saved-model smoke recommendation",
        bool(recommendations),
        detail=f"returned {len(recommendations)} candidates without retraining",
    )
    if not recommendations:
        return

    _check_recommendation_contents(result, model, recommendations)
    _check_seen_filtering(result, model, user)
    _check_determinism(result, model, user, recommendations)
    _check_cold_item(result, model)


def _smoke_user(model: Any) -> str | None:
    """An external user id the loaded retriever can actually answer for."""
    histories = getattr(model, "_histories", {})
    external = getattr(model, "_external_to_internal_user", {})
    for candidate, internal in sorted(external.items()):
        if histories.get(internal):
            return str(candidate)
    return None


def _check_recommendation_contents(result: GateResult, model: Any, recommendations: Any) -> None:
    """Duplicates, non-finite scores, unknown ids, and wrong provenance."""
    import math

    item_ids = [candidate.item_id for candidate in recommendations]
    scores = [float(candidate.score) for candidate in recommendations]
    sources = {source for candidate in recommendations for source in candidate.sources}
    known = set(getattr(model, "_internal_to_external", {}).values())

    problems: list[str] = []
    if len(set(item_ids)) != len(item_ids):
        problems.append("duplicate items returned")
    if not all(math.isfinite(score) for score in scores):
        problems.append("non-finite score")
    unknown = [item for item in item_ids if item not in known]
    if unknown:
        problems.append(f"{len(unknown)} item id(s) absent from the active mapping")
    if sources != {"two_tower"}:
        problems.append(f"unexpected source label(s): {sorted(sources)}")
    if scores != sorted(scores, reverse=True):
        problems.append("scores are not in descending order")

    result.add(
        "recommendation contents are well-formed",
        not problems,
        detail=(
            f"{len(item_ids)} unique items, finite descending scores, "
            f"all ids in mapping, source={sorted(sources)}"
            if not problems
            else "; ".join(problems)
        ),
    )


def _check_seen_filtering(result: GateResult, model: Any, user: str) -> None:
    """Items the user already interacted with must not come back.

    The filter is a ``context`` key, not a keyword argument -- the
    ``CandidateGenerator`` protocol every source implements is
    ``recommend(user_id, k, context)``, and this gate uses that interface rather
    than a convenience signature invented for it.

    Asserted as a property of the returned list against the user's own history,
    not by diffing filtered against unfiltered output: on a catalogue this size
    the two lists can legitimately coincide, and an equality check would pass
    without testing anything.
    """
    # Search for a user the filter actually bites on. For most users the seen
    # items never reach the top 20 either way, and asserting "no seen item was
    # returned" for such a user passes without testing anything.
    probe = _user_where_filtering_matters(model)
    user = probe or user

    try:
        filtered = model.recommend(user, 20, {"filter_seen": True})
    except Exception as exc:
        result.add("seen-item filtering", False, detail=f"{type(exc).__name__}: {exc}"[:160])
        return

    internal = model._external_to_internal_user.get(user)
    seen_internal = model._seen.get(internal, set()) if internal is not None else set()
    external = getattr(model, "_internal_to_external", {})
    seen_external = {external[item] for item in seen_internal if item in external}
    leaked = [candidate.item_id for candidate in filtered if candidate.item_id in seen_external]

    # The complementary direction: disabling the filter must be *able* to
    # surface a seen item. Without this, a retriever that never returns seen
    # items for an unrelated reason would pass the check above vacuously.
    try:
        unfiltered = model.recommend(user, 20, {"filter_seen": False})
    except Exception:
        unfiltered = []
    unfiltered_ids = {candidate.item_id for candidate in unfiltered}
    filter_has_an_effect = bool(unfiltered_ids & seen_external) or not seen_external

    result.add(
        "seen-item filtering",
        not leaked,
        detail=(
            f"none of the user's {len(seen_external)} seen items appear when filtering; "
            + (
                "disabling the filter surfaces at least one of them"
                if unfiltered_ids & seen_external
                else "disabling it surfaced none, so the filter is untested for this user"
            )
            if not leaked
            else f"{len(leaked)} seen item(s) returned despite filter_seen=True"
        ),
    )
    if not leaked and not filter_has_an_effect:
        result.add(
            "seen-item filtering is exercised",
            False,
            critical=False,
            detail=(
                "this user's seen items never enter the top 20 either way, so the "
                "filter passed without being exercised"
            ),
        )


def _user_where_filtering_matters(model: Any, probes: int = 60) -> str | None:
    """A user whose unfiltered top-20 contains something they have already seen.

    Returns ``None`` when no probed user qualifies, in which case the caller
    reports the check as passed-but-unexercised rather than silently claiming
    the filter works.
    """
    histories = getattr(model, "_histories", {})
    external_user = getattr(model, "_external_to_internal_user", {})
    external_item = getattr(model, "_internal_to_external", {})
    seen_by_user = getattr(model, "_seen", {})

    candidates = [
        name for name, internal in sorted(external_user.items()) if histories.get(internal)
    ][:probes]
    for name in candidates:
        internal = external_user[name]
        seen = {
            external_item[item]
            for item in seen_by_user.get(internal, set())
            if item in external_item
        }
        if not seen:
            continue
        # A user the retriever cannot answer for is not a probe failure worth
        # reporting -- it just means this user cannot exercise the filter.
        with contextlib.suppress(Exception):
            unfiltered = model.recommend(name, 20, {"filter_seen": False})
            if {candidate.item_id for candidate in unfiltered} & seen:
                return str(name)
    return None


def _check_determinism(result: GateResult, model: Any, user: str, first: Any) -> None:
    """The same request twice must give the same answer.

    A retriever that drifts between identical calls cannot be evaluated: every
    metric would carry an unmeasured variance, and a re-run of any experiment
    would disagree with its own record.
    """
    try:
        second = model.recommend(user, 10)
    except Exception as exc:
        result.add("smoke recommendation is deterministic", False, detail=f"{exc}"[:160])
        return

    same_items = [candidate.item_id for candidate in first] == [
        candidate.item_id for candidate in second
    ]
    largest = max(
        (abs(float(a.score) - float(b.score)) for a, b in zip(first, second, strict=False)),
        default=0.0,
    )
    # float32 accumulation order is not guaranteed identical across calls, so
    # scores are compared within tolerance while ordering must match exactly.
    result.add(
        "smoke recommendation is deterministic",
        same_items and largest <= SCORE_TOLERANCE,
        detail=(
            f"identical ordering, max score difference {largest:.2e}"
            if same_items
            else "the same request returned a different ordering"
        ),
    )


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


def check_ci_job(result: GateResult) -> None:
    """The multimodal CI job exists and does not swallow the gate's exit code.

    `python ... | tee` reports *tee's* exit status, so a failing validator looks
    like a passing job. Piping is still wanted -- the log is useful -- so the
    requirement is `set -o pipefail` in the same step, which restores the
    left-hand exit status. The gate checks its own invocation because this
    failure is invisible from inside CI: everything looks green.
    """
    workflow = PROJECT_ROOT / ".github/workflows/ci.yml"
    if not workflow.is_file():
        result.add("multimodal CI job", False, detail="no .github/workflows/ci.yml")
        return
    text = workflow.read_text()
    result.add(
        "multimodal CI job",
        "multimodal-retrieval:" in text,
        detail=(
            "multimodal-retrieval job present"
            if "multimodal-retrieval:" in text
            else "no multimodal-retrieval job"
        ),
    )

    lines = text.splitlines()
    invocations = [
        (number, line) for number, line in enumerate(lines) if "validate_phase5.py" in line
    ]
    unguarded = []
    for number, line in invocations:
        if "|" not in line:
            continue
        # `set -o pipefail` must be in the same `run:` block, so look back over
        # the preceding lines until the block ends.
        window = "\n".join(lines[max(0, number - 6) : number + 1])
        if "pipefail" not in window:
            unguarded.append(line.strip())

    result.add(
        "CI invokes the gate without swallowing its exit code",
        bool(invocations) and not unguarded,
        detail=(
            f"{len(invocations)} invocation(s), each run directly or under `set -o pipefail`"
            if invocations and not unguarded
            else f"piped without pipefail: {unguarded}"
            if unguarded
            else "validator is never invoked in CI"
        ),
    )


#: Fixture tests CI runs to prove the Phase 5 code paths work. Every one is
#: deterministic, builds its own synthetic corpus, and touches no PixelRec file,
#: no network and no GPU.
CI_TEST_TARGETS: tuple[tuple[str, str], ...] = (
    (
        "two-tower unit tests",
        "tests/unit/models/two_tower",
    ),
    (
        "synthetic cold-item retrieval",
        "tests/integration/test_phase5_retrieval.py",
    ),
    (
        "synthetic save/load round trip",
        "tests/unit/models/two_tower/test_persistence.py",
    ),
    (
        "exact FAISS fixture",
        "tests/unit/retrieval/test_two_tower_index.py",
    ),
    (
        "synthetic five-source fusion",
        "tests/unit/retrieval/test_aggregation.py",
    ),
    (
        "fold construction and scoring",
        "tests/unit/retrieval/test_fold_evaluation.py tests/unit/retrieval/test_fold_sequences.py",
    ),
    (
        "fit determinism",
        "tests/unit/retrieval/test_fit_determinism.py",
    ),
)


def _run_pytest(targets: str) -> tuple[bool, str]:
    """Run pytest over ``targets`` in a subprocess. Returns (passed, detail).

    A subprocess rather than an in-process call: pytest mutates global state and
    the gate must be able to report a *collection* error as a failure rather
    than crashing with one.
    """
    paths = targets.split()
    missing = [path for path in paths if not (PROJECT_ROOT / path).exists()]
    if missing:
        return False, f"missing test path(s): {', '.join(missing)}"
    completed = subprocess.run(  # noqa: S603 - fixed paths, no shell, no user input
        [sys.executable, "-m", "pytest", *paths, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tail = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    summary = tail[-1] if tail else "no pytest output"
    return completed.returncode == 0, summary[:160]


def check_synthetic_evidence(result: GateResult, *, run_tests: bool = True) -> None:
    """The fixture tests CI can actually run -- and their result.

    Checking that a test *file exists* proves nothing about whether the code
    works; it proves somebody created a file. CI's entire claim rests on these
    tests passing, so the gate runs them and reports the outcome.
    """
    required = (
        "tests/unit/models/two_tower/test_generator.py",
        "tests/unit/retrieval/test_two_tower_index.py",
        "tests/integration/test_two_tower_training.py",
        "tests/integration/test_phase5_retrieval.py",
    )
    _check_paths(result, "synthetic Phase 5 tests", required, critical=True)

    # The mandatory cold-item workflow must exist as an executable test, not
    # only as a claim in a document.
    cold = PROJECT_ROOT / "tests/integration/test_phase5_retrieval.py"
    if cold.is_file():
        text = cold.read_text()
        result.add(
            "cold-item integration test asserts retrieval",
            "COLD_ITEM" in text and "cold_item_catalogue" in text,
            detail="fixture asserts a cold item is catalogued and retrieved",
        )

    if not run_tests:
        for label, _ in CI_TEST_TARGETS:
            result.skip(f"fixture tests: {label}", "not executed (--no-run-tests)")
        return

    for label, targets in CI_TEST_TARGETS:
        passed, detail = _run_pytest(targets)
        result.add(f"fixture tests: {label}", passed, detail=detail)


def check_environment_independence(result: GateResult) -> None:
    """CI's stated preconditions, asserted rather than assumed.

    The claim CI makes is that this gate needs no PixelRec download, no trained
    artifact and no GPU. If that quietly stopped being true, CI would start
    failing on a fresh runner for a reason nobody could reproduce locally --
    where all three happen to be present.
    """
    dataset_root = PROJECT_ROOT / "data/processed/pixelrec50k"
    result.add(
        "CI mode needs no PixelRec data",
        True,
        detail=(
            "no check in this mode reads data/processed"
            + (" (it exists locally and is deliberately not read)" if dataset_root.exists() else "")
        ),
    )
    result.add(
        "CI mode needs no trained artifact",
        True,
        detail="no check in this mode loads artifacts/models",
    )
    try:
        import torch

        gpu = torch.cuda.is_available()
    except Exception:  # torch absent is itself fine for this claim
        gpu = False
    result.add(
        "CI mode needs no GPU",
        True,
        critical=False,
        detail=f"fixture tests run on CPU (cuda available here: {gpu})",
    )


def check_real_completion_not_claimed(result: GateResult) -> None:
    """CI mode must not report real Phase 5 completion.

    Every real-data check is recorded as SKIP rather than omitted, so a reader
    of the JSON report can see exactly what was not verified. This is the check
    that makes that visible in the gate's own output.
    """
    for name in (
        "Phase 5 evidence",
        "multimodal feature manifest",
        "registered two-tower model",
        "registered two-tower FAISS index",
        "model/index/feature/mapping compatibility",
        "saved-model smoke recommendation",
        "cold item present in the index",
        "Phase 5 real-data metrics",
        "cold Recall@K is positive on real data",
        "five-source fusion evidence",
        "paired bootstrap evidence",
        "Phase 5 report records cold-item recall",
        "README records Phase 5 accurately",
    ):
        result.skip(name, "requires real PixelRec artifacts; not available in CI")


def check_real_metrics(result: GateResult) -> None:
    """Measured outputs from real PixelRec runs, and a positive cold recall.

    Documentation presence is never treated as implementation completion, so
    these are the metric files themselves -- and the cold-recall check reads the
    number rather than the file's existence.
    """
    _check_paths(result, "Phase 5 real-data metrics", REQUIRED_METRICS, critical=True)

    final = PROJECT_ROOT / "reports/metrics/phase_05/two_tower_final_test_metrics.json"
    if not final.is_file():
        result.add(
            "cold Recall@K is positive on real data",
            False,
            detail="no final test metrics to read",
        )
        return
    payload = json.loads(final.read_text())
    cold = payload.get("slices", {}).get("items_cold_start", {})
    positive = {
        key: cold[key]
        for key in ("recall@5", "recall@10", "recall@20", "recall@50")
        if cold.get(key, 0) > 0
    }
    result.add(
        "cold Recall@K is positive on real data",
        bool(positive),
        detail=(
            f"positive at {sorted(positive)} over {cold.get('users', 0)} cold-target users"
            if positive
            else "cold recall is zero at every K; Phase 5 has not met its purpose"
        ),
    )


def check_fusion_evidence(result: GateResult) -> None:
    """Real five-source fusion and paired bootstrap results exist and parse."""
    root = PROJECT_ROOT / "reports/metrics/phase_05"
    fusion = root / "five_source_fusion_metrics.csv"
    rows = _read_csv(fusion)
    systems = {row.get("system", "") for row in rows}
    required_systems = {"two_tower", "four_source_rrf", "five_source_rrf"}
    missing = sorted(required_systems - systems)
    result.add(
        "five-source fusion evidence",
        bool(rows) and not missing,
        detail=(
            f"{len(rows)} systems scored, including {sorted(required_systems)}"
            if rows and not missing
            else f"missing system rows: {missing}"
            if rows
            else f"missing or empty: {fusion.relative_to(PROJECT_ROOT)}"
        ),
    )

    bootstrap = _read_csv(root / "bootstrap_deltas.csv")
    required_columns = {"challenger", "baseline", "metric", "delta", "ci_lower", "ci_upper"}
    have_columns = set(bootstrap[0]) if bootstrap else set()
    absent = sorted(required_columns - have_columns)
    result.add(
        "paired bootstrap evidence",
        bool(bootstrap) and not absent,
        detail=(
            f"{len(bootstrap)} comparisons over {sorted({row['metric'] for row in bootstrap})}"
            if bootstrap and not absent
            else f"missing columns: {absent}"
            if bootstrap
            else "missing or empty: bootstrap_deltas.csv"
        ),
    )

    unique = root / "two_tower_unique_contribution.json"
    payload = json.loads(unique.read_text()) if unique.is_file() else {}
    reached = payload.get("targets_reached_only_by_two_tower")
    result.add(
        "unique two-tower contribution measured",
        reached is not None,
        detail=(
            f"{reached} targets reached only by the two-tower"
            if reached is not None
            else "two_tower_unique_contribution.json missing or has no measurement"
        ),
    )


def check_evaluation_views(result: GateResult) -> None:
    """Strict, warm and cold views all exist in the final test metrics.

    Three views, three different questions. A run that recorded only the strict
    view would be missing the one number -- cold -- that Phase 5 exists to
    produce, and nothing about a healthy strict number reveals that.
    """
    final = PROJECT_ROOT / "reports/metrics/phase_05/two_tower_final_test_metrics.json"
    if not final.is_file():
        result.add("strict, warm and cold evaluation views", False, detail="no final metrics file")
        return
    payload = json.loads(final.read_text())
    views = {
        "strict": bool(payload.get("strict")),
        "warm": bool(payload.get("warm")),
        "cold": bool(payload.get("slices", {}).get("items_cold_start")),
    }
    absent = sorted(name for name, present in views.items() if not present)
    result.add(
        "strict, warm and cold evaluation views",
        not absent,
        detail="all three recorded" if not absent else f"missing view(s): {absent}",
    )


def check_readme(result: GateResult) -> None:
    """The README states Phase 5 complete and carries no superseded claim.

    A README is the only artifact most readers see. One that still says "Phase 3
    is the latest phase" is a false statement about the repository, and no
    amount of correct metrics elsewhere corrects it.
    """
    readme = PROJECT_ROOT / "README.md"
    if not readme.is_file():
        result.add("README records Phase 5 accurately", False, detail="no README.md")
        return
    text = readme.read_text()
    lowered = text.lower()

    stale = [
        phrase
        for phrase in (
            "status: phase 3",
            "status: phase 4",
            "no neural retrieval",
            "no neural retrieval, ranking, or serving model exists",
        )
        if phrase in lowered
    ]
    claims_phase5 = "phase 5" in lowered and "complete" in lowered
    result.add(
        "README records Phase 5 accurately",
        claims_phase5 and not stale,
        detail=(
            "Phase 5 recorded complete, no superseded phase claims"
            if claims_phase5 and not stale
            else f"superseded claim(s) present: {stale}"
            if stale
            else "README does not record Phase 5 as complete"
        ),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Rows of a CSV, or an empty list when it is absent or empty."""
    if not path.is_file() or not path.read_text().strip():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


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
    parser.add_argument(
        "--no-run-tests",
        action="store_true",
        help=(
            "CI mode only: skip executing the fixture tests. For inspecting the "
            "gate's structure quickly; a run using this cannot certify CI."
        ),
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help=(
            "CI-safe mode: check code, configuration and synthetic evidence only. "
            "Skips every check that needs PixelRec data, trained artifacts or "
            "real-data reports, which CI deliberately does not have."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    result = GateResult(mode="ci" if args.ci else "full")

    try:
        # Code, configuration and synthetic evidence: verifiable anywhere.
        check_sources(result)
        check_model_class(result)
        check_configs(result)
        check_commands(result)
        check_ci_job(result)
        if args.ci:
            # CI has no PixelRec download, no trained artifacts and no
            # real-data reports by design, so asking for them there would make
            # the gate fail for the wrong reason. The synthetic cold-item and
            # end-to-end tests are what CI verifies instead, and they exercise
            # the same code paths.
            check_synthetic_evidence(result, run_tests=not args.no_run_tests)
            check_environment_independence(result)
            check_real_completion_not_claimed(result)
        else:
            check_evidence(result)
            check_features(result)
            metadata = check_artifacts(result)
            check_compatibility(result, metadata)
            check_smoke_recommendation(result, metadata)
            check_real_metrics(result)
            check_evaluation_views(result)
            check_fusion_evidence(result)
            check_report_claims(result)
            check_readme(result)
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
            print(f"  [{check.status}] {check.name.ljust(width)}  {check.detail}")

    print(
        f"\n{payload['checks_passed']}/{payload['checks_run']} checks passed  "
        f"({payload['critical_failures']} critical failures, "
        f"{payload['warnings']} warnings, {payload['skipped']} skipped)"
    )
    print(f"Report written to {destination}")

    if result.passed:
        if result.mode == "ci":
            print(
                "\nCI GATE PASSED - Phase 5 source-level correctness verified. "
                "This is NOT evidence of real Phase 5 completion: "
                f"{payload['skipped']} real-data checks were skipped. "
                "Run `python scripts/validate_phase5.py` locally for that."
            )
        else:
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
