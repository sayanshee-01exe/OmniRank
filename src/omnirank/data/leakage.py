"""Leakage validation.

Leakage is the failure mode that offline metrics cannot detect, because it makes
them *better*. Every check here therefore runs as part of the pipeline and a
critical failure aborts the build, rather than being an optional audit somebody
runs after the numbers look surprising.

Checks are graded:

* **critical** - the dataset is unusable; the pipeline exits non-zero.
* **warning** - a property worth knowing that is not automatically wrong (for
  example, items appearing in test but never in training: real cold start, not
  a bug).

Each check returns evidence, not just a verdict, so a failure names the specific
rows to look at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd

from omnirank.core.exceptions import DataError
from omnirank.core.logging import get_logger
from omnirank.data.splitters import TEST, TRAIN, VALIDATION

logger = get_logger(__name__)


class Severity(StrEnum):
    """How a failed check affects the build."""

    CRITICAL = "critical"
    WARNING = "warning"


@dataclass(slots=True)
class LeakageCheck:
    """Outcome of one leakage check."""

    check_id: str
    description: str
    severity: Severity
    passed: bool
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Report-ready description."""
        return {
            "check_id": self.check_id,
            "description": self.description,
            "severity": self.severity.value,
            "passed": self.passed,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class LeakageReport:
    """All leakage checks for one build."""

    checks: list[LeakageCheck] = field(default_factory=list)

    def add(self, check: LeakageCheck) -> None:
        """Record a check, logging failures at a level matching their severity."""
        self.checks.append(check)
        if check.passed:
            logger.debug("leakage.check_passed", check_id=check.check_id)
        elif check.severity is Severity.CRITICAL:
            logger.error("leakage.check_failed", **check.to_dict())
        else:
            logger.warning("leakage.check_warning", **check.to_dict())

    @property
    def critical_failures(self) -> list[LeakageCheck]:
        """Failed critical checks."""
        return [
            check
            for check in self.checks
            if not check.passed and check.severity is Severity.CRITICAL
        ]

    @property
    def warnings(self) -> list[LeakageCheck]:
        """Failed warning-level checks."""
        return [
            check
            for check in self.checks
            if not check.passed and check.severity is Severity.WARNING
        ]

    @property
    def passed(self) -> bool:
        """True when no critical check failed."""
        return not self.critical_failures

    def raise_if_failed(self) -> None:
        """Abort the build when any critical check failed.

        Raises:
            DataError: One or more critical leakage checks failed.
        """
        if self.critical_failures:
            raise DataError(
                f"{len(self.critical_failures)} critical leakage check(s) failed. "
                "The processed dataset would produce optimistic, unreproducible "
                "offline metrics and must not be used.",
                failed_checks=[check.check_id for check in self.critical_failures],
            )

    def to_dict(self) -> dict[str, Any]:
        """Full report payload."""
        return {
            "passed": self.passed,
            "total_checks": len(self.checks),
            "passed_checks": sum(1 for check in self.checks if check.passed),
            "critical_failures": len(self.critical_failures),
            "warnings": len(self.warnings),
            "checks": [check.to_dict() for check in self.checks],
        }


def _boundary(frame: pd.DataFrame, split: str, aggregation: str) -> pd.Series:
    """Per-user min or max interaction_order within one split."""
    subset = frame[frame["split"] == split]
    if subset.empty:
        return pd.Series(dtype="float64")
    return subset.groupby("external_user_id", observed=True)["interaction_order"].agg(aggregation)


def check_no_duplicate_across_splits(frame: pd.DataFrame) -> LeakageCheck:
    """L01: no single interaction may appear in more than one split."""
    counts = frame.groupby("interaction_id", observed=True)["split"].nunique()
    offenders = counts[counts > 1]
    return LeakageCheck(
        check_id="L01_no_duplicate_interaction_across_splits",
        description="No interaction appears in more than one split.",
        severity=Severity.CRITICAL,
        passed=offenders.empty,
        detail=f"{len(offenders)} interactions span multiple splits",
        evidence={"examples": offenders.index[:5].tolist()},
    )


def check_train_precedes_validation(frame: pd.DataFrame) -> LeakageCheck:
    """L02: every training event precedes that user's validation targets."""
    train_max = _boundary(frame, TRAIN, "max")
    validation_min = _boundary(frame, VALIDATION, "min")
    shared = train_max.index.intersection(validation_min.index)
    offenders = shared[train_max[shared] >= validation_min[shared]]
    return LeakageCheck(
        check_id="L02_train_precedes_validation",
        description="Every training interaction precedes the user's validation targets.",
        severity=Severity.CRITICAL,
        passed=len(offenders) == 0,
        detail=f"{len(offenders)} users have training events at or after a validation target",
        evidence={"examples": offenders[:5].tolist()},
    )


def check_validation_precedes_test(frame: pd.DataFrame) -> LeakageCheck:
    """L03: every validation event precedes that user's test targets."""
    validation_max = _boundary(frame, VALIDATION, "max")
    test_min = _boundary(frame, TEST, "min")
    shared = validation_max.index.intersection(test_min.index)
    offenders = shared[validation_max[shared] >= test_min[shared]]
    return LeakageCheck(
        check_id="L03_validation_precedes_test",
        description="Every validation interaction precedes the user's test targets.",
        severity=Severity.CRITICAL,
        passed=len(offenders) == 0,
        detail=f"{len(offenders)} users have validation events at or after a test target",
        evidence={"examples": offenders[:5].tolist()},
    )


def check_train_precedes_test(frame: pd.DataFrame) -> LeakageCheck:
    """L04: every training event precedes that user's test targets."""
    train_max = _boundary(frame, TRAIN, "max")
    test_min = _boundary(frame, TEST, "min")
    shared = train_max.index.intersection(test_min.index)
    offenders = shared[train_max[shared] >= test_min[shared]]
    return LeakageCheck(
        check_id="L04_train_precedes_test",
        description="Every training interaction precedes the user's test targets.",
        severity=Severity.CRITICAL,
        passed=len(offenders) == 0,
        detail=f"{len(offenders)} users have training events at or after a test target",
        evidence={"examples": offenders[:5].tolist()},
    )


def check_sequences_contain_only_past(sequences: pd.DataFrame, *, split_name: str) -> LeakageCheck:
    """L05/L06: every sequence position precedes its own target.

    Covers both "the history is entirely in the past" and "the target is not
    inside the input", which are the two ways a sequential example can be built
    wrong. Together they are the reason a SASRec-style model can report
    implausible accuracy.
    """
    if sequences.empty:
        return LeakageCheck(
            check_id=f"L05_{split_name}_sequence_history_is_past",
            description=f"{split_name} sequence histories contain only pre-target events.",
            severity=Severity.CRITICAL,
            passed=True,
            detail="no sequences to check",
        )
    max_history_order = sequences["interaction_order_sequence"].apply(
        lambda orders: max(orders) if len(orders) else -1
    )
    not_past = max_history_order >= sequences["target_order"]
    target_inside = [
        target in list(items)
        for target, items in zip(sequences["target_item"], sequences["item_sequence"], strict=True)
    ]
    offenders = int(not_past.sum()) + int(sum(target_inside))
    return LeakageCheck(
        check_id=f"L05_{split_name}_sequence_history_is_past",
        description=(
            f"{split_name} sequence histories contain only events strictly before the "
            "target, and never the target itself."
        ),
        severity=Severity.CRITICAL,
        passed=offenders == 0,
        detail=(
            f"{int(not_past.sum())} sequences include a non-past event; "
            f"{int(sum(target_inside))} include the target in the input"
        ),
        evidence={
            "users_with_future_history": sequences.loc[not_past, "internal_user_id"][:5].tolist(),
        },
    )


def check_graph_is_training_only(edges: pd.DataFrame, frame: pd.DataFrame) -> LeakageCheck:
    """L07: the training graph contains no validation or test interaction."""
    held_out = frame[frame["split"] != TRAIN]
    if edges.empty or held_out.empty:
        return LeakageCheck(
            check_id="L07_graph_training_only",
            description="Training graph edges come from training interactions only.",
            severity=Severity.CRITICAL,
            passed=True,
            detail="no edges or no held-out rows to compare",
        )
    held_pairs = set(zip(held_out["internal_user_id"], held_out["internal_item_id"], strict=True))
    edge_pairs = set(zip(edges["internal_user_id"], edges["internal_item_id"], strict=True))
    # An edge is only leakage if that (user, item) pair exists *solely* in a
    # held-out split: a user who interacted with an item in training and again in
    # test legitimately has a training edge for it.
    train_pairs = set(
        zip(
            frame.loc[frame["split"] == TRAIN, "internal_user_id"],
            frame.loc[frame["split"] == TRAIN, "internal_item_id"],
            strict=True,
        )
    )
    offenders = (edge_pairs & held_pairs) - train_pairs
    return LeakageCheck(
        check_id="L07_graph_training_only",
        description="Training graph edges come from training interactions only.",
        severity=Severity.CRITICAL,
        passed=not offenders,
        detail=f"{len(offenders)} edges exist only in a held-out split",
        evidence={"examples": [list(pair) for pair in list(offenders)[:5]]},
    )


def check_popularity_is_training_only(
    popularity: pd.DataFrame, frame: pd.DataFrame
) -> LeakageCheck:
    """L08: item popularity counts are computed from training rows only.

    Recomputes the statistic independently from the training split and compares.
    Trusting the producing code to have filtered correctly is exactly the
    assumption this check exists to test.
    """
    if popularity.empty:
        return LeakageCheck(
            check_id="L08_popularity_training_only",
            description="Item popularity is derived from training interactions only.",
            severity=Severity.CRITICAL,
            passed=True,
            detail="no popularity rows",
        )
    expected = (
        frame[frame["split"] == TRAIN]
        .groupby("internal_item_id", observed=True)
        .size()
        .rename("expected")
    )
    merged = popularity.set_index("internal_item_id").join(expected, how="left")
    merged["expected"] = merged["expected"].fillna(0).astype("int64")
    mismatched = merged[merged["training_interaction_count"] != merged["expected"]]
    return LeakageCheck(
        check_id="L08_popularity_training_only",
        description="Item popularity counts equal a recount over training rows alone.",
        severity=Severity.CRITICAL,
        passed=mismatched.empty,
        detail=f"{len(mismatched)} items have counts that do not match a training-only recount",
        evidence={"examples": mismatched.index[:5].tolist()},
    )


def check_user_statistics_are_training_only(
    statistics: pd.DataFrame, frame: pd.DataFrame
) -> LeakageCheck:
    """L09: user activity statistics are computed from training rows only."""
    if statistics.empty:
        return LeakageCheck(
            check_id="L09_user_statistics_training_only",
            description="User statistics are derived from training interactions only.",
            severity=Severity.CRITICAL,
            passed=True,
            detail="no user statistic rows",
        )
    expected = (
        frame[frame["split"] == TRAIN]
        .groupby("internal_user_id", observed=True)
        .size()
        .rename("expected")
    )
    merged = statistics.set_index("internal_user_id").join(expected, how="left")
    merged["expected"] = merged["expected"].fillna(0).astype("int64")
    mismatched = merged[merged["training_interaction_count"] != merged["expected"]]
    return LeakageCheck(
        check_id="L09_user_statistics_training_only",
        description="User interaction counts equal a recount over training rows alone.",
        severity=Severity.CRITICAL,
        passed=mismatched.empty,
        detail=f"{len(mismatched)} users have counts that do not match a training-only recount",
        evidence={"examples": mismatched.index[:5].tolist()},
    )


def check_mapping_covers_all_splits(frame: pd.DataFrame) -> LeakageCheck:
    """L10: every split's ids resolve through the same mapping.

    A mapping fitted per split, or fitted on training only, leaves held-out rows
    unmappable - which surfaces later as mysteriously missing evaluation users.
    """
    unmapped_users = int(frame["internal_user_id"].isna().sum())
    unmapped_items = int(frame["internal_item_id"].isna().sum())
    negative = int(((frame["internal_user_id"] < 0) | (frame["internal_item_id"] < 0)).sum())
    total = unmapped_users + unmapped_items + negative
    return LeakageCheck(
        check_id="L10_mapping_consistent_across_splits",
        description="One mapping resolves every id in every split.",
        severity=Severity.CRITICAL,
        passed=total == 0,
        detail=(
            f"{unmapped_users} unmapped users, {unmapped_items} unmapped items, "
            f"{negative} sentinel ids"
        ),
        evidence={},
    )


def check_no_future_items_in_training(frame: pd.DataFrame) -> LeakageCheck:
    """L11: report items that appear only in held-out splits.

    A **warning**, not a failure: an item first seen at test time is genuine
    new-item cold start, and PixelRec50K produces it naturally. It is reported
    because it bounds what any collaborative model can possibly achieve.
    """
    train_items = set(frame.loc[frame["split"] == TRAIN, "internal_item_id"].unique())
    held_items = set(frame.loc[frame["split"] != TRAIN, "internal_item_id"].unique())
    cold = held_items - train_items
    return LeakageCheck(
        check_id="L11_cold_items_in_held_out",
        description="Items appearing in held-out splits but never in training.",
        severity=Severity.WARNING,
        passed=not cold,
        detail=f"{len(cold)} items are evaluated but never seen in training (genuine cold start)",
        evidence={"cold_item_count": len(cold), "examples": sorted(cold)[:5]},
    )


def check_split_labels_absent_from_features(
    feature_frames: dict[str, pd.DataFrame],
) -> LeakageCheck:
    """L12: no feature table carries a split label or a target column.

    Cheap and worth doing: a ``split`` column that survives into a feature table
    is a perfect predictor of the label, and it is an easy accident when tables
    are built by joining.
    """
    forbidden = {"split", "target_item", "target_order", "label", "relevance"}
    offenders = {
        name: sorted(forbidden.intersection(frame.columns))
        for name, frame in feature_frames.items()
        if forbidden.intersection(frame.columns)
    }
    return LeakageCheck(
        check_id="L12_no_labels_in_feature_tables",
        description="Feature tables carry no split label or target column.",
        severity=Severity.CRITICAL,
        passed=not offenders,
        detail=f"{len(offenders)} feature tables contain forbidden columns",
        evidence={"offenders": offenders},
    )


def run_all_checks(
    frame: pd.DataFrame,
    *,
    sequences: dict[str, pd.DataFrame] | None = None,
    graph_edges: pd.DataFrame | None = None,
    item_popularity: pd.DataFrame | None = None,
    user_statistics: pd.DataFrame | None = None,
    feature_frames: dict[str, pd.DataFrame] | None = None,
) -> LeakageReport:
    """Run every applicable leakage check and return the collected report."""
    report = LeakageReport()
    report.add(check_no_duplicate_across_splits(frame))
    report.add(check_train_precedes_validation(frame))
    report.add(check_validation_precedes_test(frame))
    report.add(check_train_precedes_test(frame))
    report.add(check_mapping_covers_all_splits(frame))
    report.add(check_no_future_items_in_training(frame))

    for split_name, sequence_frame in (sequences or {}).items():
        report.add(check_sequences_contain_only_past(sequence_frame, split_name=split_name))
    if graph_edges is not None:
        report.add(check_graph_is_training_only(graph_edges, frame))
    if item_popularity is not None:
        report.add(check_popularity_is_training_only(item_popularity, frame))
    if user_statistics is not None:
        report.add(check_user_statistics_are_training_only(user_statistics, frame))
    if feature_frames is not None:
        report.add(check_split_labels_absent_from_features(feature_frames))

    logger.info(
        "leakage.report",
        passed=report.passed,
        total=len(report.checks),
        critical_failures=len(report.critical_failures),
        warnings=len(report.warnings),
    )
    return report


__all__ = [
    "LeakageCheck",
    "LeakageReport",
    "Severity",
    "check_graph_is_training_only",
    "check_mapping_covers_all_splits",
    "check_no_duplicate_across_splits",
    "check_no_future_items_in_training",
    "check_popularity_is_training_only",
    "check_sequences_contain_only_past",
    "check_split_labels_absent_from_features",
    "check_train_precedes_test",
    "check_train_precedes_validation",
    "check_user_statistics_are_training_only",
    "check_validation_precedes_test",
]
