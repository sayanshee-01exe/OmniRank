"""Structured experiment reports.

Every number this project publishes goes through here, so a metric always
arrives with the protocol, the fit boundary, and the dataset identity that
produced it. A bare "NDCG@20 = 0.031" is not a result.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from omnirank.core.logging import get_logger
from omnirank.evaluation.bootstrap import ConfidenceInterval
from omnirank.evaluation.experiment import ExperimentResult

logger = get_logger(__name__)

REPORT_ROOT = Path("reports/metrics/phase_03")


def write_json(payload: Any, path: Path | str) -> Path:
    """Write a JSON report with sorted keys."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return target


def append_jsonl(payload: Any, path: Path | str) -> Path:
    """Append one record to a JSON-lines file.

    Used for the validation-run log: every trial is recorded as it happens, so a
    later selection cannot quietly omit a configuration that did badly.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    return target


def write_csv(rows: Sequence[dict[str, Any]], path: Path | str) -> Path:
    """Write rows as CSV, using the union of keys as the header."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("")
        return target
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def write_text(content: str, path: Path | str) -> Path:
    """Write a markdown report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content if content.endswith("\n") else content + "\n")
    return target


def comparison_table(results: Iterable[ExperimentResult]) -> list[dict[str, Any]]:
    """The primary model-comparison rows."""
    return [result.summary_row() for result in results]


def slice_table(results: Iterable[ExperimentResult]) -> list[dict[str, Any]]:
    """Flatten every model's slice metrics into one table."""
    rows: list[dict[str, Any]] = []
    for result in results:
        for item in result.slices:
            payload = item.to_dict()
            rows.append(
                {
                    "model": result.model_name,
                    "version": result.model_version,
                    "target_split": result.target_split,
                    **payload,
                }
            )
    return rows


def runtime_table(results: Iterable[ExperimentResult]) -> list[dict[str, Any]]:
    """Flatten runtime measurements into one table."""
    rows: list[dict[str, Any]] = []
    for result in results:
        for item in result.runtimes:
            rows.append(
                {
                    "model": result.model_name,
                    "version": result.model_version,
                    **item.to_dict(),
                }
            )
    return rows


def render_markdown_summary(
    results: Sequence[ExperimentResult],
    *,
    title: str,
    deltas: Sequence[ConfidenceInterval] = (),
    notes: Sequence[str] = (),
) -> str:
    """Render the human-readable phase summary."""
    lines = [f"# {title}", ""]

    if results:
        identity = results[0].dataset_identity
        lines += [
            f"Dataset: **{identity.get('dataset_name')}@{identity.get('dataset_version')}** "
            f"· split v{identity.get('split_version')} "
            f"· mapping v{identity.get('mapping_version')} "
            f"· manifest `{str(identity.get('dataset_manifest_sha256', ''))[:16]}…`",
            "",
            "Protocol: **full-catalogue**, seen items excluded, users with no "
            "recommendations scored zero.",
            "",
        ]

    lines += [
        "## Primary comparison",
        "",
        "| Model | Protocol | Recall@20 | NDCG@20 | Coverage@20 | Novelty@20 | Gini@20 "
        "| Reachable | Train s | Eval s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        row = result.summary_row()
        for view in ("strict", "warm"):
            lines.append(
                f"| {row['model']} ({row['version']}) | {view} "
                f"| {row[f'recall@20_{view}']:.5f} | {row[f'ndcg@20_{view}']:.5f} "
                f"| {row['coverage@20']:.5f} | {row['novelty@20']:.3f} "
                f"| {row['gini@20']:.4f} | {row['reachable_fraction']:.4f} "
                f"| {row['train_seconds'] or 0:.1f} | {row['evaluate_seconds'] or 0:.1f} |"
            )

    intervals = [
        (result, name, interval)
        for result in results
        for name, interval in result.intervals.items()
    ]
    if intervals:
        lines += [
            "",
            "## Confidence intervals (user-level bootstrap, strict view)",
            "",
            "| Model | Metric | Estimate | 95% CI |",
            "|---|---|---:|---|",
        ]
        for result, name, interval in intervals:
            lines.append(
                f"| {result.model_name} | {name} | {interval.point_estimate:.5f} "
                f"| [{interval.lower:.5f}, {interval.upper:.5f}] |"
            )

    if deltas:
        lines += [
            "",
            "## Paired deltas",
            "",
            "| Metric | Delta | 95% CI | Excludes zero |",
            "|---|---:|---|---|",
        ]
        for interval in deltas:
            lines.append(
                f"| {interval.metric} | {interval.point_estimate:+.5f} "
                f"| [{interval.lower:+.5f}, {interval.upper:+.5f}] "
                f"| {'yes' if interval.excludes_zero else '**no**'} |"
            )
        lines += [
            "",
            "A delta whose interval contains zero is reported as inconclusive, never as a win.",
        ]

    if notes:
        lines += ["", "## Notes", ""]
        lines += [f"- {note}" for note in notes]

    return "\n".join(lines) + "\n"


__all__ = [
    "REPORT_ROOT",
    "append_jsonl",
    "comparison_table",
    "render_markdown_summary",
    "runtime_table",
    "slice_table",
    "write_csv",
    "write_json",
    "write_text",
]
