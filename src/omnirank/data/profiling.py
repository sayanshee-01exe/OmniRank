"""Raw and processed data profiling.

Profiling exists to make the dataset's shape a recorded fact rather than
folklore. Every number here is computed from the data in front of it; nothing is
estimated, and a statistic that cannot be computed is omitted rather than filled
with a plausible value.

The raw profile runs *before* cleaning, so it describes the source as delivered -
which is the only baseline against which "we removed 3% of rows" means anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from omnirank.core.logging import get_logger

logger = get_logger(__name__)

#: Quantiles reported for every distribution. Includes the extremes because the
#: tails are where recommendation datasets misbehave.
DISTRIBUTION_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def describe_distribution(values: pd.Series) -> dict[str, Any]:
    """Summarise a numeric distribution."""
    if values.empty:
        return {"count": 0}
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0}
    quantiles = numeric.quantile(list(DISTRIBUTION_QUANTILES))
    return {
        "count": int(numeric.size),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": round(float(numeric.mean()), 6),
        "median": float(numeric.median()),
        "std": round(float(numeric.std(ddof=0)), 6),
        "quantiles": {f"p{int(q * 100)}": float(quantiles.loc[q]) for q in DISTRIBUTION_QUANTILES},
    }


def compute_sparsity(*, users: int, items: int, interactions: int) -> float:
    """Return ``1 - interactions / (users * items)``.

    Returns 0.0 for an empty catalogue rather than dividing by zero; a dataset
    with no users or items has no meaningful sparsity.
    """
    denominator = users * items
    if denominator <= 0:
        return 0.0
    return 1.0 - (interactions / denominator)


@dataclass(slots=True)
class RawProfile:
    """Profile of the source data, before any cleaning."""

    dataset_name: str
    source_files: list[dict[str, Any]] = field(default_factory=list)
    interactions: dict[str, Any] = field(default_factory=dict)
    items: dict[str, Any] = field(default_factory=dict)
    missingness: pd.DataFrame = field(default_factory=pd.DataFrame)
    user_activity: pd.DataFrame = field(default_factory=pd.DataFrame)
    item_popularity: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_coverage: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_failures: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready profile payload."""
        return {
            "dataset_name": self.dataset_name,
            "source_files": self.source_files,
            "interactions": self.interactions,
            "items": self.items,
        }


def profile_raw(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    *,
    dataset_name: str,
    source_files: list[dict[str, Any]],
) -> RawProfile:
    """Profile the canonical-but-uncleaned data.

    Args:
        interactions: Canonical interaction frame.
        items: Canonical item frame.
        dataset_name: Recorded in the profile.
        source_files: Loader-produced file descriptors.
    """
    per_user = interactions.groupby("external_user_id", observed=True).size()
    per_item = interactions.groupby("external_item_id", observed=True).size()

    users = int(per_user.size)
    item_count = int(per_item.size)
    row_count = len(interactions)

    interaction_stats: dict[str, Any] = {
        "rows": row_count,
        "unique_users": users,
        "unique_items": item_count,
        "sparsity": round(
            compute_sparsity(users=users, items=item_count, interactions=row_count), 9
        ),
        "event_types": interactions["event_type"].value_counts().to_dict(),
        "exact_duplicate_rows": int(
            interactions.duplicated(
                subset=["external_user_id", "external_item_id", "event_type", "timestamp"]
            ).sum()
        ),
        "interactions_per_user": describe_distribution(per_user),
        "interactions_per_item": describe_distribution(per_item),
    }
    if "event_timestamp_utc" in interactions.columns and row_count:
        stamps = interactions["event_timestamp_utc"].dropna()
        if not stamps.empty:
            interaction_stats["timestamp_range"] = {
                "min_epoch": int(interactions["timestamp"].min()),
                "max_epoch": int(interactions["timestamp"].max()),
                "min_utc": stamps.min().isoformat(),
                "max_utc": stamps.max().isoformat(),
                "span_days": int((stamps.max() - stamps.min()).days),
            }

    known_items = set(items["external_item_id"].astype(str)) if len(items) else set()
    interaction_stats["interactions_with_unknown_item"] = (
        int((~interactions["external_item_id"].astype(str).isin(known_items)).sum())
        if known_items
        else row_count
    )

    item_stats: dict[str, Any] = {
        "rows": len(items),
        "duplicate_item_ids": int(items["external_item_id"].duplicated().sum())
        if len(items)
        else 0,
        "items_without_interactions": (
            len(known_items - set(interactions["external_item_id"].astype(str)))
            if known_items
            else 0
        ),
    }
    if "category" in items.columns and len(items):
        item_stats["distinct_categories"] = int(items["category"].nunique(dropna=True))

    missingness = _missingness_frame({"interactions": interactions, "items": items})
    user_activity = (
        per_user.rename("interaction_count")
        .reset_index()
        .rename(columns={"index": "external_user_id"})
    )
    popularity = (
        per_item.rename("interaction_count")
        .reset_index()
        .rename(columns={"index": "external_item_id"})
    )

    coverage_rows = []
    for column, label in (
        ("title", "title"),
        ("description", "description"),
        ("category", "category"),
        ("image_reference", "image_reference"),
    ):
        if column in items.columns and len(items):
            present = int(items[column].notna().sum())
            coverage_rows.append(
                {
                    "field": label,
                    "present": present,
                    "missing": int(len(items) - present),
                    "coverage": round(present / len(items), 6),
                }
            )

    profile = RawProfile(
        dataset_name=dataset_name,
        source_files=source_files,
        interactions=interaction_stats,
        items=item_stats,
        missingness=missingness,
        user_activity=user_activity,
        item_popularity=popularity,
        feature_coverage=pd.DataFrame(coverage_rows),
    )
    logger.info(
        "profiling.raw_completed",
        rows=row_count,
        users=users,
        items=item_count,
        sparsity=interaction_stats["sparsity"],
    )
    return profile


def _missingness_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-column null counts across several frames."""
    rows: list[dict[str, Any]] = []
    for entity, frame in frames.items():
        for column in frame.columns:
            missing = int(frame[column].isna().sum())
            rows.append(
                {
                    "entity": entity,
                    "column": column,
                    "rows": len(frame),
                    "missing": missing,
                    "missing_pct": round(missing / len(frame) * 100, 6) if len(frame) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def render_raw_profile_markdown(profile: RawProfile) -> str:
    """Render the raw profile as a readable report."""
    interactions = profile.interactions
    items = profile.items
    lines = [
        f"# Raw data profile - {profile.dataset_name}",
        "",
        "Computed from the source files **before any cleaning**, so it is the "
        "baseline every later row-count claim is measured against.",
        "",
        "## Source files",
        "",
        "| File | Bytes | Rows | SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for source in profile.source_files:
        digest = source.get("sha256") or ""
        lines.append(
            f"| `{source['name']}` | {source['bytes']:,} | "
            f"{source.get('rows') or 0:,} | `{digest[:16]}…` |"
        )

    lines += [
        "",
        "## Interactions",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rows | {interactions.get('rows', 0):,} |",
        f"| Unique users | {interactions.get('unique_users', 0):,} |",
        f"| Unique items | {interactions.get('unique_items', 0):,} |",
        f"| Sparsity | {interactions.get('sparsity', 0):.9f} |",
        f"| Exact duplicate events | {interactions.get('exact_duplicate_rows', 0):,} |",
        f"| Interactions referencing an unknown item | "
        f"{interactions.get('interactions_with_unknown_item', 0):,} |",
    ]
    if "timestamp_range" in interactions:
        window = interactions["timestamp_range"]
        lines += [
            f"| Timestamp range | {window['min_utc']} → {window['max_utc']} |",
            f"| Span (days) | {window['span_days']:,} |",
        ]

    for label, key in (
        ("Interactions per user", "interactions_per_user"),
        ("Interactions per item", "interactions_per_item"),
    ):
        stats = interactions.get(key, {})
        if stats.get("count"):
            lines += [
                "",
                f"### {label}",
                "",
                "| min | p25 | median | mean | p95 | p99 | max |",
                "|---:|---:|---:|---:|---:|---:|---:|",
                f"| {stats['min']:.0f} | {stats['quantiles']['p25']:.0f} | "
                f"{stats['median']:.0f} | {stats['mean']:.2f} | "
                f"{stats['quantiles']['p95']:.0f} | {stats['quantiles']['p99']:.0f} | "
                f"{stats['max']:.0f} |",
            ]

    lines += [
        "",
        "## Items",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rows | {items.get('rows', 0):,} |",
        f"| Duplicate item ids | {items.get('duplicate_item_ids', 0):,} |",
        f"| Items with no interaction | {items.get('items_without_interactions', 0):,} |",
        f"| Distinct categories | {items.get('distinct_categories', 0):,} |",
    ]

    if not profile.feature_coverage.empty:
        lines += [
            "",
            "## Metadata field coverage",
            "",
            "| Field | Present | Missing | Coverage |",
            "|---|---:|---:|---:|",
        ]
        for row in profile.feature_coverage.to_dict(orient="records"):
            lines.append(
                f"| `{row['field']}` | {row['present']:,} | {row['missing']:,} | "
                f"{row['coverage']:.4f} |"
            )

    return "\n".join(lines) + "\n"


@dataclass(slots=True)
class ProcessedProfile:
    """Profile of the processed dataset."""

    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready payload."""
        return self.payload


def profile_processed(
    frame: pd.DataFrame,
    *,
    raw_profile: RawProfile,
    cleaning_report: dict[str, Any],
    filtering_report: dict[str, Any],
    split_statistics: dict[str, Any],
    sequence_statistics: list[dict[str, Any]],
    feature_validations: list[dict[str, Any]],
    slice_definitions: list[dict[str, Any]],
    leakage_report: dict[str, Any],
) -> ProcessedProfile:
    """Assemble the processed-dataset profile from every stage's report."""
    users = int(frame["internal_user_id"].nunique())
    items = int(frame["internal_item_id"].nunique())
    rows = len(frame)
    lengths = frame.groupby("internal_user_id", observed=True).size()

    payload = {
        "counts": {
            "raw_users": raw_profile.interactions.get("unique_users", 0),
            "processed_users": users,
            "raw_items": raw_profile.interactions.get("unique_items", 0),
            "processed_items": items,
            "raw_interactions": raw_profile.interactions.get("rows", 0),
            "processed_interactions": rows,
        },
        "removed": {
            "users": raw_profile.interactions.get("unique_users", 0) - users,
            "items": raw_profile.interactions.get("unique_items", 0) - items,
            "interactions": raw_profile.interactions.get("rows", 0) - rows,
        },
        "sparsity": round(compute_sparsity(users=users, items=items, interactions=rows), 9),
        "duplicates_removed": cleaning_report.get("rejected_by_reason", {}).get(
            "duplicate_interaction", 0
        ),
        "cleaning": cleaning_report,
        "filtering": filtering_report,
        "split": split_statistics,
        "sequences": sequence_statistics,
        "history_length_distribution": describe_distribution(lengths),
        "features": feature_validations,
        "slices": slice_definitions,
        "leakage": leakage_report,
    }
    return ProcessedProfile(payload=payload)


def render_processed_profile_markdown(profile: ProcessedProfile) -> str:
    """Render the processed profile as a readable report."""
    payload = profile.payload
    counts = payload["counts"]
    removed = payload["removed"]
    split = payload["split"]

    lines = [
        "# Processed data profile - pixelrec50k",
        "",
        "## Raw → processed",
        "",
        "| Entity | Raw | Processed | Removed |",
        "|---|---:|---:|---:|",
        f"| Users | {counts['raw_users']:,} | {counts['processed_users']:,} "
        f"| {removed['users']:,} |",
        f"| Items | {counts['raw_items']:,} | {counts['processed_items']:,} "
        f"| {removed['items']:,} |",
        f"| Interactions | {counts['raw_interactions']:,} | {counts['processed_interactions']:,} "
        f"| {removed['interactions']:,} |",
        "",
        f"Sparsity: **{payload['sparsity']:.9f}** · "
        f"Duplicates removed: **{payload['duplicates_removed']:,}**",
        "",
        "## Splits",
        "",
        "| Split | Rows | Users | Items |",
        "|---|---:|---:|---:|",
    ]
    for name in ("train", "validation", "test"):
        lines.append(
            f"| {name} | {split.get(f'{name}_rows', 0):,} | "
            f"{split.get(f'{name}_users', 0):,} | {split.get(f'{name}_items', 0):,} |"
        )
    lines += [
        "",
        f"Strategy: `{split.get('split_strategy')}` · ordering field: "
        f"`{split.get('ordering_field')}` · eligible users: "
        f"{split.get('eligible_users', 0):,} · ineligible: {split.get('ineligible_users', 0):,}",
        "",
        "## Sequential examples",
        "",
        "| Split | Examples | Users | Skipped (short history) | Truncated | Mean length |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stats in payload["sequences"]:
        lines.append(
            f"| {stats['split']} | {stats['examples']:,} | {stats['users']:,} | "
            f"{stats['skipped_short_history']:,} | {stats['truncated']:,} | "
            f"{stats['mean_length']:.2f} |"
        )

    lines += [
        "",
        "## Multimodal feature coverage",
        "",
        "| Modality | Available | Dimension | Matched | Coverage |",
        "|---|---|---:|---:|---:|",
    ]
    for validation in payload["features"]:
        lines.append(
            f"| {validation['modality']} | {'yes' if validation['available'] else '**no**'} | "
            f"{validation['dimension'] or 0} | {validation['rows_matched']:,} | "
            f"{validation['coverage']:.4f} |"
        )

    lines += ["", "## Evaluation slices", "", "| Slice | Entity | Size |", "|---|---|---:|"]
    for definition in payload["slices"]:
        lines.append(
            f"| `{definition['slice_name']}` | {definition['entity_type']} | "
            f"{definition['size']:,} |"
        )

    leakage = payload["leakage"]
    lines += [
        "",
        "## Leakage checks",
        "",
        f"**{'PASSED' if leakage.get('passed') else 'FAILED'}** — "
        f"{leakage.get('passed_checks', 0)}/{leakage.get('total_checks', 0)} checks passed, "
        f"{leakage.get('critical_failures', 0)} critical failures, "
        f"{leakage.get('warnings', 0)} warnings.",
        "",
        "| Check | Severity | Result | Detail |",
        "|---|---|---|---|",
    ]
    for check in leakage.get("checks", []):
        verdict = (
            "pass"
            if check["passed"]
            else ("**FAIL**" if check["severity"] == "critical" else "warn")
        )
        lines.append(
            f"| `{check['check_id']}` | {check['severity']} | {verdict} | {check['detail']} |"
        )

    return "\n".join(lines) + "\n"


def write_distribution_csv(frame: pd.DataFrame, path: Path | str) -> None:
    """Write a profiling table as CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)


__all__ = [
    "DISTRIBUTION_QUANTILES",
    "ProcessedProfile",
    "RawProfile",
    "compute_sparsity",
    "describe_distribution",
    "profile_processed",
    "profile_raw",
    "render_processed_profile_markdown",
    "render_raw_profile_markdown",
    "write_distribution_csv",
]
