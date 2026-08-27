"""Point-in-time candidate snapshots -- what the ranker is allowed to learn from.

A ranker trained on candidates that retrieval could not actually have produced
at prediction time learns a distribution that will never occur in serving. The
failure is silent and flattering: offline metrics improve because the candidate
pool was quietly better than the real one.

Everything here exists to make that impossible:

* Candidates come from retrievers **fitted only on pre-cutoff interactions**.
  Not "fitted on everything and queried with a truncated history" -- the
  weights themselves must not have seen the future, because a model that
  trained on an interaction has encoded it whether or not you feed it back in.
* The held-out target is **never** inserted into the pool. A group with no
  positive is a retrieval miss, and recording it as one is the only way
  candidate recall stays honest.
* Every snapshot records the identity of the retrievers, the cutoff, and the
  aggregation that produced it, so a row can be traced to the exact system
  state that generated it.

Presence indicators are carried per source because **a missing score and a real
score of zero are different states**. Popularity genuinely scores an unpopular
item near zero; it scores an item it never proposed not at all. Collapsing the
two teaches the ranker that "not retrieved" means "retrieved and bad".
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from omnirank.core.exceptions import DataError
from omnirank.core.logging import get_logger
from omnirank.models.base import ScoredCandidate

logger = get_logger(__name__)

#: Bumped when the snapshot schema or its semantics change.
SNAPSHOT_VERSION: Final = "1"

#: The five retrieval sources, in the order their columns appear. Fixed so a
#: snapshot written today aligns with one written next month.
SOURCES: Final[tuple[str, ...]] = (
    "popularity",
    "matrix_factorization",
    "lightgcn",
    "sasrec",
    "two_tower",
)

#: Column prefixes, identical to the source names. Kept as an explicit mapping
#: rather than derived, so a future rename of a source cannot silently rename
#: columns in already-written snapshots.
SOURCE_COLUMN_PREFIX: Final[dict[str, str]] = {source: source for source in SOURCES}

#: Identity columns every snapshot row carries.
IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "query_id",
    "external_user_id",
    "internal_user_id",
    "external_item_id",
    "internal_item_id",
    "target_item_id",
    # The real prediction cutoff, in Unix seconds. Distinct from `fold_id`:
    # one is a time, the other is which experiment produced the row, and a
    # time-dependent feature computed from a fold label is not a feature.
    "as_of_timestamp",
    "label",
    "fold_id",
    "split",
    "candidate_budget",
)

#: Aggregation columns produced by rank fusion.
AGGREGATE_COLUMNS: Final[tuple[str, ...]] = (
    "aggregate_rank",
    "aggregate_score",
    "candidate_source_count",
    "candidate_sources",
)


def source_columns() -> tuple[str, ...]:
    """Per-source presence, rank and score columns, in stable order."""
    columns: list[str] = []
    for source in SOURCES:
        prefix = SOURCE_COLUMN_PREFIX[source]
        columns += [f"{prefix}_present", f"{prefix}_rank", f"{prefix}_score"]
    return tuple(columns)


def snapshot_columns() -> tuple[str, ...]:
    """The full canonical snapshot schema, in order."""
    return (*IDENTITY_COLUMNS, *AGGREGATE_COLUMNS, *source_columns())


#: Cutoff policy this build applies, recorded in every manifest so a reader
#: never has to infer which of the two defensible choices was made.
CUTOFF_POLICY: Final = (
    "held-out target timestamp, exclusive: a query's features may use only "
    "interactions strictly earlier than the target being predicted"
)


@dataclass(frozen=True, slots=True)
class RetrieverIdentity:
    """What produced one source's candidates, and what it was allowed to see.

    ``fit_boundary_timestamp`` and ``max_fit_timestamp`` together are the
    machine-checkable claim: the latest interaction the model trained on is
    strictly earlier than the cutoff it is being used to predict. A snapshot
    whose sources cannot show that is not a point-in-time snapshot, whatever
    its prose says.
    """

    source: str
    model_class: str
    model_version: str
    configuration_hash: str
    seed: int | None
    fit_fold: str
    fit_boundary: str
    fit_boundary_timestamp: int | None
    max_fit_timestamp: int | None
    fit_interactions: int
    fit_users: int
    fit_items: int
    dataset_version: str
    split_version: str
    mapping_checksum: str
    candidate_budget: int
    device: str
    fit_seconds: float
    retrieval_status: str = "ok"
    failure_reason: str | None = None
    artifact_checksum: str | None = None

    @property
    def boundary_respected(self) -> bool | None:
        """Whether the latest fitted interaction precedes the cutoff.

        ``None`` when either timestamp is unknown -- which is itself reportable,
        and is never silently treated as a pass.
        """
        if self.fit_boundary_timestamp is None or self.max_fit_timestamp is None:
            return None
        return self.max_fit_timestamp < self.fit_boundary_timestamp

    def to_dict(self) -> dict[str, Any]:
        """Manifest-ready payload."""
        return {
            "source": self.source,
            "model_class": self.model_class,
            "model_version": self.model_version,
            "configuration_hash": self.configuration_hash,
            "seed": self.seed,
            "fit_fold": self.fit_fold,
            "fit_boundary": self.fit_boundary,
            "fit_boundary_timestamp": self.fit_boundary_timestamp,
            "max_fit_timestamp": self.max_fit_timestamp,
            "boundary_respected": self.boundary_respected,
            "fit_interactions": self.fit_interactions,
            "fit_users": self.fit_users,
            "fit_items": self.fit_items,
            "dataset_version": self.dataset_version,
            "split_version": self.split_version,
            "mapping_checksum": self.mapping_checksum,
            "candidate_budget": self.candidate_budget,
            "device": self.device,
            "fit_seconds": round(self.fit_seconds, 2),
            "retrieval_status": self.retrieval_status,
            "failure_reason": self.failure_reason,
            "artifact_checksum": self.artifact_checksum,
        }


@dataclass(slots=True)
class SnapshotStats:
    """What a snapshot build actually produced.

    Zero-positive groups are counted rather than dropped silently: they are
    retrieval misses, and that count *is* the candidate-recall denominator. A
    snapshot that quietly discarded them would report a recall computed over
    the queries that happened to succeed.
    """

    queries: int = 0
    rows: int = 0
    positive_queries: int = 0
    zero_positive_queries: int = 0
    candidates_per_query: list[int] = field(default_factory=list)
    cutoffs: list[int] = field(default_factory=list)
    source_contributions: dict[str, int] = field(default_factory=dict)

    @property
    def candidate_recall(self) -> float:
        """Share of queries whose held-out target reached the pool at all."""
        return self.positive_queries / self.queries if self.queries else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Manifest-ready payload."""
        pool = self.candidates_per_query
        return {
            "queries": self.queries,
            "rows": self.rows,
            "positive_queries": self.positive_queries,
            "zero_positive_queries": self.zero_positive_queries,
            "candidate_recall": round(self.candidate_recall, 8),
            "mean_candidates_per_query": round(float(np.mean(pool)), 2) if pool else 0.0,
            "median_candidates_per_query": float(np.median(pool)) if pool else 0.0,
            "min_candidates_per_query": int(min(pool)) if pool else 0,
            "max_candidates_per_query": int(max(pool)) if pool else 0,
            "min_cutoff_timestamp": int(min(self.cutoffs)) if self.cutoffs else None,
            "max_cutoff_timestamp": int(max(self.cutoffs)) if self.cutoffs else None,
            "source_contributions": dict(sorted(self.source_contributions.items())),
        }


def build_snapshot_rows(
    *,
    query_id: str,
    external_user_id: str,
    internal_user_id: int,
    target_external_item: str,
    target_internal_item: int,
    as_of_timestamp: int,
    fold_id: str,
    split: str,
    candidate_budget: int,
    per_source: Mapping[str, Sequence[ScoredCandidate]],
    fused: Sequence[tuple[str, float]],
    external_to_internal_item: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Turn one query's retrieved lists into canonical snapshot rows.

    ``per_source`` maps a source name to the :class:`ScoredCandidate` list that
    source actually returned -- carrying the model's own score, not a value
    reconstructed from the position. ``fused`` is the RRF order, whose score is
    a *separate* aggregate feature and never overwrites a source score.

    The target is labelled if and only if it is genuinely present. There is no
    code path in this function that inserts it.

    Raises:
        DataError: A source returned an unmapped item, or the cutoff is not a
            usable timestamp.
    """
    if not isinstance(as_of_timestamp, (int, np.integer)) or as_of_timestamp <= 0:
        raise DataError(
            "as_of_timestamp must be a positive Unix timestamp, not a fold label",
            value=repr(as_of_timestamp),
            query_id=query_id,
        )

    ranks: dict[str, dict[str, int]] = {}
    scores: dict[str, dict[str, float | None]] = {}
    for source, ranked in per_source.items():
        ranks[source] = {candidate.item_id: candidate.rank for candidate in ranked}
        scores[source] = {candidate.item_id: candidate.score for candidate in ranked}

    rows: list[dict[str, Any]] = []
    for position, (item_id, aggregate_score) in enumerate(fused, start=1):
        internal_item = external_to_internal_item.get(item_id)
        if internal_item is None:
            raise DataError(
                "A retriever returned an item absent from the id mapping",
                item_id=item_id,
                query_id=query_id,
            )
        contributing = [source for source in SOURCES if item_id in ranks.get(source, {})]
        row: dict[str, Any] = {
            "query_id": query_id,
            "external_user_id": external_user_id,
            "internal_user_id": int(internal_user_id),
            "external_item_id": item_id,
            "internal_item_id": int(internal_item),
            "target_item_id": target_external_item,
            "as_of_timestamp": int(as_of_timestamp),
            # The only place a label is assigned, and it is an equality test.
            "label": int(item_id == target_external_item),
            "fold_id": fold_id,
            "split": split,
            "candidate_budget": int(candidate_budget),
            # RRF output. A separate quantity from any source score, and never
            # written into a source column.
            "aggregate_rank": position,
            "aggregate_score": float(aggregate_score),
            "candidate_source_count": len(contributing),
            "candidate_sources": "|".join(contributing),
        }
        for source in SOURCES:
            prefix = SOURCE_COLUMN_PREFIX[source]
            present = item_id in ranks.get(source, {})
            row[f"{prefix}_present"] = int(present)
            # NaN, not 0.0, when a source did not propose the item -- and NaN
            # again when it proposed the item without a score. A zero in either
            # case would be indistinguishable from a genuine zero score, which
            # is a real value for popularity and for a dot product.
            row[f"{prefix}_rank"] = float(ranks[source][item_id]) if present else np.nan
            raw = scores[source].get(item_id) if present else None
            row[f"{prefix}_score"] = float(raw) if raw is not None else np.nan
        rows.append(row)
    return rows


def validate_snapshot(frame: pd.DataFrame, *, expect_single_positive: bool = True) -> None:
    """Assert the structural invariants LightGBM's grouping depends on.

    Every failure here is one that produces a *trained model* rather than an
    error: non-contiguous groups silently mis-assign rows to queries, and a
    duplicated candidate double-counts a positive.

    Raises:
        DataError: Any invariant is violated.
    """
    if frame.empty:
        raise DataError("Candidate snapshot is empty")

    missing = [column for column in snapshot_columns() if column not in frame.columns]
    if missing:
        raise DataError("Snapshot is missing canonical columns", missing=missing)

    # Query rows must be contiguous: LightGBM's `group` argument is a list of
    # sizes, not ids, so a scattered query is silently mis-grouped.
    query_ids = frame["query_id"].to_numpy()
    boundaries = np.flatnonzero(query_ids[1:] != query_ids[:-1]) + 1
    blocks = np.split(query_ids, boundaries)
    seen: set[str] = set()
    for block in blocks:
        name = str(block[0])
        if name in seen:
            raise DataError("Query rows are not contiguous", query_id=name)
        seen.add(name)

    duplicated = frame.duplicated(subset=["query_id", "external_item_id"]).sum()
    if duplicated:
        raise DataError("Snapshot contains duplicate candidates", duplicates=int(duplicated))

    positives = frame.groupby("query_id", sort=False)["label"].sum()
    if expect_single_positive and (positives > 1).any():
        offenders = positives[positives > 1]
        raise DataError(
            "A query has more than one positive under leave-one-out",
            queries=offenders.index.tolist()[:5],
        )

    if not frame["label"].isin((0, 1)).all():
        raise DataError("Labels must be binary implicit relevance")

    logger.info(
        "ranking.snapshot_validated",
        rows=len(frame),
        queries=int(frame["query_id"].nunique()),
        positive_queries=int((positives > 0).sum()),
        zero_positive_queries=int((positives == 0).sum()),
    )


def query_groups(frame: pd.DataFrame) -> np.ndarray:
    """Group sizes for LightGBM, in row order.

    Raises:
        DataError: The sizes do not sum to the frame length, which would make
            every group after the first describe the wrong rows.
    """
    query_ids = frame["query_id"].to_numpy()
    boundaries = np.flatnonzero(query_ids[1:] != query_ids[:-1]) + 1
    sizes = np.diff([0, *boundaries.tolist(), len(query_ids)])
    if int(sizes.sum()) != len(frame):
        raise DataError(
            "Query group sizes do not cover the dataset",
            total=int(sizes.sum()),
            rows=len(frame),
        )
    return sizes.astype("int64")


def drop_zero_positive_queries(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove groups with no positive, returning the frame and the count.

    Such a group carries no within-group ranking signal -- every item is a
    negative, so no pairwise comparison exists to learn from. They are dropped
    for *training* only, and the count is returned so end-to-end evaluation can
    add them back as the failures they are.
    """
    positives = frame.groupby("query_id", sort=False)["label"].transform("sum")
    keep = positives > 0
    dropped = int(frame.loc[~keep, "query_id"].nunique())
    return frame.loc[keep].reset_index(drop=True), dropped


def snapshot_checksum(frame: pd.DataFrame) -> str:
    """Content hash over the identity and label columns.

    Deliberately excludes float score columns: they carry platform-dependent
    last-bit noise, and a checksum that changed between machines for that
    reason would be ignored within a week.
    """
    subset = frame[["query_id", "external_item_id", "label", "aggregate_rank"]]
    payload = subset.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_manifest(
    *,
    fold_id: str,
    split: str,
    stats: SnapshotStats,
    retrievers: Sequence[RetrieverIdentity],
    dataset_identity: Mapping[str, Any],
    mapping_checksum: str,
    candidate_budget: int,
    aggregation: Mapping[str, Any],
    checksum: str,
    degraded: bool,
    degraded_sources: Sequence[str] = (),
    wall_seconds: float = 0.0,
    peak_memory_mb: float = 0.0,
) -> dict[str, Any]:
    """The provenance record a snapshot is worthless without.

    ``degraded`` is carried at the top level rather than inferred from the
    source list, because every downstream consumer must be able to refuse a
    degraded snapshot with a single check rather than a scan.
    """
    from datetime import UTC, datetime

    from omnirank.artifacts.metadata import detect_git_commit

    boundaries = {identity.fit_boundary for identity in retrievers}
    respected = [identity.boundary_respected for identity in retrievers]
    if len(boundaries) > 1:
        logger.error(
            "ranking.snapshot_boundary_disagreement",
            fold_id=fold_id,
            boundaries=sorted(boundaries),
        )

    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "fold_id": fold_id,
        "split": split,
        "cutoff_policy": CUTOFF_POLICY,
        "candidate_budget": candidate_budget,
        "aggregation": dict(aggregation),
        "sources": [identity.to_dict() for identity in retrievers],
        "source_count": len(retrievers),
        "fit_boundary": sorted(boundaries),
        "boundaries_agree": len(boundaries) == 1,
        # False only when a source *proves* it trained past the cutoff; None
        # entries mean unknown and are reported rather than assumed safe.
        "all_boundaries_respected": all(value is True for value in respected),
        "boundary_unknown_sources": [
            identity.source for identity in retrievers if identity.boundary_respected is None
        ],
        "degraded": degraded,
        "degraded_sources": list(degraded_sources),
        "dataset_identity": dict(dataset_identity),
        "mapping_checksum": mapping_checksum,
        "snapshot_checksum": checksum,
        "statistics": stats.to_dict(),
        "wall_seconds": round(wall_seconds, 1),
        "peak_memory_mb": round(peak_memory_mb, 1),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "git_commit": detect_git_commit(),
        "schema": list(snapshot_columns()),
    }


def write_manifest(manifest: dict[str, Any], path: Any) -> None:
    """Write a snapshot manifest as sorted JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")


def guard_overwrite(
    destination: Path, manifest_path: Path, *, overwrite: bool, fold_id: str, split: str
) -> None:
    """Refuse to replace an existing snapshot unless explicitly told to.

    A snapshot costs hours to produce. Silently replacing one -- especially with
    a *different* fold, which is the mistake a repeated command line actually
    makes -- destroys the only copy of an expensive measurement and leaves no
    trace that it happened.

    Raises:
        DataError: The destination exists and ``overwrite`` was not given, or
            the existing manifest describes a different fold or split.
    """
    if not destination.exists() and not manifest_path.exists():
        return

    existing: dict[str, Any] = {}
    if manifest_path.is_file():
        with contextlib.suppress(json.JSONDecodeError):
            existing = json.loads(manifest_path.read_text())

    identity = {
        "fold_id": existing.get("fold_id"),
        "split": existing.get("split"),
        "snapshot_checksum": str(existing.get("snapshot_checksum", ""))[:16],
        "created_at": existing.get("created_at"),
        "queries": existing.get("statistics", {}).get("queries"),
    }
    if not overwrite:
        raise DataError(
            "Snapshot already exists. Pass --overwrite to replace it.",
            destination=str(destination),
            existing=identity,
        )

    # Overwriting a *different* fold is almost always a mistyped command rather
    # than an intention, so it is refused even with the flag.
    if existing and (existing.get("fold_id") != fold_id or existing.get("split") != split):
        raise DataError(
            "Refusing to overwrite a snapshot describing a different fold or split",
            existing=identity,
            requested={"fold_id": fold_id, "split": split},
        )
    logger.warning(
        "ranking.snapshot_overwrite",
        destination=str(destination),
        replacing=identity,
        with_fold=fold_id,
        with_split=split,
    )


def write_snapshot_atomically(frame: pd.DataFrame, destination: Path) -> None:
    """Write Parquet through a temporary path, then rename into place.

    A crash midway through a direct write leaves a truncated Parquet that reads
    as a valid but shorter dataset. The rename is atomic on the same filesystem,
    so a reader sees either the old snapshot or the complete new one.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def require_official_snapshot(manifest: Mapping[str, Any], *, purpose: str) -> None:
    """Refuse a degraded snapshot for selection or final evaluation.

    A degraded snapshot is missing one or more retrievers. Its candidate pool is
    not the pool serving would produce, so a ranker selected on it is selected
    against the wrong distribution -- and a final metric computed from it is not
    the system's metric.

    Raises:
        DataError: The snapshot is degraded, its boundaries do not hold, or it
            does not record all five sources.
    """
    if manifest.get("degraded"):
        raise DataError(
            f"Refusing to use a degraded snapshot for {purpose}",
            degraded_sources=list(manifest.get("degraded_sources", [])),
            fold_id=manifest.get("fold_id"),
        )
    if manifest.get("all_boundaries_respected") is False:
        raise DataError(
            f"Refusing to use a snapshot whose sources trained past the cutoff for {purpose}",
            fold_id=manifest.get("fold_id"),
        )
    expected = set(SOURCES)
    present = {str(source["source"]) for source in manifest.get("sources", [])}
    if present != expected:
        raise DataError(
            f"Snapshot does not record all five sources; cannot use for {purpose}",
            missing=sorted(expected - present),
            unexpected=sorted(present - expected),
        )


__all__ = [
    "AGGREGATE_COLUMNS",
    "CUTOFF_POLICY",
    "IDENTITY_COLUMNS",
    "SNAPSHOT_VERSION",
    "SOURCES",
    "SOURCE_COLUMN_PREFIX",
    "RetrieverIdentity",
    "SnapshotStats",
    "build_manifest",
    "build_snapshot_rows",
    "drop_zero_positive_queries",
    "guard_overwrite",
    "query_groups",
    "require_official_snapshot",
    "snapshot_checksum",
    "snapshot_columns",
    "source_columns",
    "validate_snapshot",
    "write_manifest",
    "write_snapshot_atomically",
]
