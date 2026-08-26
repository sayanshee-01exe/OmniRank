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

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd

from omnirank.core.exceptions import DataError
from omnirank.core.logging import get_logger

logger = get_logger(__name__)

#: The five retrieval sources, in the order their columns appear. Fixed so a
#: snapshot written today aligns with one written next month.
SOURCES: Final[tuple[str, ...]] = (
    "popularity",
    "matrix_factorization",
    "lightgcn",
    "sasrec",
    "two_tower",
)

#: Short column prefixes. `matrix_factorization` is abbreviated to `bpr` in
#: column names because the algorithm is what a reader of the schema
#: recognises, and the mapping is recorded here rather than inferred.
SOURCE_COLUMN_PREFIX: Final[dict[str, str]] = {
    "popularity": "popularity",
    "matrix_factorization": "bpr",
    "lightgcn": "lightgcn",
    "sasrec": "sasrec",
    "two_tower": "two_tower",
}

#: Identity columns every snapshot row carries.
IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "query_id",
    "external_user_id",
    "internal_user_id",
    "external_item_id",
    "internal_item_id",
    "target_item_id",
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


@dataclass(frozen=True, slots=True)
class RetrieverIdentity:
    """What produced one source's candidates, and what it was allowed to see.

    ``fit_boundary`` is the whole point: it states, in words a reader can
    check, which interactions the weights were fitted on. A snapshot whose
    retrievers disagree about their boundary is not a point-in-time snapshot.
    """

    source: str
    model_version: str
    configuration_hash: str
    mapping_checksum: str
    fit_boundary: str
    fit_interactions: int

    def to_dict(self) -> dict[str, Any]:
        """Manifest-ready payload."""
        return {
            "source": self.source,
            "model_version": self.model_version,
            "configuration_hash": self.configuration_hash,
            "mapping_checksum": self.mapping_checksum,
            "fit_boundary": self.fit_boundary,
            "fit_interactions": self.fit_interactions,
        }


@dataclass(slots=True)
class SnapshotStats:
    """What a snapshot build actually produced.

    Zero-positive groups are counted rather than dropped silently: they are
    retrieval misses, and the count *is* the candidate-recall denominator.
    """

    queries: int = 0
    rows: int = 0
    positive_queries: int = 0
    zero_positive_queries: int = 0
    candidates_per_query: list[int] = field(default_factory=list)

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
            "mean_pool_size": round(float(np.mean(pool)), 2) if pool else 0.0,
            "min_pool_size": int(min(pool)) if pool else 0,
            "max_pool_size": int(max(pool)) if pool else 0,
        }


def build_snapshot_rows(
    *,
    query_id: str,
    external_user_id: str,
    internal_user_id: int,
    target_external_item: str,
    target_internal_item: int,
    as_of_timestamp: Any,
    fold_id: str,
    split: str,
    candidate_budget: int,
    per_source: dict[str, list[tuple[str, float]]],
    fused: list[tuple[str, float]],
    external_to_internal_item: dict[str, int],
) -> list[dict[str, Any]]:
    """Turn one query's retrieved lists into canonical snapshot rows.

    ``per_source`` maps a source name to its ranked ``(external_item_id,
    score)`` list. ``fused`` is the aggregated order. The target is labelled if
    and only if it is genuinely present -- this function has no code path that
    adds it.

    Raises:
        DataError: A source returned an item the mapping does not know, which
            would produce a row nothing downstream can resolve.
    """
    ranks: dict[str, dict[str, int]] = {}
    scores: dict[str, dict[str, float]] = {}
    for source, ranked in per_source.items():
        ranks[source] = {item: position for position, (item, _) in enumerate(ranked, start=1)}
        scores[source] = dict(ranked)

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
            "as_of_timestamp": as_of_timestamp,
            # The only place a label is assigned, and it is an equality test.
            "label": int(item_id == target_external_item),
            "fold_id": fold_id,
            "split": split,
            "candidate_budget": int(candidate_budget),
            "aggregate_rank": position,
            "aggregate_score": float(aggregate_score),
            "candidate_source_count": len(contributing),
            "candidate_sources": "|".join(contributing),
        }
        for source in SOURCES:
            prefix = SOURCE_COLUMN_PREFIX[source]
            present = item_id in ranks.get(source, {})
            row[f"{prefix}_present"] = int(present)
            # NaN, not 0.0, when a source did not propose the item. A zero here
            # would be indistinguishable from a genuine zero score.
            row[f"{prefix}_rank"] = float(ranks[source][item_id]) if present else np.nan
            row[f"{prefix}_score"] = float(scores[source][item_id]) if present else np.nan
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
    retrievers: list[RetrieverIdentity],
    dataset_identity: dict[str, Any],
    mapping_checksum: str,
    candidate_budget: int,
    aggregation: dict[str, Any],
    checksum: str,
) -> dict[str, Any]:
    """The provenance record a snapshot is worthless without."""
    from datetime import UTC, datetime

    from omnirank.artifacts.metadata import detect_git_commit

    boundaries = {identity.fit_boundary for identity in retrievers}
    if len(boundaries) > 1:
        # Not fatal to write, but it means the sources disagree about what
        # "before the cutoff" meant, and the snapshot is not point-in-time.
        logger.error(
            "ranking.snapshot_boundary_disagreement",
            fold_id=fold_id,
            boundaries=sorted(boundaries),
        )

    return {
        "fold_id": fold_id,
        "split": split,
        "candidate_budget": candidate_budget,
        "aggregation": aggregation,
        "sources": [identity.to_dict() for identity in retrievers],
        "fit_boundary": sorted(boundaries),
        "boundaries_agree": len(boundaries) == 1,
        "dataset_identity": dataset_identity,
        "mapping_checksum": mapping_checksum,
        "snapshot_checksum": checksum,
        "statistics": stats.to_dict(),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "git_commit": detect_git_commit(),
        "schema": list(snapshot_columns()),
    }


def write_manifest(manifest: dict[str, Any], path: Any) -> None:
    """Write a snapshot manifest as sorted JSON."""
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")


__all__ = [
    "AGGREGATE_COLUMNS",
    "IDENTITY_COLUMNS",
    "SOURCES",
    "SOURCE_COLUMN_PREFIX",
    "RetrieverIdentity",
    "SnapshotStats",
    "build_manifest",
    "build_snapshot_rows",
    "drop_zero_positive_queries",
    "query_groups",
    "snapshot_checksum",
    "snapshot_columns",
    "source_columns",
    "validate_snapshot",
    "write_manifest",
]
