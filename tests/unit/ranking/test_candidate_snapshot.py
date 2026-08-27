"""Point-in-time candidate snapshots.

Every property here guards a failure that produces a *trained ranker* rather
than an error. A snapshot with fabricated scores, a fold label where a
timestamp belongs, or a silently-missing source all yield a model that trains
cleanly, scores well offline, and is measuring something other than what it
claims.

These run before the six-hour refit, because that is the only point at which
finding them is cheap.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omnirank.core.exceptions import DataError
from omnirank.models.base import ScoredCandidate
from omnirank.ranking.candidate_snapshot import (
    CUTOFF_POLICY,
    SOURCES,
    RetrieverIdentity,
    SnapshotStats,
    build_manifest,
    build_snapshot_rows,
    drop_zero_positive_queries,
    guard_overwrite,
    query_groups,
    require_official_snapshot,
    snapshot_checksum,
    snapshot_columns,
    source_columns,
    validate_snapshot,
    write_manifest,
    write_snapshot_atomically,
)

CUTOFF = 1_500_000_000


def scored(source: str, items: list[str], scores: list[float | None]) -> list[ScoredCandidate]:
    """One source's ranked list with its own scores."""
    return [
        ScoredCandidate(item_id=item, rank=position, score=score, source=source)
        for position, (item, score) in enumerate(zip(items, scores, strict=True), start=1)
    ]


def rows_for(
    per_source: dict[str, list[ScoredCandidate]],
    fused: list[tuple[str, float]],
    *,
    target: str = "t",
    as_of: int = CUTOFF,
    query_id: str = "q1",
) -> list[dict]:
    """Build one query's snapshot rows against a trivial mapping."""
    mapping = {item: index for index, (item, _) in enumerate(fused)}
    mapping.setdefault(target, 999)
    return build_snapshot_rows(
        query_id=query_id,
        external_user_id="u1",
        internal_user_id=1,
        target_external_item=target,
        target_internal_item=999,
        as_of_timestamp=as_of,
        fold_id="offset_3",
        split="train",
        candidate_budget=500,
        per_source=per_source,
        fused=fused,
        external_to_internal_item=mapping,
    )


class TestSchema:
    def test_every_source_has_presence_rank_and_score(self) -> None:
        columns = source_columns()
        for source in SOURCES:
            for suffix in ("present", "rank", "score"):
                assert f"{source}_{suffix}" in columns

    def test_the_schema_names_matrix_factorization_not_bpr(self) -> None:
        """The column is named for the source, matching every other identifier."""
        assert "matrix_factorization_score" in source_columns()
        assert "bpr_score" not in source_columns()

    def test_fold_id_and_timestamp_are_separate_columns(self) -> None:
        """One is which experiment, the other is when. They are not the same."""
        columns = snapshot_columns()
        assert "fold_id" in columns
        assert "as_of_timestamp" in columns

    def test_the_aggregate_score_is_not_a_source_column(self) -> None:
        assert "aggregate_score" in snapshot_columns()
        assert "aggregate_score" not in source_columns()


class TestGenuineScores:
    def test_source_scores_are_the_models_own_values(self) -> None:
        """The defect this guards: 1/(rank+1) written into a score column."""
        per_source = {"popularity": scored("popularity", ["a", "b"], [15.8, 3.2])}
        rows = rows_for(per_source, [("a", 0.03), ("b", 0.016)])
        assert rows[0]["popularity_score"] == pytest.approx(15.8)
        assert rows[1]["popularity_score"] == pytest.approx(3.2)
        # A reciprocal-rank stand-in would have produced these instead.
        assert rows[0]["popularity_score"] != pytest.approx(1.0)
        assert rows[1]["popularity_score"] != pytest.approx(0.5)

    def test_rrf_score_is_kept_apart_from_source_scores(self) -> None:
        per_source = {"popularity": scored("popularity", ["a"], [15.8])}
        rows = rows_for(per_source, [("a", 0.0328)])
        assert rows[0]["aggregate_score"] == pytest.approx(0.0328)
        assert rows[0]["popularity_score"] == pytest.approx(15.8)

    def test_a_source_that_reports_no_score_records_missing_not_zero(self) -> None:
        """`None` means "this source has no score", which is not "scored zero"."""
        per_source = {"lightgcn": scored("lightgcn", ["a"], [None])}
        rows = rows_for(per_source, [("a", 0.016)])
        assert rows[0]["lightgcn_present"] == 1
        assert np.isnan(rows[0]["lightgcn_score"])

    def test_a_genuine_zero_score_survives_as_zero(self) -> None:
        """Zero is a real dot product and must not become missing."""
        per_source = {"matrix_factorization": scored("matrix_factorization", ["a"], [0.0])}
        rows = rows_for(per_source, [("a", 0.016)])
        assert rows[0]["matrix_factorization_score"] == 0.0
        assert not np.isnan(rows[0]["matrix_factorization_score"])

    def test_an_absent_source_is_missing_in_both_rank_and_score(self) -> None:
        per_source = {"popularity": scored("popularity", ["a"], [1.0])}
        rows = rows_for(per_source, [("a", 0.016)])
        assert rows[0]["sasrec_present"] == 0
        assert np.isnan(rows[0]["sasrec_rank"])
        assert np.isnan(rows[0]["sasrec_score"])

    def test_ranks_come_from_the_source_not_the_fused_order(self) -> None:
        per_source = {"sasrec": scored("sasrec", ["b", "a"], [9.0, 8.0])}
        rows = rows_for(per_source, [("a", 0.9), ("b", 0.8)])
        by_item = {row["external_item_id"]: row for row in rows}
        assert by_item["b"]["sasrec_rank"] == 1
        assert by_item["a"]["sasrec_rank"] == 2
        assert by_item["a"]["aggregate_rank"] == 1


class TestTimestamps:
    def test_a_real_unix_timestamp_is_stored(self) -> None:
        rows = rows_for({"popularity": scored("popularity", ["a"], [1.0])}, [("a", 0.1)])
        assert rows[0]["as_of_timestamp"] == CUTOFF
        assert rows[0]["fold_id"] == "offset_3"

    @pytest.mark.parametrize("bad", ["offset_3", None, 0, -1, 1.5])
    def test_a_non_timestamp_cutoff_is_refused(self, bad: object) -> None:
        """The defect this guards: a fold label stored where a time belongs."""
        with pytest.raises(DataError, match="positive Unix timestamp"):
            rows_for(
                {"popularity": scored("popularity", ["a"], [1.0])},
                [("a", 0.1)],
                as_of=bad,  # type: ignore[arg-type]
            )

    def test_the_cutoff_policy_is_recorded_rather_than_implied(self) -> None:
        assert "strictly earlier" in CUTOFF_POLICY


class TestLabelling:
    def test_the_target_is_labelled_only_when_genuinely_retrieved(self) -> None:
        per_source = {"popularity": scored("popularity", ["a", "t"], [2.0, 1.0])}
        rows = rows_for(per_source, [("a", 0.2), ("t", 0.1)])
        assert [row["label"] for row in rows] == [0, 1]

    def test_a_missed_target_produces_no_positive_and_is_not_inserted(self) -> None:
        """A retrieval miss must stay a miss. Adding the target fakes recall."""
        per_source = {"popularity": scored("popularity", ["a", "b"], [2.0, 1.0])}
        rows = rows_for(per_source, [("a", 0.2), ("b", 0.1)])
        assert sum(row["label"] for row in rows) == 0
        assert "t" not in {row["external_item_id"] for row in rows}

    def test_an_unmapped_item_is_refused(self) -> None:
        with pytest.raises(DataError, match="absent from the id mapping"):
            build_snapshot_rows(
                query_id="q",
                external_user_id="u",
                internal_user_id=1,
                target_external_item="t",
                target_internal_item=9,
                as_of_timestamp=CUTOFF,
                fold_id="offset_3",
                split="train",
                candidate_budget=10,
                per_source={},
                fused=[("ghost", 0.1)],
                external_to_internal_item={},
            )


def frame_from(records: list[tuple[str, str, int]]) -> pd.DataFrame:
    """A minimal valid snapshot frame from (query, item, label) triples."""
    rows = []
    for query, item, label in records:
        row = dict.fromkeys(snapshot_columns(), 0)
        row.update(
            query_id=query,
            external_item_id=item,
            label=label,
            aggregate_rank=1,
            as_of_timestamp=CUTOFF,
            fold_id="offset_3",
        )
        rows.append(row)
    return pd.DataFrame(rows)


class TestStructuralValidation:
    def test_a_well_formed_snapshot_passes(self) -> None:
        validate_snapshot(frame_from([("q1", "a", 0), ("q1", "t", 1), ("q2", "b", 0)]))

    def test_non_contiguous_query_rows_are_refused(self) -> None:
        """LightGBM groups are sizes, not ids: a scattered query mis-groups."""
        with pytest.raises(DataError, match="not contiguous"):
            validate_snapshot(frame_from([("q1", "a", 0), ("q2", "b", 0), ("q1", "c", 0)]))

    def test_duplicate_candidates_are_refused(self) -> None:
        with pytest.raises(DataError, match="duplicate candidates"):
            validate_snapshot(frame_from([("q1", "a", 0), ("q1", "a", 0)]))

    def test_multiple_positives_are_refused_under_leave_one_out(self) -> None:
        with pytest.raises(DataError, match="more than one positive"):
            validate_snapshot(frame_from([("q1", "a", 1), ("q1", "b", 1)]))

    def test_multiple_positives_are_allowed_when_declared(self) -> None:
        validate_snapshot(
            frame_from([("q1", "a", 1), ("q1", "b", 1)]), expect_single_positive=False
        )

    def test_an_empty_snapshot_is_refused(self) -> None:
        with pytest.raises(DataError, match="empty"):
            validate_snapshot(pd.DataFrame())

    def test_missing_canonical_columns_are_named(self) -> None:
        frame = frame_from([("q1", "a", 0)]).drop(columns=["sasrec_score"])
        with pytest.raises(DataError, match="missing canonical columns"):
            validate_snapshot(frame)


class TestQueryGroups:
    def test_group_sizes_sum_to_the_row_count(self) -> None:
        frame = frame_from([("q1", "a", 0), ("q1", "b", 1), ("q2", "c", 0)])
        groups = query_groups(frame)
        assert groups.tolist() == [2, 1]
        assert int(groups.sum()) == len(frame)

    def test_a_single_query_is_one_group(self) -> None:
        assert query_groups(frame_from([("q1", "a", 0), ("q1", "b", 0)])).tolist() == [2]


class TestZeroPositiveQueries:
    def test_zero_positive_groups_are_dropped_and_counted(self) -> None:
        frame = frame_from([("q1", "a", 0), ("q1", "t", 1), ("q2", "b", 0)])
        kept, dropped = drop_zero_positive_queries(frame)
        assert dropped == 1
        assert set(kept["query_id"]) == {"q1"}

    def test_dropping_never_mutates_the_original(self) -> None:
        """End-to-end evaluation needs the misses back."""
        frame = frame_from([("q1", "a", 0), ("q2", "t", 1)])
        drop_zero_positive_queries(frame)
        assert len(frame) == 2

    def test_candidate_recall_uses_every_query_as_the_denominator(self) -> None:
        stats = SnapshotStats(queries=10, positive_queries=3, zero_positive_queries=7)
        assert stats.candidate_recall == pytest.approx(0.3)


class TestChecksum:
    def test_identical_input_gives_an_identical_checksum(self) -> None:
        records = [("q1", "a", 0), ("q1", "t", 1)]
        assert snapshot_checksum(frame_from(records)) == snapshot_checksum(frame_from(records))

    def test_changed_labels_change_the_checksum(self) -> None:
        first = snapshot_checksum(frame_from([("q1", "a", 0)]))
        second = snapshot_checksum(frame_from([("q1", "a", 1)]))
        assert first != second

    def test_changed_candidates_change_the_checksum(self) -> None:
        first = snapshot_checksum(frame_from([("q1", "a", 0)]))
        second = snapshot_checksum(frame_from([("q1", "b", 0)]))
        assert first != second


def identity(source: str, *, max_fit: int = CUTOFF - 100, status: str = "ok") -> RetrieverIdentity:
    """A retriever provenance record with a boundary that holds."""
    return RetrieverIdentity(
        source=source,
        model_class=f"{source.title()}Model",
        model_version=f"pit-{source}",
        configuration_hash="cfg",
        seed=42,
        fit_fold="offset_3",
        fit_boundary="strictly before offset 3",
        fit_boundary_timestamp=CUTOFF,
        max_fit_timestamp=max_fit,
        fit_interactions=100,
        fit_users=10,
        fit_items=20,
        dataset_version="v1",
        split_version="1",
        mapping_checksum="map",
        candidate_budget=500,
        device="cpu",
        fit_seconds=1.0,
        retrieval_status=status,
    )


class TestProvenance:
    def test_a_boundary_before_the_cutoff_is_respected(self) -> None:
        assert identity("popularity").boundary_respected is True

    def test_training_past_the_cutoff_is_detected(self) -> None:
        """The claim the whole snapshot rests on, checked arithmetically."""
        assert identity("popularity", max_fit=CUTOFF + 1).boundary_respected is False

    def test_an_unknown_timestamp_is_unknown_not_assumed_safe(self) -> None:
        record = RetrieverIdentity(
            source="popularity",
            model_class="P",
            model_version="v",
            configuration_hash="c",
            seed=None,
            fit_fold="offset_3",
            fit_boundary="b",
            fit_boundary_timestamp=None,
            max_fit_timestamp=None,
            fit_interactions=1,
            fit_users=1,
            fit_items=1,
            dataset_version="v1",
            split_version="1",
            mapping_checksum="m",
            candidate_budget=500,
            device="cpu",
            fit_seconds=0.0,
        )
        assert record.boundary_respected is None

    def test_the_manifest_records_all_five_sources(self) -> None:
        manifest = build_manifest(
            fold_id="offset_3",
            split="train",
            stats=SnapshotStats(queries=1, rows=1, positive_queries=1),
            retrievers=[identity(source) for source in SOURCES],
            dataset_identity={"dataset_version": "v1"},
            mapping_checksum="map",
            candidate_budget=500,
            aggregation={"strategy": "reciprocal_rank_fusion"},
            checksum="abc",
            degraded=False,
        )
        assert {source["source"] for source in manifest["sources"]} == set(SOURCES)
        assert manifest["all_boundaries_respected"] is True
        assert manifest["degraded"] is False
        assert manifest["cutoff_policy"] == CUTOFF_POLICY

    def test_a_manifest_reports_a_violated_boundary(self) -> None:
        manifest = build_manifest(
            fold_id="offset_3",
            split="train",
            stats=SnapshotStats(),
            retrievers=[identity("popularity", max_fit=CUTOFF + 5)],
            dataset_identity={},
            mapping_checksum="m",
            candidate_budget=500,
            aggregation={},
            checksum="c",
            degraded=False,
        )
        assert manifest["all_boundaries_respected"] is False


class TestDegradedSnapshotRefusal:
    @staticmethod
    def manifest(**overrides: object) -> dict:
        base = build_manifest(
            fold_id="offset_3",
            split="train",
            stats=SnapshotStats(queries=1),
            retrievers=[identity(source) for source in SOURCES],
            dataset_identity={},
            mapping_checksum="m",
            candidate_budget=500,
            aggregation={},
            checksum="c",
            degraded=False,
        )
        base.update(overrides)
        return base

    def test_an_official_snapshot_is_accepted(self) -> None:
        require_official_snapshot(self.manifest(), purpose="selection")

    def test_a_degraded_snapshot_is_refused(self) -> None:
        with pytest.raises(DataError, match="degraded snapshot"):
            require_official_snapshot(
                self.manifest(degraded=True, degraded_sources=["sasrec"]), purpose="selection"
            )

    def test_a_snapshot_trained_past_the_cutoff_is_refused(self) -> None:
        with pytest.raises(DataError, match="trained past the cutoff"):
            require_official_snapshot(
                self.manifest(all_boundaries_respected=False), purpose="final evaluation"
            )

    def test_a_snapshot_missing_a_source_is_refused(self) -> None:
        partial = self.manifest()
        partial["sources"] = partial["sources"][:4]
        with pytest.raises(DataError, match="all five sources"):
            require_official_snapshot(partial, purpose="selection")


class TestOverwriteProtection:
    @staticmethod
    def existing(
        tmp_path: Path, *, fold: str = "offset_3", split: str = "train"
    ) -> tuple[Path, Path]:
        destination = tmp_path / "train_candidates.parquet"
        manifest_path = tmp_path / "train_snapshot_manifest.json"
        destination.write_bytes(b"parquet")
        manifest_path.write_text(
            json.dumps(
                {
                    "fold_id": fold,
                    "split": split,
                    "snapshot_checksum": "deadbeefdeadbeef",
                    "created_at": "2026-01-01T00:00:00Z",
                    "statistics": {"queries": 42},
                }
            )
        )
        return destination, manifest_path

    def test_a_fresh_destination_is_allowed(self, tmp_path: Path) -> None:
        guard_overwrite(
            tmp_path / "new.parquet",
            tmp_path / "new.json",
            overwrite=False,
            fold_id="offset_3",
            split="train",
        )

    def test_an_existing_snapshot_is_protected_by_default(self, tmp_path: Path) -> None:
        destination, manifest_path = self.existing(tmp_path)
        with pytest.raises(DataError, match="Pass --overwrite"):
            guard_overwrite(
                destination, manifest_path, overwrite=False, fold_id="offset_3", split="train"
            )

    def test_overwrite_permits_replacing_the_same_fold(self, tmp_path: Path) -> None:
        destination, manifest_path = self.existing(tmp_path)
        guard_overwrite(
            destination, manifest_path, overwrite=True, fold_id="offset_3", split="train"
        )

    def test_overwriting_a_different_fold_is_refused_even_with_the_flag(
        self, tmp_path: Path
    ) -> None:
        """A repeated command line with a changed offset is the real mistake."""
        destination, manifest_path = self.existing(tmp_path, fold="offset_2", split="validation")
        with pytest.raises(DataError, match="different fold or split"):
            guard_overwrite(
                destination, manifest_path, overwrite=True, fold_id="offset_3", split="train"
            )


class TestAtomicWrite:
    def test_the_snapshot_lands_complete(self, tmp_path: Path) -> None:
        frame = frame_from([("q1", "a", 0), ("q1", "t", 1)])
        destination = tmp_path / "out.parquet"
        write_snapshot_atomically(frame, destination)
        assert len(pd.read_parquet(destination)) == 2

    def test_no_temporary_file_survives(self, tmp_path: Path) -> None:
        destination = tmp_path / "out.parquet"
        write_snapshot_atomically(frame_from([("q1", "a", 0)]), destination)
        assert not list(tmp_path.glob("*.tmp"))

    def test_a_manifest_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        write_manifest({"fold_id": "offset_3", "degraded": False}, path)
        assert json.loads(path.read_text())["fold_id"] == "offset_3"
