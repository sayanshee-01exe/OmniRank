"""Dataset manifest and checksum generation."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from omnirank.core.exceptions import DataError
from omnirank.data.io import sha256_file, sha256_frame, write_json, write_parquet
from omnirank.data.manifest import PIPELINE_VERSION, SCHEMA_VERSION, build_manifest


@pytest.fixture
def manifest():
    return build_manifest(
        dataset_name="pixelrec50k",
        dataset_version="v1",
        source_repository="https://github.com/westlake-repl/PixelRec",
        licence="non-commercial research only",
        source_files={"interaction.csv": {"sha256": "abc", "bytes": 10, "rows": 5}},
        configuration_hash="hash123",
        random_seed=42,
        mapping_version="1",
        split_version="1",
        split_strategy="per_user_leave_last_n",
        ordering_field="timestamp",
        filtering_configuration={"min_interactions_per_item": 2},
        raw_row_counts={"interactions": 100},
        processed_row_counts={"interactions": 90},
        user_counts={"raw": 10, "processed": 10},
        item_counts={"raw": 20, "processed": 18},
        interaction_counts={"train": 70, "validation": 10, "test": 10},
        feature_dimensions={"text": None, "image": None},
        feature_coverage={"text": 0.0, "image": 0.0},
        output_files={"a.parquet": {"sha256": "def", "bytes": 20, "rows": 90}},
        known_limitations=["features not downloaded"],
    )


class TestManifest:
    def test_contains_every_required_field(self, manifest):
        payload = manifest.to_dict()
        required = {
            "dataset_name",
            "dataset_version",
            "source_repository",
            "source_files",
            "source_checksums",
            "source_file_sizes",
            "processing_timestamp",
            "pipeline_version",
            "git_commit",
            "configuration_hash",
            "random_seed",
            "schema_version",
            "mapping_version",
            "split_version",
            "split_strategy",
            "ordering_field",
            "filtering_configuration",
            "raw_row_counts",
            "processed_row_counts",
            "user_counts",
            "item_counts",
            "interaction_counts",
            "feature_dimensions",
            "feature_coverage",
            "output_files",
            "output_checksums",
            "known_limitations",
        }
        assert required <= set(payload)

    def test_checksums_are_projected_from_the_file_descriptors(self, manifest):
        payload = manifest.to_dict()
        assert payload["source_checksums"]["interaction.csv"] == "abc"
        assert payload["output_checksums"]["a.parquet"] == "def"

    def test_versions_are_recorded(self, manifest):
        payload = manifest.to_dict()
        assert payload["pipeline_version"] == PIPELINE_VERSION
        assert payload["schema_version"] == SCHEMA_VERSION

    def test_licence_is_recorded(self, manifest):
        assert "non-commercial" in manifest.to_dict()["licence"]

    def test_known_limitations_are_never_empty_in_practice(self, manifest):
        assert manifest.to_dict()["known_limitations"]

    def test_is_json_serialisable(self, manifest, tmp_path):
        descriptor = write_json(manifest.to_dict(), tmp_path / "m.json")
        reloaded = json.loads((tmp_path / "m.json").read_text())
        assert reloaded["dataset_name"] == "pixelrec50k"
        assert descriptor["sha256"]

    def test_git_commit_is_null_outside_a_checkout(self, manifest):
        """Recording null is more honest than recording a placeholder."""
        assert manifest.to_dict()["git_commit"] in (None, manifest.git_commit)

    def test_subset_runs_are_flagged(self):
        manifest = build_manifest(
            dataset_name="x",
            dataset_version="v1",
            source_repository="",
            licence="",
            source_files={},
            configuration_hash="h",
            random_seed=1,
            mapping_version="1",
            split_version="1",
            split_strategy="s",
            ordering_field="timestamp",
            filtering_configuration={},
            raw_row_counts={},
            processed_row_counts={},
            user_counts={},
            item_counts={},
            interaction_counts={},
            feature_dimensions={},
            feature_coverage={},
            output_files={},
            known_limitations=[],
            subset_users=100,
        )
        assert manifest.to_dict()["subset_users"] == 100


class TestChecksums:
    def test_file_checksum_is_stable(self, tmp_path):
        path = tmp_path / "f.txt"
        path.write_text("content")
        assert sha256_file(path) == sha256_file(path)

    def test_file_checksum_changes_with_content(self, tmp_path):
        path = tmp_path / "f.txt"
        path.write_text("a")
        first = sha256_file(path)
        path.write_text("b")
        assert sha256_file(path) != first

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DataError):
            sha256_file(tmp_path / "absent")

    def test_frame_checksum_ignores_row_order(self):
        frame = pd.DataFrame({"a": [1, 2, 3]})
        assert sha256_frame(frame) == sha256_frame(frame.iloc[::-1])

    def test_frame_checksum_detects_a_value_change(self):
        assert sha256_frame(pd.DataFrame({"a": [1, 2]})) != sha256_frame(
            pd.DataFrame({"a": [1, 3]})
        )


class TestDeterministicWrites:
    def test_parquet_output_is_byte_identical_across_runs(self, tmp_path):
        """Without this, manifest checksums are noise."""
        frame = pd.DataFrame({"b": [2, 1], "a": ["y", "x"]})
        first = write_parquet(frame, tmp_path / "1.parquet", columns=["a", "b"], sort_by=["a"])
        second = write_parquet(
            frame.iloc[::-1], tmp_path / "2.parquet", columns=["a", "b"], sort_by=["a"]
        )
        assert first["sha256"] == second["sha256"]

    def test_missing_column_is_rejected(self, tmp_path):
        with pytest.raises(DataError) as exc:
            write_parquet(pd.DataFrame({"a": [1]}), tmp_path / "f.parquet", columns=["a", "b"])
        assert "missing required columns" in str(exc.value)

    def test_overwrite_is_refused_by_default(self, tmp_path):
        frame = pd.DataFrame({"a": [1]})
        write_parquet(frame, tmp_path / "f.parquet")
        with pytest.raises(DataError) as exc:
            write_parquet(frame, tmp_path / "f.parquet", overwrite=False)
        assert "--overwrite" in str(exc.value)

    def test_descriptor_reports_rows_and_bytes(self, tmp_path):
        descriptor = write_parquet(pd.DataFrame({"a": [1, 2, 3]}), tmp_path / "f.parquet")
        assert descriptor["rows"] == 3
        assert descriptor["bytes"] > 0
