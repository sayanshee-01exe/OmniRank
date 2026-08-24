"""End-to-end dataset pipeline over a synthetic PixelRec-shaped fixture.

Exercises every stage in sequence: raw fixture, canonical data, mappings,
filtering, splitting, graph, sequences, feature alignment, evaluation slices,
reports, manifest.

Offline, CPU-only, no database, no network, no pretrained weights, a fraction of
a second. The fixture is generated, never the real dataset - its licence forbids
redistribution and the tests must run on a fresh checkout.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from omnirank.data.pipeline import (
    COLLABORATIVE_COLUMNS,
    GRAPH_COLUMNS,
    ITEM_METADATA_COLUMNS,
    PipelineOptions,
    run_pipeline,
)
from omnirank.data.sequences import SEQUENCE_COLUMNS
from tests.fixtures.pixelrec import write_feature_file

pytestmark = pytest.mark.integration


@pytest.fixture
def processed_dir(pixelrec_config, tmp_path):
    return tmp_path / "processed"


@pytest.fixture
def result(pixelrec_config, tmp_path):
    """One full pipeline run over the fixture."""
    return run_pipeline(pixelrec_config, PipelineOptions(overwrite=True), project_root=tmp_path)


def read(processed_dir, relative: str) -> pd.DataFrame:
    return pd.read_parquet(processed_dir / relative)


class TestRunCompletes:
    def test_pipeline_succeeds(self, result):
        assert result.manifest_path is not None
        assert result.leakage_passed

    def test_counts_are_reported(self, result):
        assert result.counts["interactions"] > 0
        assert result.counts["train"] > 0
        assert result.counts["test"] > 0


class TestOutputFiles:
    @pytest.mark.parametrize(
        "relative",
        [
            "train_interactions.parquet",
            "validation_interactions.parquet",
            "test_interactions.parquet",
            "collaborative/interactions.parquet",
            "graph/train_graph_edges.parquet",
            "sequential/train_sequences.parquet",
            "sequential/validation_sequences.parquet",
            "sequential/test_sequences.parquet",
            "metadata/item_metadata.parquet",
            "features/user_training_statistics.parquet",
            "features/item_training_popularity.parquet",
            "features/text_feature_index.parquet",
            "features/image_feature_index.parquet",
            "split_metadata.json",
            "dataset_manifest.json",
        ],
    )
    def test_expected_output_exists(self, result, processed_dir, relative):
        assert (processed_dir / relative).is_file()

    def test_interim_outputs_exist(self, result, tmp_path):
        interim = tmp_path / "interim"
        for name in (
            "canonical_users.parquet",
            "canonical_items.parquet",
            "canonical_interactions.parquet",
            "rejected_records.parquet",
        ):
            assert (interim / name).is_file()

    def test_mappings_are_written(self, result, tmp_path):
        mappings = tmp_path / "artifacts" / "mappings" / "pixelrec50k"
        assert (mappings / "user_id_mapping.parquet").is_file()
        assert (mappings / "item_id_mapping.parquet").is_file()
        assert (mappings / "mapping_metadata.json").is_file()

    def test_reports_are_written(self, result, tmp_path):
        reports = tmp_path / "reports" / "data_quality" / "pixelrec50k"
        for relative in (
            "raw/raw_profile.json",
            "raw/raw_profile.md",
            "raw/missingness.csv",
            "raw/user_activity.csv",
            "raw/item_popularity.csv",
            "raw/feature_coverage.csv",
            "raw/validation_failures.csv",
            "filtering/filtering_report.json",
            "filtering/filtering_report.md",
            "leakage/leakage_report.json",
            "leakage/leakage_report.md",
            "processed/processed_profile.json",
            "processed/processed_profile.md",
        ):
            assert (reports / relative).is_file(), relative


class TestSchemas:
    def test_collaborative_columns(self, result, processed_dir):
        frame = read(processed_dir, "collaborative/interactions.parquet")
        assert list(frame.columns) == list(COLLABORATIVE_COLUMNS)

    def test_graph_columns(self, result, processed_dir):
        frame = read(processed_dir, "graph/train_graph_edges.parquet")
        assert set(GRAPH_COLUMNS) <= set(frame.columns)

    def test_sequence_columns(self, result, processed_dir):
        frame = read(processed_dir, "sequential/test_sequences.parquet")
        assert list(frame.columns) == list(SEQUENCE_COLUMNS)

    def test_item_metadata_columns(self, result, processed_dir):
        frame = read(processed_dir, "metadata/item_metadata.parquet")
        assert list(frame.columns) == list(ITEM_METADATA_COLUMNS)

    def test_no_ecommerce_fields_appear(self, result, processed_dir):
        frame = read(processed_dir, "metadata/item_metadata.parquet")
        for forbidden in ("price", "brand", "rating", "inventory"):
            assert forbidden not in frame.columns


class TestSplitIntegrity:
    def test_splits_partition_the_interactions(self, result, processed_dir):
        combined = read(processed_dir, "collaborative/interactions.parquet")
        parts = sum(
            len(read(processed_dir, f"{split}_interactions.parquet"))
            for split in ("train", "validation", "test")
        )
        assert parts == len(combined)

    def test_train_precedes_held_out_for_every_user(self, result, processed_dir):
        combined = read(processed_dir, "collaborative/interactions.parquet")
        for _, group in combined.groupby("internal_user_id"):
            train = group[group.split == "train"]["interaction_order"]
            held = group[group.split != "train"]["interaction_order"]
            if len(train) and len(held):
                assert train.max() < held.min()

    def test_split_metadata_matches_the_files(self, result, processed_dir):
        metadata = json.loads((processed_dir / "split_metadata.json").read_text())
        for split in ("train", "validation", "test"):
            assert metadata[f"{split}_rows"] == len(
                read(processed_dir, f"{split}_interactions.parquet")
            )


class TestGraph:
    def test_edges_come_only_from_training(self, result, processed_dir):
        edges = read(processed_dir, "graph/train_graph_edges.parquet")
        combined = read(processed_dir, "collaborative/interactions.parquet")
        train_pairs = set(
            zip(
                combined.loc[combined.split == "train", "internal_user_id"],
                combined.loc[combined.split == "train", "internal_item_id"],
                strict=True,
            )
        )
        edge_pairs = set(zip(edges["internal_user_id"], edges["internal_item_id"], strict=True))
        assert edge_pairs <= train_pairs

    def test_edge_weights_are_binary_and_documented(self, result, processed_dir):
        edges = read(processed_dir, "graph/train_graph_edges.parquet")
        assert set(edges["edge_weight"]) == {1.0}

    def test_raw_repeat_counts_are_preserved(self, result, processed_dir):
        edges = read(processed_dir, "graph/train_graph_edges.parquet")
        assert "interaction_count" in edges.columns
        assert (edges["interaction_count"] >= 1).all()


class TestSequences:
    def test_targets_are_never_in_their_own_input(self, result, processed_dir):
        for split in ("train", "validation", "test"):
            frame = read(processed_dir, f"sequential/{split}_sequences.parquet")
            for _, row in frame.iterrows():
                assert row["target_item"] not in list(row["item_sequence"])

    def test_history_is_strictly_before_the_target(self, result, processed_dir):
        for split in ("train", "validation", "test"):
            frame = read(processed_dir, f"sequential/{split}_sequences.parquet")
            for _, row in frame.iterrows():
                assert all(
                    order < row["target_order"] for order in row["interaction_order_sequence"]
                )

    def test_max_length_is_respected(self, result, processed_dir, pixelrec_config):
        limit = pixelrec_config.data.sequences.max_length
        for split in ("train", "validation", "test"):
            frame = read(processed_dir, f"sequential/{split}_sequences.parquet")
            if len(frame):
                assert frame["sequence_length"].max() <= limit


class TestStatistics:
    def test_popularity_matches_a_training_only_recount(self, result, processed_dir):
        popularity = read(processed_dir, "features/item_training_popularity.parquet")
        combined = read(processed_dir, "collaborative/interactions.parquet")
        expected = combined[combined.split == "train"].groupby("internal_item_id").size().to_dict()
        actual = dict(
            zip(
                popularity["internal_item_id"],
                popularity["training_interaction_count"],
                strict=True,
            )
        )
        assert actual == expected

    def test_user_statistics_match_a_training_only_recount(self, result, processed_dir):
        statistics = read(processed_dir, "features/user_training_statistics.parquet")
        combined = read(processed_dir, "collaborative/interactions.parquet")
        expected = combined[combined.split == "train"].groupby("internal_user_id").size().to_dict()
        actual = dict(
            zip(
                statistics["internal_user_id"],
                statistics["training_interaction_count"],
                strict=True,
            )
        )
        assert actual == expected


class TestFeatures:
    def test_absent_features_are_reported_as_absent(self, result, processed_dir):
        index = read(processed_dir, "features/text_feature_index.parquet")
        assert not index["has_text_feature"].any()

    def test_present_features_are_aligned(self, pixelrec_config, tmp_path, pixelrec_fixture_dir):
        """Re-run with a real feature file to prove alignment works end to end.

        The vectors are written at the profile's declared 1024 dimensions; a
        narrower fixture would be rejected by the dimension assertion, which is
        the correct behaviour and is covered by the unit tests.
        """
        items = pd.read_csv(pixelrec_fixture_dir / "item_info.csv")["item_id"].tolist()
        write_feature_file(
            pixelrec_fixture_dir / "text_feature.json",
            item_ids=items[:10],
            dimension=pixelrec_config.data.features.expected_dimension,
        )
        run_pipeline(pixelrec_config, PipelineOptions(overwrite=True), project_root=tmp_path)
        index = pd.read_parquet(tmp_path / "processed" / "features" / "text_feature_index.parquet")
        assert index["has_text_feature"].any()
        assert (tmp_path / "processed" / "features" / "text_features.npy").is_file()


class TestSlices:
    def test_slice_manifest_lists_every_slice(self, result, processed_dir):
        manifest = json.loads(
            (processed_dir / "evaluation_slices" / "slice_manifest.json").read_text()
        )
        names = {entry["slice_name"] for entry in manifest}
        assert "items_long_tail" in names
        assert "users_cold_start" in names
        assert "items_cold_start" in names

    def test_cold_user_slice_is_empty_by_construction(self, result, processed_dir):
        frame = read(processed_dir, "evaluation_slices/users_cold_start.parquet")
        assert len(frame) == 0


class TestManifest:
    def test_is_valid_and_complete(self, result, processed_dir):
        manifest = json.loads((processed_dir / "dataset_manifest.json").read_text())
        assert manifest["dataset_name"] == "pixelrec50k"
        assert manifest["split_strategy"] == "per_user_leave_last_n"
        assert manifest["ordering_field"] == "timestamp"
        assert manifest["source_checksums"]
        assert manifest["output_checksums"]

    def test_records_honest_limitations(self, result, processed_dir):
        manifest = json.loads((processed_dir / "dataset_manifest.json").read_text())
        limitations = " ".join(manifest["known_limitations"])
        assert "interaction" in limitations
        assert "coverage is 0.0" in limitations.lower() or "not present" in limitations.lower()

    def test_licence_is_recorded(self, result, processed_dir):
        manifest = json.loads((processed_dir / "dataset_manifest.json").read_text())
        assert "non-commercial" in manifest["licence"]


class TestDeterminism:
    def test_two_runs_produce_identical_outputs(self, pixelrec_config, tmp_path):
        first = run_pipeline(
            pixelrec_config, PipelineOptions(overwrite=True), project_root=tmp_path
        )
        checksums = {
            name: descriptor["sha256"]
            for name, descriptor in first.outputs.items()
            if isinstance(descriptor, dict) and "sha256" in descriptor
        }
        second = run_pipeline(
            pixelrec_config, PipelineOptions(overwrite=True), project_root=tmp_path
        )
        for name, digest in checksums.items():
            # The manifest embeds a timestamp, so only data files are compared.
            if name.endswith(".parquet"):
                assert second.outputs[name]["sha256"] == digest, name


class TestModes:
    def test_validate_only_stops_before_loading(self, pixelrec_config, tmp_path):
        result = run_pipeline(
            pixelrec_config, PipelineOptions(validate_only=True), project_root=tmp_path
        )
        assert result.manifest_path is None
        assert not (tmp_path / "processed" / "dataset_manifest.json").exists()

    def test_profile_only_writes_reports_but_no_dataset(self, pixelrec_config, tmp_path):
        result = run_pipeline(
            pixelrec_config, PipelineOptions(profile_only=True), project_root=tmp_path
        )
        reports = tmp_path / "reports" / "data_quality" / "pixelrec50k" / "raw"
        assert (reports / "raw_profile.json").is_file()
        assert result.manifest_path is None
        assert not (tmp_path / "processed" / "dataset_manifest.json").exists()
