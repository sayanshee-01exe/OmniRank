"""Internal ID mappings: determinism, reversibility, contiguity, persistence."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from omnirank.core.exceptions import IdMappingError
from omnirank.data.mapping import (
    FIRST_INTERNAL_ID,
    UNKNOWN_INTERNAL_ID,
    build_dataset_mappings,
    build_entity_mapping,
    load_mappings,
    write_mappings,
)


@pytest.fixture
def interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "external_user_id": ["u3", "u1", "u2", "u1"],
            "external_item_id": ["i9", "i2", "i9", "i5"],
        }
    )


@pytest.fixture
def mappings(interactions):
    return build_dataset_mappings(interactions, dataset_version="v1")


class TestConstruction:
    def test_ids_start_at_the_documented_value(self, mappings):
        assert mappings.users.frame["internal_user_id"].min() == FIRST_INTERNAL_ID

    def test_ids_are_contiguous(self, mappings):
        mappings.users.check_contiguous()
        mappings.items.check_contiguous()

    def test_assignment_follows_sorted_external_ids(self, mappings):
        """Sorting is what makes the mapping independent of source row order."""
        assert mappings.users.to_internal() == {"u1": 0, "u2": 1, "u3": 2}

    def test_duplicates_collapse(self, mappings):
        assert mappings.users.size == 3
        assert mappings.items.size == 3

    def test_empty_input_is_rejected(self):
        with pytest.raises(IdMappingError):
            build_entity_mapping(
                pd.Series([], dtype="object"),
                entity="user",
                external_column="external_user_id",
                internal_column="internal_user_id",
            )

    def test_row_order_does_not_affect_the_mapping(self, interactions):
        forward = build_dataset_mappings(interactions, dataset_version="v1")
        reverse = build_dataset_mappings(interactions.iloc[::-1], dataset_version="v1")
        assert forward.users.to_internal() == reverse.users.to_internal()
        assert forward.users.checksum == reverse.users.checksum


class TestReversibility:
    def test_round_trip(self, mappings):
        forward = mappings.items.to_internal()
        backward = mappings.items.to_external()
        for external, internal in forward.items():
            assert backward[internal] == external

    def test_reverse_mapping_is_complete(self, mappings):
        assert len(mappings.items.to_external()) == mappings.items.size


class TestAttachingIds:
    def test_adds_both_internal_columns(self, mappings, interactions):
        attached = mappings.attach_internal_ids(interactions)
        assert attached["internal_user_id"].dtype == "int64"
        assert attached["internal_item_id"].dtype == "int64"

    def test_values_match_the_mapping(self, mappings, interactions):
        attached = mappings.attach_internal_ids(interactions)
        lookup = mappings.users.to_internal()
        for _, row in attached.iterrows():
            assert row["internal_user_id"] == lookup[row["external_user_id"]]

    def test_unmapped_ids_fail_loudly(self, mappings):
        """Silent mis-resolution is the failure this exists to prevent."""
        stray = pd.DataFrame({"external_user_id": ["u_unknown"], "external_item_id": ["i2"]})
        with pytest.raises(IdMappingError) as exc:
            mappings.attach_internal_ids(stray)
        assert "not in the mapping" in str(exc.value)


class TestMetadata:
    def test_contains_every_required_field(self, mappings):
        metadata = mappings.metadata()
        assert {
            "mapping_version",
            "dataset_version",
            "created_at",
            "number_of_users",
            "number_of_items",
            "user_mapping_checksum",
            "item_mapping_checksum",
            "mapping_strategy",
            "unknown_user_policy",
            "unknown_item_policy",
        } <= set(metadata)

    def test_counts_match(self, mappings):
        metadata = mappings.metadata()
        assert metadata["number_of_users"] == 3
        assert metadata["number_of_items"] == 3

    def test_checksums_differ_between_entities(self, mappings):
        metadata = mappings.metadata()
        assert metadata["user_mapping_checksum"] != metadata["item_mapping_checksum"]

    def test_unknown_policies_name_the_sentinel(self, mappings):
        metadata = mappings.metadata()
        assert str(UNKNOWN_INTERNAL_ID) in metadata["unknown_user_policy"]


class TestPhase1Bridge:
    def test_converts_to_the_phase_1_id_mapping_contract(self, mappings):
        id_mapping = mappings.items.as_id_mapping()
        assert len(id_mapping) == mappings.items.size
        assert id_mapping.to_index("i2") == mappings.items.to_internal()["i2"]

    def test_fingerprint_is_recorded_for_artifact_metadata(self, mappings):
        assert mappings.metadata()["item_mapping_fingerprint"] == (
            mappings.items.as_id_mapping().fingerprint
        )


class TestPersistence:
    def test_round_trip(self, mappings, tmp_path):
        write_mappings(mappings, tmp_path)
        loaded = load_mappings(tmp_path, dataset_version="v1")
        assert loaded.users.to_internal() == mappings.users.to_internal()
        assert loaded.items.to_internal() == mappings.items.to_internal()

    def test_writes_the_three_expected_files(self, mappings, tmp_path):
        outputs = write_mappings(mappings, tmp_path)
        assert set(outputs) == {
            "user_id_mapping.parquet",
            "item_id_mapping.parquet",
            "mapping_metadata.json",
        }
        for name in outputs:
            assert (tmp_path / name).is_file()

    def test_metadata_file_is_valid_json(self, mappings, tmp_path):
        write_mappings(mappings, tmp_path)
        payload = json.loads((tmp_path / "mapping_metadata.json").read_text())
        assert payload["number_of_users"] == 3

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(IdMappingError):
            load_mappings(tmp_path, dataset_version="v1")

    def test_checksum_survives_the_round_trip(self, mappings, tmp_path):
        write_mappings(mappings, tmp_path)
        loaded = load_mappings(tmp_path, dataset_version="v1")
        assert loaded.users.checksum == mappings.users.checksum
