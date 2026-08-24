"""Multimodal feature streaming, validation, and alignment."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from omnirank.core.exceptions import DataSourceError
from omnirank.data.pixelrec.features import (
    align_features,
    stream_feature_vectors,
    write_feature_matrix,
)
from tests.fixtures.pixelrec import write_feature_file


@pytest.fixture
def mapping() -> pd.DataFrame:
    return pd.DataFrame({"external_item_id": ["i0", "i1", "i2"], "internal_item_id": [0, 1, 2]})


@pytest.fixture
def feature_file(tmp_path):
    return write_feature_file(tmp_path / "text_feature.json", item_ids=["i0", "i1"], dimension=4)


class TestStreaming:
    def test_reads_every_record(self, tmp_path):
        path = write_feature_file(
            tmp_path / "f.json", item_ids=[f"i{n}" for n in range(50)], dimension=8
        )
        assert len(dict(stream_feature_vectors(path))) == 50

    def test_filters_to_wanted_ids(self, feature_file):
        assert set(dict(stream_feature_vectors(feature_file, wanted_ids={"i1"}))) == {"i1"}

    def test_vectors_are_intact(self, feature_file):
        vectors = dict(stream_feature_vectors(feature_file))
        assert vectors["i0"] == [0.0, 1.0, 2.0, 3.0]

    @pytest.mark.parametrize("block", [16, 64, 512, 1_000_000])
    def test_block_size_does_not_change_the_result(self, tmp_path, block):
        """The parser must not depend on records fitting inside one read."""
        path = write_feature_file(
            tmp_path / "f.json", item_ids=[f"i{n}" for n in range(30)], dimension=16
        )
        assert dict(stream_feature_vectors(path, read_block_bytes=block)) == dict(
            stream_feature_vectors(path, read_block_bytes=1_000_000)
        )

    def test_handles_a_record_spanning_many_blocks(self, tmp_path):
        path = write_feature_file(tmp_path / "f.json", item_ids=["i0", "i1"], dimension=1024)
        vectors = dict(stream_feature_vectors(path, read_block_bytes=32))
        assert len(vectors) == 2
        assert len(vectors["i0"]) == 1024

    def test_handles_whitespace_and_newlines(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_text('\n  {\n  "i0" : [1.0, 2.0],\n  "i1" : [3.0, 4.0]\n }\n')
        assert dict(stream_feature_vectors(path)) == {"i0": [1.0, 2.0], "i1": [3.0, 4.0]}

    def test_empty_object(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_text("{}")
        assert dict(stream_feature_vectors(path)) == {}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DataSourceError):
            list(stream_feature_vectors(tmp_path / "absent.json"))

    def test_non_object_json_is_rejected(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(DataSourceError) as exc:
            list(stream_feature_vectors(path))
        assert "JSON object" in str(exc.value)


class TestAlignment:
    def test_maps_vectors_onto_internal_ids(self, feature_file, mapping):
        _index, matrix, validation = align_features("text", feature_file, mapping)
        assert validation.available
        assert validation.dimension == 4
        assert matrix.shape == (3, 4)
        assert np.allclose(matrix[0], [0.0, 1.0, 2.0, 3.0])

    def test_missing_items_are_flagged_not_dropped(self, feature_file, mapping):
        index, _, _validation = align_features("text", feature_file, mapping)
        assert len(index) == 3
        assert index.set_index("internal_item_id").loc[2, "has_text_feature"] is np.False_
        assert index.set_index("internal_item_id").loc[2, "text_feature_row"] == -1

    def test_coverage_is_measured_not_assumed(self, feature_file, mapping):
        _, _, validation = align_features("text", feature_file, mapping)
        assert validation.coverage == pytest.approx(2 / 3)

    def test_absent_file_degrades_honestly(self, tmp_path, mapping):
        """A missing modality must never be reported as present."""
        index, matrix, validation = align_features("image", tmp_path / "absent.json", mapping)
        assert validation.available is False
        assert validation.coverage == 0.0
        assert matrix is None
        assert not index["has_image_feature"].any()
        assert "not downloaded by default" in validation.notes

    def test_none_path_degrades_honestly(self, mapping):
        _, matrix, validation = align_features("text", None, mapping)
        assert validation.available is False
        assert matrix is None

    def test_index_schema_is_the_same_present_or_absent(self, feature_file, mapping, tmp_path):
        present, _, _ = align_features("text", feature_file, mapping)
        absent, _, _ = align_features("text", tmp_path / "absent.json", mapping)
        assert list(present.columns) == list(absent.columns)

    def test_dimension_mismatch_is_counted_and_excluded(self, tmp_path, mapping):
        path = tmp_path / "f.json"
        path.write_text(json.dumps({"i0": [1.0, 2.0], "i1": [1.0, 2.0, 3.0]}))
        _, _, validation = align_features("text", path, mapping, expected_dimension=2)
        assert validation.dimension_mismatches == 1
        assert validation.rows_matched == 1

    def test_nan_rows_are_rejected(self, tmp_path, mapping):
        path = tmp_path / "f.json"
        path.write_text('{"i0": [1.0, NaN], "i1": [1.0, 2.0]}')
        _, _, validation = align_features("text", path, mapping)
        assert validation.rows_with_nan == 1
        assert validation.rows_matched == 1

    def test_infinite_rows_are_rejected(self, tmp_path, mapping):
        path = tmp_path / "f.json"
        path.write_text('{"i0": [1.0, Infinity], "i1": [1.0, 2.0]}')
        _, _, validation = align_features("text", path, mapping)
        assert validation.rows_with_inf == 1

    def test_duplicate_ids_are_counted_and_first_wins(self, tmp_path, mapping):
        path = tmp_path / "f.json"
        path.write_text('{"i0": [1.0, 2.0], "i0": [9.0, 9.0]}')
        _, matrix, validation = align_features("text", path, mapping)
        assert validation.duplicate_ids == 1
        assert np.allclose(matrix[0], [1.0, 2.0])

    def test_normalisation_is_reported_as_not_applied(self, feature_file, mapping):
        """The source documents none, so none is claimed."""
        _, _, validation = align_features("text", feature_file, mapping)
        assert validation.normalized is False

    def test_encoder_is_none_unless_supplied(self, feature_file, mapping):
        _, _, validation = align_features("text", feature_file, mapping)
        assert validation.encoder is None


class TestMatrixPersistence:
    def test_round_trip_is_memory_mappable(self, feature_file, mapping, tmp_path):
        _, matrix, _ = align_features("text", feature_file, mapping)
        descriptor = write_feature_matrix(matrix, tmp_path / "text.npy")
        loaded = np.load(tmp_path / "text.npy", mmap_mode="r")
        assert descriptor["shape"] == [3, 4]
        assert descriptor["dtype"] == "float32"
        assert np.allclose(loaded[0], matrix[0])

    def test_none_matrix_writes_nothing(self, tmp_path):
        assert write_feature_matrix(None, tmp_path / "absent.npy") is None
        assert not (tmp_path / "absent.npy").exists()
