"""Exact FAISS over two-tower embeddings, and the identity it must carry.

The exactness check is the only thing separating "fast" from "fast and wrong":
an index built with the wrong metric, over a transposed matrix, or from a
different model's embeddings still returns k neighbours with plausible scores
for every query, and nothing raises.

The identity checks cover a failure a collaborative index cannot have. A
LightGCN index is wrong if paired with the wrong model or mapping. A two-tower
index has a third way: its vectors derive from a feature store, so a store with
different content -- same items, same mapping -- produces a different index that
nothing downstream would notice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss", reason="Vector index requires the 'retrieval' extra")

from omnirank.core.exceptions import ArtifactValidationError, DataError
from omnirank.models.two_tower.catalogue import build_catalogue
from omnirank.retrieval.two_tower_index import (
    build_two_tower_index,
    load_item_embeddings,
    verify_index_against_brute_force,
    write_item_embeddings,
)

ITEMS = 200
DIMENSION = 32
COLD_FROM = 150


@pytest.fixture
def catalogue():
    """A catalogue with a genuine cold tail."""
    warm = np.zeros(ITEMS, dtype=bool)
    warm[:COLD_FROM] = True
    return build_catalogue(
        warm_items=warm,
        text_available=np.ones(ITEMS, dtype=bool),
        image_available=np.ones(ITEMS, dtype=bool),
        internal_to_external={index: f"i{index}" for index in range(ITEMS)},
    )


@pytest.fixture
def embeddings():
    """L2-normalised float32 vectors, as the item tower exports them."""
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(ITEMS, DIMENSION)).astype("float32")
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


@pytest.fixture
def identity() -> dict[str, str]:
    return {
        "model_version": "phase5-test",
        "model_checksum": "model-abc",
        "mapping_checksum": "map-abc",
        "feature_version": "1",
        "feature_manifest_checksum": "features-abc",
        "normalization": "l2",
    }


@pytest.fixture
def built(embeddings, catalogue, identity):
    return build_two_tower_index(embeddings, catalogue, **identity)


class TestExactness:
    def test_matches_brute_force_in_set_and_order(self, built, embeddings) -> None:
        index, _ = built
        queries = embeddings[:20]
        report = verify_index_against_brute_force(index, embeddings, queries, k=10)
        assert report["matches_brute_force"]
        assert report["exact_order_agreement"] == 1.0
        assert report["set_overlap"] == 1.0

    def test_score_difference_is_float32_noise(self, built, embeddings) -> None:
        index, _ = built
        report = verify_index_against_brute_force(index, embeddings, embeddings[:20], k=10)
        assert report["max_score_difference"] < 1e-4

    def test_a_vector_is_its_own_nearest_neighbour(self, built, embeddings) -> None:
        index, _ = built
        found, _ = index.search(embeddings[:5], 1)
        assert [row[0] for row in found] == [0, 1, 2, 3, 4]


class TestCatalogueComposition:
    def test_cold_items_are_in_the_index(self, built, catalogue) -> None:
        """Phase 5 rests on this; an index without them still answers queries."""
        _, metadata = built
        assert metadata["cold_item_count"] == ITEMS - COLD_FROM
        assert metadata["cold_item_count"] > 0

    def test_counts_add_up(self, built) -> None:
        _, metadata = built
        assert metadata["warm_item_count"] + metadata["cold_item_count"] == ITEMS

    def test_a_cold_item_can_be_retrieved(self, built, embeddings) -> None:
        """Query with a cold item's own vector; it must come back first."""
        index, _ = built
        found, _ = index.search(embeddings[COLD_FROM : COLD_FROM + 1], 1)
        assert found[0][0] == COLD_FROM


class TestMetadata:
    def test_records_scoring_semantics(self, built) -> None:
        _, metadata = built
        assert metadata["metric"] == "inner_product"
        assert metadata["normalization_policy"] == "l2"
        assert metadata["dimension"] == DIMENSION

    def test_records_every_identity(self, built) -> None:
        """Each of these, mismatched, produces wrong answers rather than errors."""
        _, metadata = built
        for key in (
            "model_version",
            "model_checksum",
            "embedding_checksum",
            "mapping_checksum",
            "feature_version",
            "feature_manifest_checksum",
            "catalogue_checksum",
            "index_checksum",
        ):
            assert metadata[key]

    def test_index_checksum_changes_with_content(self, embeddings, catalogue, identity) -> None:
        _, first = build_two_tower_index(embeddings, catalogue, **identity)
        altered = embeddings.copy()
        altered[0, 0] += np.float32(0.5)
        _, second = build_two_tower_index(altered, catalogue, **identity)
        assert first["embedding_checksum"] != second["embedding_checksum"]


class TestPersistence:
    def test_embeddings_round_trip(self, tmp_path, embeddings, catalogue, identity) -> None:
        write_item_embeddings(tmp_path / "emb", embeddings, catalogue, **identity)
        loaded, restored, manifest = load_item_embeddings(tmp_path / "emb")
        assert np.array_equal(loaded, embeddings)
        assert restored.checksum() == catalogue.checksum()
        assert manifest["cold_items"] == catalogue.cold_count

    def test_index_round_trip_returns_identical_results(self, tmp_path, built, embeddings) -> None:
        from omnirank.retrieval.faiss_index import FaissVectorIndex

        index, _ = built
        index.save(tmp_path / "index")
        reloaded = FaissVectorIndex.load(tmp_path / "index")
        assert reloaded.search(embeddings[:10], 5) == index.search(embeddings[:10], 5)

    def test_a_corrupted_matrix_is_detected(
        self, tmp_path, embeddings, catalogue, identity
    ) -> None:
        write_item_embeddings(tmp_path / "emb", embeddings, catalogue, **identity)
        altered = embeddings.copy()
        altered[0, 0] += np.float32(1.0)
        np.save(tmp_path / "emb" / "item_embeddings.npy", altered)
        with pytest.raises(ArtifactValidationError, match="checksum"):
            load_item_embeddings(tmp_path / "emb")

    def test_a_reordered_catalogue_is_detected(
        self, tmp_path, embeddings, catalogue, identity
    ) -> None:
        """Rows are positional; reordering pairs every embedding with the wrong item."""
        write_item_embeddings(tmp_path / "emb", embeddings, catalogue, **identity)
        table = pd.read_parquet(tmp_path / "emb" / "item_index.parquet")
        table = table.iloc[::-1].reset_index(drop=True)
        table.to_parquet(tmp_path / "emb" / "item_index.parquet", index=False)
        with pytest.raises(ArtifactValidationError, match="checksum"):
            load_item_embeddings(tmp_path / "emb")

    def test_missing_files_are_reported(self, tmp_path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(ArtifactValidationError):
            load_item_embeddings(tmp_path / "empty")


class TestValidation:
    def test_rejects_rows_that_do_not_match_the_catalogue(
        self, tmp_path, catalogue, identity
    ) -> None:
        with pytest.raises(DataError, match="do not match the catalogue"):
            write_item_embeddings(
                tmp_path / "emb",
                np.zeros((ITEMS - 1, DIMENSION), dtype="float32"),
                catalogue,
                **identity,
            )

    def test_rejects_non_float32(self, tmp_path, catalogue, identity) -> None:
        with pytest.raises(DataError, match="float32"):
            write_item_embeddings(
                tmp_path / "emb",
                np.zeros((ITEMS, DIMENSION), dtype="float64"),
                catalogue,
                **identity,
            )

    def test_rejects_non_finite_values(self, tmp_path, catalogue, identity) -> None:
        broken = np.zeros((ITEMS, DIMENSION), dtype="float32")
        broken[0, 0] = np.nan
        with pytest.raises(DataError, match="non-finite"):
            write_item_embeddings(tmp_path / "emb", broken, catalogue, **identity)


class TestIndexCompatibility:
    def test_matching_identity_passes(self, built) -> None:
        index, _ = built
        index.require_compatible(
            model_name="two_tower",
            model_version="phase5-test",
            item_mapping_checksum="map-abc",
        )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("model_name", "lightgcn"),
            ("model_version", "another-version"),
            ("item_mapping_checksum", "another-mapping"),
        ],
    )
    def test_any_mismatch_is_refused(self, built, field: str, value: str) -> None:
        index, _ = built
        identity = {
            "model_name": "two_tower",
            "model_version": "phase5-test",
            "item_mapping_checksum": "map-abc",
        }
        identity[field] = value
        with pytest.raises(ArtifactValidationError):
            index.require_compatible(**identity)

    def test_a_different_feature_store_produces_a_different_index(
        self, embeddings, catalogue, identity
    ) -> None:
        """The failure a collaborative index cannot have."""
        _, first = build_two_tower_index(embeddings, catalogue, **identity)
        other = {**identity, "feature_manifest_checksum": "different-features"}
        _, second = build_two_tower_index(embeddings, catalogue, **other)
        assert first["index_checksum"] != second["index_checksum"]
