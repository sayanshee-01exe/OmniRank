"""The FAISS vector index.

An approximate index is checked against exact brute force. That comparison is
the only thing that distinguishes "fast" from "fast and wrong": an index built
with a mismatched metric, or over a transposed matrix, still returns k
neighbours with plausible scores for every query. Nothing raises.

The exclusion path gets its own attention because it is where a nearest-
neighbour search can quietly fail to terminate -- a user who has seen most of
the catalogue forces unbounded over-retrieval unless the growth is capped.

Skipped wholesale when faiss is absent, so the core suite still runs without
the ``retrieval`` extra installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("faiss", reason="Vector index requires the 'retrieval' extra")

from omnirank.core.exceptions import (
    ArtifactValidationError,
    VectorIndexError,
)
from omnirank.retrieval.faiss_index import (
    EMPTY_SLOT,
    FLAT_IP,
    FLAT_L2,
    FaissVectorIndex,
    brute_force_top_k,
    embedding_checksum,
)

VECTORS = 500
DIMENSION = 32


@pytest.fixture
def embeddings() -> np.ndarray:
    """A deterministic float32 matrix, as a model would export."""
    return np.random.default_rng(0).normal(size=(VECTORS, DIMENSION)).astype("float32")


@pytest.fixture
def queries() -> np.ndarray:
    return np.random.default_rng(1).normal(size=(10, DIMENSION)).astype("float32")


@pytest.fixture
def index(embeddings: np.ndarray) -> FaissVectorIndex:
    built = FaissVectorIndex(index_type=FLAT_IP)
    built.build(embeddings)
    return built


class TestBuild:
    def test_records_shape_and_checksum(self, index: FaissVectorIndex) -> None:
        assert index.num_vectors == VECTORS
        assert index.dimension == DIMENSION

    def test_rejects_a_one_dimensional_matrix(self) -> None:
        with pytest.raises(VectorIndexError):
            FaissVectorIndex().build(np.zeros(10, dtype="float32"))

    def test_rejects_nan_and_infinity(self) -> None:
        """FAISS accepts these and then returns silent nonsense."""
        broken = np.ones((5, 4), dtype="float32")
        broken[2, 1] = np.nan
        with pytest.raises(VectorIndexError):
            FaissVectorIndex().build(broken)
        broken[2, 1] = np.inf
        with pytest.raises(VectorIndexError):
            FaissVectorIndex().build(broken)

    def test_rejects_an_empty_matrix(self) -> None:
        with pytest.raises(VectorIndexError):
            FaissVectorIndex().build(np.zeros((0, 8), dtype="float32"))

    def test_rejects_an_unknown_index_type(self) -> None:
        with pytest.raises(VectorIndexError):
            FaissVectorIndex(index_type="magic-index")

    def test_rejects_an_oversampling_cap_below_the_factor(self) -> None:
        with pytest.raises(VectorIndexError):
            FaissVectorIndex(oversampling_factor=8, maximum_search_multiplier=2)


class TestExactness:
    """A flat index is exact, so it must agree with brute force completely."""

    def test_flat_inner_product_matches_brute_force_in_order(
        self, index: FaissVectorIndex, embeddings: np.ndarray, queries: np.ndarray
    ) -> None:
        items, _ = index.search(queries, 10)
        expected, _ = brute_force_top_k(embeddings, queries, 10)
        assert items == expected.tolist()

    def test_flat_inner_product_matches_brute_force_in_score(
        self, index: FaissVectorIndex, embeddings: np.ndarray, queries: np.ndarray
    ) -> None:
        _, scores = index.search(queries, 10)
        _, expected = brute_force_top_k(embeddings, queries, 10)
        assert np.allclose(np.array(scores), expected, atol=1e-4)

    def test_l2_index_matches_brute_force(
        self, embeddings: np.ndarray, queries: np.ndarray
    ) -> None:
        """A metric mismatch is the classic silent index bug."""
        built = FaissVectorIndex(index_type=FLAT_L2, metric="l2")
        built.build(embeddings, metric="l2")
        items, _ = built.search(queries, 10)
        expected, _ = brute_force_top_k(embeddings, queries, 10, metric="l2")
        assert items == expected.tolist()

    def test_a_vector_is_its_own_nearest_neighbour(
        self, index: FaissVectorIndex, embeddings: np.ndarray
    ) -> None:
        items, _ = index.search(embeddings[:5], 1)
        assert [row[0] for row in items] == [0, 1, 2, 3, 4]


class TestSearch:
    def test_rejects_a_dimension_mismatch(self, index: FaissVectorIndex) -> None:
        with pytest.raises(VectorIndexError):
            index.search(np.zeros((2, DIMENSION + 1), dtype="float32"), 5)

    def test_rejects_a_non_positive_k(self, index: FaissVectorIndex) -> None:
        with pytest.raises(VectorIndexError):
            index.search(np.zeros((2, DIMENSION), dtype="float32"), 0)

    def test_rejects_searching_before_building(self) -> None:
        with pytest.raises(VectorIndexError):
            FaissVectorIndex().search(np.zeros((1, 4), dtype="float32"), 5)

    def test_k_beyond_the_catalogue_pads_with_empty_slots(
        self, queries: np.ndarray, embeddings: np.ndarray
    ) -> None:
        small = FaissVectorIndex()
        small.build(embeddings[:3])
        items, _ = small.search(queries[:1], 10)
        assert len(items[0]) == 10
        assert items[0][3:] == [EMPTY_SLOT] * 7


class TestSearchExcluding:
    def test_excluded_items_never_appear(
        self, index: FaissVectorIndex, queries: np.ndarray, embeddings: np.ndarray
    ) -> None:
        top, _ = index.search(queries, 10)
        excluded = [set(row[:5]) for row in top]
        filtered, _ = index.search_excluding(queries, 5, excluded=excluded)
        for row, blocked in zip(filtered, excluded, strict=True):
            assert not set(row) & blocked

    def test_returns_the_next_best_items(
        self, index: FaissVectorIndex, queries: np.ndarray
    ) -> None:
        """Filtering must shift the list down, not perturb the ordering."""
        top, _ = index.search(queries, 10)
        excluded = [set(row[:3]) for row in top]
        filtered, _ = index.search_excluding(queries, 5, excluded=excluded)
        for row, full in zip(filtered, top, strict=True):
            assert row == full[3:8]

    def test_a_user_who_has_seen_almost_everything_terminates(
        self, index: FaissVectorIndex, queries: np.ndarray
    ) -> None:
        """The bounded-growth case: unbounded over-retrieval would hang here."""
        dense = [set(range(VECTORS - 5))]
        items, _ = index.search_excluding(queries[:1], 10, excluded=dense)
        assert len(items[0]) == 10
        assert not set(items[0]) & dense[0] - {EMPTY_SLOT}

    def test_excluding_everything_yields_empty_slots(
        self, index: FaissVectorIndex, queries: np.ndarray
    ) -> None:
        items, _ = index.search_excluding(queries[:1], 5, excluded=[set(range(VECTORS))])
        assert items[0] == [EMPTY_SLOT] * 5

    def test_rejects_misaligned_exclusions(
        self, index: FaissVectorIndex, queries: np.ndarray
    ) -> None:
        with pytest.raises(VectorIndexError):
            index.search_excluding(queries, 5, excluded=[set()])

    def test_empty_exclusions_match_a_plain_search(
        self, index: FaissVectorIndex, queries: np.ndarray
    ) -> None:
        plain, _ = index.search(queries, 5)
        filtered, _ = index.search_excluding(queries, 5, excluded=[set() for _ in queries])
        assert filtered == plain


class TestChecksum:
    def test_is_stable_for_identical_matrices(self, embeddings: np.ndarray) -> None:
        assert embedding_checksum(embeddings) == embedding_checksum(embeddings.copy())

    def test_changes_when_a_single_value_changes(self, embeddings: np.ndarray) -> None:
        altered = embeddings.copy()
        altered[0, 0] += np.float32(0.001)
        assert embedding_checksum(embeddings) != embedding_checksum(altered)


class TestPersistence:
    def test_round_trip_returns_identical_results(
        self, index: FaissVectorIndex, queries: np.ndarray, tmp_path
    ) -> None:
        index.save(tmp_path / "index")
        loaded = FaissVectorIndex.load(tmp_path / "index")
        assert loaded.search(queries, 10) == index.search(queries, 10)

    def test_round_trip_preserves_identity(self, index: FaissVectorIndex, tmp_path) -> None:
        index.attach_metadata(
            model_name="lightgcn",
            model_version="v1",
            item_mapping_checksum="map-abc",
            build_timestamp="2026-01-01T00:00:00Z",
        )
        index.save(tmp_path / "index")
        loaded = FaissVectorIndex.load(tmp_path / "index")
        assert loaded.metadata is not None
        assert loaded.metadata.model_name == "lightgcn"
        assert loaded.metadata.model_version == "v1"
        assert loaded.metadata.item_mapping_checksum == "map-abc"
        # The build-procedure version, carried so an incompatible index is
        # refused rather than silently queried.
        assert loaded.metadata.index_version == index.index_version
        assert loaded.metadata.embedding_checksum
        assert loaded.num_vectors == index.num_vectors

    def test_missing_files_are_reported_not_crashed(self, tmp_path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises((VectorIndexError, ArtifactValidationError)):
            FaissVectorIndex.load(tmp_path / "empty")

    def test_saving_before_building_is_refused(self, tmp_path) -> None:
        with pytest.raises(VectorIndexError):
            FaissVectorIndex().save(tmp_path / "index")


class TestCompatibility:
    """ADR-006: an index paired with the wrong model returns confident nonsense."""

    @pytest.fixture
    def described(self, index: FaissVectorIndex) -> FaissVectorIndex:
        index.attach_metadata(
            model_name="lightgcn",
            model_version="v1",
            item_mapping_checksum="map-abc",
            build_timestamp="2026-01-01T00:00:00Z",
        )
        return index

    def test_matching_identity_passes(self, described: FaissVectorIndex) -> None:
        described.require_compatible(
            model_name="lightgcn", model_version="v1", item_mapping_checksum="map-abc"
        )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("model_name", "sasrec"),
            ("model_version", "v2"),
            ("item_mapping_checksum", "map-xyz"),
        ],
    )
    def test_any_mismatch_is_refused(
        self, described: FaissVectorIndex, field: str, value: str
    ) -> None:
        identity = {
            "model_name": "lightgcn",
            "model_version": "v1",
            "item_mapping_checksum": "map-abc",
        }
        identity[field] = value
        with pytest.raises(ArtifactValidationError):
            described.require_compatible(**identity)

    def test_an_index_without_metadata_cannot_be_checked(self, index: FaissVectorIndex) -> None:
        with pytest.raises(ArtifactValidationError):
            index.require_compatible(
                model_name="lightgcn", model_version="v1", item_mapping_checksum="map-abc"
            )
