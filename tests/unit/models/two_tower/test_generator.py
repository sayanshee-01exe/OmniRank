"""The two-tower retriever's `CandidateGenerator` surface.

Two properties carry most of this file.

**Cold items must be reachable.** Every other retriever's catalogue is "items
with a fitting interaction"; this one adds items that have content but no
history. A catalogue that quietly contained none would still answer every query,
so its composition is asserted rather than assumed.

**No fallback lives in this class.** An unknown user with no history returns an
empty list, not popularity. If it substituted another model's output, "the
two-tower retrieved it" would be untrue for an unknown fraction of requests and
every downstream contribution metric would inherit that.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from omnirank.core.exceptions import ArtifactValidationError, DataError
from omnirank.models.two_tower import TwoTowerRetriever, build_catalogue

from .conftest import COLD_ITEM, ITEMS, USERS


@pytest.fixture
def histories(dataset) -> dict[int, list[int]]:
    """One history per user, taken from the fixture sequences."""
    built: dict[int, list[int]] = {}
    for index in range(len(dataset)):
        example = dataset[index]
        user = example["internal_user_id"]
        combined = [*example["history_item_ids"].tolist(), example["positive_item_id"]]
        if user not in built or len(combined) > len(built[user]):
            built[user] = combined
    return built


@pytest.fixture
def retriever(model, store, dataset, histories, item_tags):
    return TwoTowerRetriever(
        model,
        store,
        internal_to_external_item={index: f"i{index}" for index in range(ITEMS)},
        external_to_internal_user={f"u{user}": user for user in range(USERS)},
        item_tags=item_tags,
        histories=histories,
        warm_items=dataset.warm_mask,
        device="cpu",
        mapping_checksum="fixture-mapping",
    )


class TestCatalogue:
    def test_cold_items_are_in_the_catalogue(self, retriever) -> None:
        """The whole point: an item with no interactions is still retrievable."""
        assert COLD_ITEM in retriever.fit_item_catalogue
        assert COLD_ITEM in retriever.cold_item_catalogue
        assert retriever.catalogue.cold_count > 0

    def test_warm_and_cold_partition_the_catalogue(self, retriever) -> None:
        catalogue = retriever.catalogue
        assert catalogue.warm_count + catalogue.cold_count == len(catalogue)

    def test_ordering_is_deterministic(self, retriever) -> None:
        """Embeddings, index and table are positional; reordering misaligns all three."""
        ids = retriever.catalogue.internal_ids
        assert ids.tolist() == sorted(ids.tolist())

    def test_checksum_is_stable_and_content_sensitive(self, retriever) -> None:
        first = retriever.catalogue.checksum()
        assert first == retriever.catalogue.checksum()
        altered = retriever.catalogue.items.copy()
        altered.loc[0, "warm_item_flag"] = not altered.loc[0, "warm_item_flag"]
        from omnirank.models.two_tower import RetrievalCatalogue

        other = RetrievalCatalogue(altered, 0, 0, 0, ())
        assert other.checksum() != first

    def test_items_with_no_content_and_no_warmth_are_excluded(self) -> None:
        """Nothing can represent them, and an arbitrary vector would rank somewhere."""
        catalogue = build_catalogue(
            warm_items=np.array([True, False, False]),
            text_available=np.array([False, True, False]),
            image_available=np.array([False, False, False]),
            internal_to_external={0: "a", 1: "b", 2: "c"},
        )
        assert len(catalogue) == 2
        assert catalogue.excluded_count == 1
        assert 2 not in catalogue.internal_ids.tolist()

    def test_refuses_a_catalogue_where_nothing_is_representable(self) -> None:
        with pytest.raises(DataError):
            build_catalogue(
                warm_items=np.zeros(3, dtype=bool),
                text_available=np.zeros(3, dtype=bool),
                image_available=np.zeros(3, dtype=bool),
                internal_to_external={},
            )

    def test_mismatched_masks_are_refused(self) -> None:
        with pytest.raises(DataError):
            build_catalogue(
                warm_items=np.zeros(3, dtype=bool),
                text_available=np.zeros(4, dtype=bool),
                image_available=np.zeros(3, dtype=bool),
                internal_to_external={},
            )


class TestEmbeddingExport:
    def test_shape_matches_the_catalogue(self, retriever) -> None:
        embeddings = retriever.export_item_embeddings(batch_size=8)
        assert embeddings.shape == (len(retriever.catalogue), 16)

    def test_output_is_float32(self, retriever) -> None:
        assert retriever.export_item_embeddings(batch_size=8).dtype == np.float32

    def test_values_are_finite(self, retriever) -> None:
        assert np.isfinite(retriever.export_item_embeddings(batch_size=8)).all()

    def test_batching_does_not_change_the_result(self, retriever) -> None:
        """Memory-bounded export must be an implementation detail, not a variable."""
        small = retriever.export_item_embeddings(batch_size=4)
        large = retriever.export_item_embeddings(batch_size=1024)
        assert np.allclose(small, large, atol=1e-6)

    def test_cold_rows_use_content_only(self, retriever, store, item_tags) -> None:
        """The row written for a cold item must equal its content encoding."""
        embeddings = retriever.export_item_embeddings(batch_size=8)
        row = retriever._row_of[COLD_ITEM]
        features = store.get_batch(np.array([COLD_ITEM]))
        with torch.no_grad():
            content = retriever.model.encode_items(
                torch.from_numpy(features.text),
                torch.from_numpy(features.image),
                torch.from_numpy(features.text_mask),
                torch.from_numpy(features.image_mask),
                torch.tensor([int(item_tags[COLD_ITEM])]),
                None,
                None,
            )
        assert np.allclose(embeddings[row], content.numpy()[0], atol=1e-6)

    def test_rejects_a_non_positive_batch_size(self, retriever) -> None:
        with pytest.raises(DataError):
            retriever.export_item_embeddings(batch_size=0)


class TestRecommendation:
    def test_known_user_receives_candidates(self, retriever) -> None:
        candidates = retriever.recommend("u0", 5)
        assert len(candidates) == 5
        assert all(candidate.item_id.startswith("i") for candidate in candidates)

    def test_provenance_is_two_tower(self, retriever) -> None:
        candidate = retriever.recommend("u0", 1)[0]
        assert candidate.sources == ("two_tower",)
        assert candidate.source_scores["two_tower"] == candidate.score

    def test_no_duplicates(self, retriever) -> None:
        items = [candidate.item_id for candidate in retriever.recommend("u0", 20)]
        assert len(items) == len(set(items))

    def test_seen_items_are_excluded(self, retriever, histories) -> None:
        recommended = retriever.recommend_batch([f"u{u}" for u in range(USERS)], 10)
        for user, items in recommended.items():
            internal = int(user[1:])
            seen = {f"i{item}" for item in histories.get(internal, [])}
            assert not set(items) & seen

    def test_seen_filtering_can_be_disabled(self, retriever, histories) -> None:
        """Asserted as the property, not as "the two lists differ".

        On a small catalogue a user's seen items may simply not rank in the top
        k, so filtered and unfiltered can legitimately coincide. What must
        always hold is that filtering never admits a seen item, and that
        disabling it still returns a full, duplicate-free list.
        """
        seen = {f"i{item}" for item in histories[0]}
        filtered = retriever.recommend_batch(["u0"], 10, filter_seen=True)["u0"]
        unfiltered = retriever.recommend_batch(["u0"], 10, filter_seen=False)["u0"]
        assert not set(filtered) & seen
        assert len(unfiltered) == len(filtered) == 10
        assert len(set(unfiltered)) == 10

    def test_batch_matches_single_user(self, retriever) -> None:
        batched = retriever.recommend_batch(["u0"], 5)["u0"]
        single = [candidate.item_id for candidate in retriever.recommend("u0", 5)]
        assert batched == single

    def test_returned_items_are_in_the_catalogue(self, retriever) -> None:
        catalogue = {f"i{item}" for item in retriever.fit_item_catalogue}
        assert set(retriever.recommend_batch(["u0"], 10)["u0"]) <= catalogue

    def test_rejects_a_non_positive_k(self, retriever) -> None:
        with pytest.raises(DataError):
            retriever.recommend("u0", 0)

    def test_k_beyond_the_catalogue_returns_what_exists(self, retriever) -> None:
        assert len(retriever.recommend("u0", 10_000)) <= len(retriever.catalogue)


class TestUnknownUsers:
    def test_unknown_user_without_history_returns_nothing(self, retriever) -> None:
        """Fallback is the orchestrator's job, not this class's."""
        assert retriever.recommend("stranger", 10) == []

    def test_unknown_user_with_supplied_history_is_served(self, retriever) -> None:
        candidates = retriever.recommend("stranger", 5, {"history": ["i1", "i2", "i3"]})
        assert len(candidates) == 5

    def test_supplied_history_of_unknown_items_yields_nothing(self, retriever) -> None:
        assert retriever.recommend("stranger", 5, {"history": ["nope", "also-nope"]}) == []

    def test_batch_returns_empty_for_unknown_users(self, retriever) -> None:
        assert retriever.recommend_batch(["stranger"], 5)["stranger"] == []


class TestScoring:
    def test_unknown_item_scores_zero(self, retriever) -> None:
        assert retriever.score("u0", ["not-an-item"]) == [0.0]

    def test_unknown_user_scores_zero(self, retriever) -> None:
        assert retriever.score("stranger", ["i1", "i2"]) == [0.0, 0.0]

    def test_known_items_receive_real_scores(self, retriever) -> None:
        scores = retriever.score("u0", ["i1", "i2", "i3"])
        assert len(scores) == 3
        assert any(score != 0.0 for score in scores)

    def test_cold_items_are_scoreable(self, retriever) -> None:
        assert retriever.score("u0", [f"i{COLD_ITEM}"]) != [0.0]


class TestIdentity:
    def test_matching_mapping_passes(self, retriever) -> None:
        retriever.require_mapping("fixture-mapping")

    def test_a_different_mapping_is_refused(self, retriever) -> None:
        with pytest.raises(ArtifactValidationError):
            retriever.require_mapping("another-mapping")

    def test_metadata_records_catalogue_composition(self, retriever) -> None:
        metadata = retriever.metadata()
        assert metadata["model"] == "two_tower"
        assert metadata["cold_items"] > 0
        assert metadata["catalogue_checksum"]
        assert metadata["normalization"] == "l2"

    def test_fitting_is_refused(self, retriever) -> None:
        """Training belongs to TwoTowerTrainer; duplicating it here would drift."""
        with pytest.raises(DataError):
            retriever.fit(None)

    def test_rejects_an_inverted_search_bound(
        self, model, store, dataset, histories, item_tags
    ) -> None:
        with pytest.raises(DataError):
            TwoTowerRetriever(
                model,
                store,
                internal_to_external_item={index: f"i{index}" for index in range(ITEMS)},
                external_to_internal_user={},
                item_tags=item_tags,
                histories=histories,
                warm_items=dataset.warm_mask,
                oversampling_factor=8,
                maximum_search_multiplier=2,
            )


class TestPersistence:
    def test_round_trip_preserves_recommendations(self, retriever, store, tmp_path) -> None:
        users = [f"u{u}" for u in range(USERS)]
        before = retriever.recommend_batch(users, 10)
        retriever.save(tmp_path / "retriever")
        loaded = TwoTowerRetriever.load(tmp_path / "retriever", store=store, device="cpu")
        assert loaded.recommend_batch(users, 10) == before

    def test_round_trip_preserves_the_catalogue(self, retriever, store, tmp_path) -> None:
        retriever.save(tmp_path / "retriever")
        loaded = TwoTowerRetriever.load(tmp_path / "retriever", store=store, device="cpu")
        assert loaded.catalogue.checksum() == retriever.catalogue.checksum()
        assert loaded.cold_item_catalogue == retriever.cold_item_catalogue

    def test_loading_without_a_store_is_refused(self, retriever, tmp_path) -> None:
        """Item vectors come from the store, not the checkpoint."""
        retriever.save(tmp_path / "retriever")
        with pytest.raises(DataError):
            TwoTowerRetriever.load(tmp_path / "retriever", device="cpu")

    def test_missing_retrieval_context_is_reported(self, retriever, store, tmp_path) -> None:
        retriever.save(tmp_path / "retriever")
        (tmp_path / "retriever" / "retrieval_context.npz").unlink()
        with pytest.raises(ArtifactValidationError):
            TwoTowerRetriever.load(tmp_path / "retriever", store=store, device="cpu")
