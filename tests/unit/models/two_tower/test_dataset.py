"""Two-tower training examples and collation.

The leakage assertions carry the weight here. A history that contains its own
target trains a model to copy the answer out of its input: the loss falls
faster, every offline metric improves, and nothing in the training log says so.
Phase 2 already guarantees the property, and these tests assert it again at the
consumer boundary so a future change to sequence construction cannot
reintroduce it silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omnirank.core.exceptions import DataError

from .conftest import COLD_ITEM, IMAGE_DIM, ITEMS, TEXT_DIM, USERS, write_feature_store


class TestConstruction:
    def test_reports_examples_and_warm_items(self, dataset) -> None:
        assert len(dataset) > 0
        assert dataset.warm_mask.sum() > 0

    def test_padding_id_is_one_past_the_catalogue(self, dataset) -> None:
        """Reusing a valid id would train the model on someone else's content."""
        assert dataset.padding_id == ITEMS
        assert dataset.padding_id not in range(ITEMS)

    def test_an_item_never_interacted_with_is_cold(self, dataset) -> None:
        assert not dataset.warm_mask[COLD_ITEM]

    def test_rejects_missing_columns(self, store) -> None:
        from omnirank.models.two_tower import TwoTowerTrainingDataset

        with pytest.raises(DataError):
            TwoTowerTrainingDataset(
                pd.DataFrame({"internal_user_id": [0]}),
                store,
                num_items=ITEMS,
                num_users=USERS,
            )

    def test_rejects_an_empty_frame(self, store, sequences) -> None:
        from omnirank.models.two_tower import TwoTowerTrainingDataset

        with pytest.raises(DataError):
            TwoTowerTrainingDataset(sequences.iloc[:0], store, num_items=ITEMS, num_users=USERS)

    def test_rejects_a_store_covering_a_different_catalogue(self, store, sequences) -> None:
        """Vectors describing other items is a wrong-answer bug, not a crash."""
        from omnirank.models.two_tower import TwoTowerTrainingDataset

        with pytest.raises(DataError):
            TwoTowerTrainingDataset(sequences, store, num_items=ITEMS + 5, num_users=USERS)

    def test_rejects_an_item_id_outside_the_catalogue(self, store, sequences) -> None:
        from omnirank.models.two_tower import TwoTowerTrainingDataset

        broken = sequences.copy()
        broken.loc[0, "target_item"] = ITEMS + 99
        with pytest.raises(DataError):
            TwoTowerTrainingDataset(broken, store, num_items=ITEMS, num_users=USERS)

    def test_rejects_a_target_hidden_inside_its_own_history(self, store, sequences) -> None:
        """The leakage that improves every metric while destroying the model."""
        from omnirank.models.two_tower import TwoTowerTrainingDataset

        broken = sequences.copy()
        target = int(broken.loc[0, "target_item"])
        broken.at[0, "item_sequence"] = [*broken.loc[0, "item_sequence"], target]
        with pytest.raises(DataError, match="inside its own input history"):
            TwoTowerTrainingDataset(broken, store, num_items=ITEMS, num_users=USERS)


class TestHistoryTruncation:
    def test_truncates_to_the_configured_maximum(self, dataset) -> None:
        assert all(dataset.history_for(index).size <= 5 for index in range(len(dataset)))

    def test_drops_the_oldest_first(self, store, item_tags) -> None:
        """Dropping recent history would defeat the point of using history."""
        from omnirank.models.two_tower import TwoTowerTrainingDataset

        frame = pd.DataFrame(
            [(0, [1, 2, 3, 4, 5, 6, 7], 8)],
            columns=["internal_user_id", "item_sequence", "target_item"],
        )
        built = TwoTowerTrainingDataset(
            frame,
            store,
            num_items=ITEMS,
            num_users=USERS,
            maximum_history_length=3,
            item_tags=item_tags,
        )
        assert built.history_for(0).tolist() == [5, 6, 7]

    def test_rejects_a_non_positive_maximum(self, store, sequences) -> None:
        from omnirank.models.two_tower import TwoTowerTrainingDataset

        with pytest.raises(DataError):
            TwoTowerTrainingDataset(
                sequences, store, num_items=ITEMS, num_users=USERS, maximum_history_length=0
            )


class TestCollation:
    @pytest.fixture
    def batch(self, dataset):
        return dataset.collate(np.arange(8))

    def test_batch_validates(self, batch) -> None:
        batch.validate(text_dim=TEXT_DIM, image_dim=IMAGE_DIM, max_history=5)

    def test_shapes_match_the_feature_manifest(self, batch) -> None:
        assert batch.history_text_features.shape == (8, 5, TEXT_DIM)
        assert batch.history_image_features.shape == (8, 5, IMAGE_DIM)
        assert batch.positive_text_features.shape == (8, TEXT_DIM)

    def test_histories_are_right_aligned(self, batch) -> None:
        """The newest item sits in the final column, so recency indexes from the end."""
        assert (batch.history_item_ids[batch.history_padding_mask] == ITEMS).all()
        for row, length in enumerate(batch.history_lengths):
            if length:
                assert not batch.history_padding_mask[row, -1]

    def test_padded_positions_claim_no_modality(self, batch) -> None:
        """A padded slot must not look like an item that happens to have text."""
        assert not batch.history_text_available[batch.history_padding_mask].any()
        assert not batch.history_image_available[batch.history_padding_mask].any()

    def test_padded_feature_rows_are_zeroed(self, batch) -> None:
        padded = batch.history_text_features[batch.history_padding_mask]
        assert padded.size == 0 or np.all(padded == 0.0)

    def test_target_is_never_in_its_own_history(self, batch) -> None:
        for target, history in zip(batch.positive_item_ids, batch.history_item_ids, strict=True):
            assert target not in history

    def test_collation_is_deterministic(self, dataset) -> None:
        first = dataset.collate(np.arange(8))
        second = dataset.collate(np.arange(8))
        assert np.array_equal(first.history_item_ids, second.history_item_ids)
        assert np.array_equal(first.positive_text_features, second.positive_text_features)

    def test_rejects_an_empty_batch(self, dataset) -> None:
        with pytest.raises(DataError):
            dataset.collate(np.empty(0, dtype="int64"))

    def test_validate_rejects_wrong_dimensions(self, batch) -> None:
        with pytest.raises(DataError):
            batch.validate(text_dim=TEXT_DIM + 1, image_dim=IMAGE_DIM, max_history=5)

    def test_validate_rejects_non_finite_features(self, batch) -> None:
        """A NaN here kills every parameter in one backward pass."""
        batch.positive_text_features[0, 0] = np.nan
        with pytest.raises(DataError, match="non-finite"):
            batch.validate(text_dim=TEXT_DIM, image_dim=IMAGE_DIM, max_history=5)


class TestMissingModalities:
    def test_an_item_without_text_is_flagged(self, dataset) -> None:
        from .conftest import NO_TEXT_ITEM

        batch = dataset.collate(np.arange(len(dataset)))
        rows = batch.positive_item_ids == NO_TEXT_ITEM
        if rows.any():
            assert not batch.positive_text_available[rows].any()

    def test_a_store_with_no_modalities_still_collates(
        self, tmp_path, sequences, item_tags
    ) -> None:
        """Absence must degrade, never crash."""
        from omnirank.models.two_tower import TwoTowerTrainingDataset

        empty = write_feature_store(tmp_path / "none")
        # Force both masks off to simulate a catalogue with no content at all.
        empty._masks["text"][:] = False
        empty._masks["image"][:] = False
        built = TwoTowerTrainingDataset(
            sequences,
            empty,
            num_items=ITEMS,
            num_users=USERS,
            maximum_history_length=5,
            item_tags=item_tags,
        )
        batch = built.collate(np.arange(4))
        assert not batch.positive_text_available.any()
        assert np.all(batch.positive_text_features == 0.0)


class TestBatching:
    def test_unshuffled_batching_is_in_order(self, dataset) -> None:
        blocks = list(dataset.batches(10))
        assert blocks[0].tolist() == list(range(10))

    def test_seeded_shuffling_is_reproducible(self, dataset) -> None:
        first = list(dataset.batches(10, rng=np.random.default_rng(3)))
        second = list(dataset.batches(10, rng=np.random.default_rng(3)))
        assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))

    def test_different_seeds_differ(self, dataset) -> None:
        first = next(iter(dataset.batches(10, rng=np.random.default_rng(3))))
        second = next(iter(dataset.batches(10, rng=np.random.default_rng(4))))
        assert not np.array_equal(first, second)

    def test_covers_every_example_exactly_once(self, dataset) -> None:
        seen = np.concatenate(list(dataset.batches(7, rng=np.random.default_rng(0))))
        assert sorted(seen.tolist()) == list(range(len(dataset)))

    def test_rejects_a_non_positive_batch_size(self, dataset) -> None:
        with pytest.raises(DataError):
            list(dataset.batches(0))


class TestFalseNegativeSupport:
    def test_positives_include_history_and_target(self, dataset) -> None:
        """A user's history is a set of known positives, not candidate negatives."""
        positives = dataset.positives_by_row(np.array([0]))
        history = set(dataset.history_for(0).tolist())
        assert history <= positives[0]
        assert dataset[0]["positive_item_id"] in positives[0]

    def test_one_set_per_row(self, dataset) -> None:
        assert len(dataset.positives_by_row(np.arange(5))) == 5
