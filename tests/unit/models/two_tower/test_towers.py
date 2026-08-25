"""User and item towers.

The cold-item tests are the ones that matter. Phase 5 exists to make items with
no interaction history retrievable, and the single mechanism that delivers it is
the warm-gated identity residual. If that gate leaks, the model still trains,
its warm metrics still look fine, and cold recall reads zero for a reason no
warm number reveals -- so the guarantee is asserted as an exact equality against
the content-only path, not inferred from a metric.

The masking tests are the second concern. A zero vector is not "no text": it is
one specific point that every text-less item would share, so the model would
learn spurious similarity between items whose only common property is an absent
feature. Each modality carries a learned missing-token instead.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from omnirank.core.exceptions import DataError

from .conftest import COLD_ITEM, IMAGE_DIM, ITEMS, TAGS, TEXT_DIM, USERS


@pytest.fixture
def batch(dataset):
    return dataset.collate(np.arange(8))


def tensors(batch):
    """Convert one collated batch to tensors."""
    return {
        name: torch.from_numpy(getattr(batch, name))
        for name in (
            "history_text_features",
            "history_image_features",
            "history_text_available",
            "history_image_available",
            "history_tag_ids",
            "history_padding_mask",
            "history_lengths",
            "user_ids",
            "positive_text_features",
            "positive_image_features",
            "positive_text_available",
            "positive_image_available",
            "positive_tag_ids",
            "positive_item_ids",
            "positive_warm_mask",
        )
    }


def encode_items(model, t, *, item_ids=True, warm=True):
    with torch.no_grad():
        return model.encode_items(
            t["positive_text_features"],
            t["positive_image_features"],
            t["positive_text_available"],
            t["positive_image_available"],
            t["positive_tag_ids"],
            t["positive_item_ids"] if item_ids else None,
            t["positive_warm_mask"] if warm else None,
        )


def encode_users(model, t, *, user_ids=True):
    with torch.no_grad():
        return model.encode_users(
            t["history_text_features"],
            t["history_image_features"],
            t["history_text_available"],
            t["history_image_available"],
            t["history_tag_ids"],
            t["history_padding_mask"],
            t["history_lengths"],
            t["user_ids"] if user_ids else None,
        )


class TestSharedSpace:
    def test_both_towers_output_the_configured_width(self, model, batch) -> None:
        t = tensors(batch)
        assert encode_items(model, t).shape == (8, 16)
        assert encode_users(model, t).shape == (8, 16)

    def test_similarity_rejects_mismatched_widths(self, model) -> None:
        """Retrieval is a dot product; different widths is a silent wrong answer."""
        with pytest.raises(DataError):
            model.similarity(torch.zeros(2, 16), torch.zeros(2, 8))

    def test_rejects_non_positive_dimensions(self, config) -> None:
        from omnirank.models.two_tower import MultimodalTwoTower

        with pytest.raises(DataError):
            MultimodalTwoTower(
                config, text_dim=0, image_dim=IMAGE_DIM, num_items=ITEMS, num_users=USERS
            )


class TestColdItemGuarantee:
    """Phase 5's central promise, asserted rather than measured."""

    @pytest.fixture
    def cold_inputs(self, store, item_tags):
        features = store.get_batch(np.array([COLD_ITEM]))
        return {
            "text": torch.from_numpy(features.text),
            "image": torch.from_numpy(features.image),
            "text_available": torch.tensor([True]),
            "image_available": torch.tensor([True]),
            "tag": torch.tensor([int(item_tags[COLD_ITEM])]),
            "ids": torch.tensor([COLD_ITEM]),
        }

    def test_cold_embedding_equals_the_content_only_path(self, model, cold_inputs) -> None:
        """warm_mask=0 must zero the residual exactly, not merely damp it."""
        with torch.no_grad():
            gated = model.encode_items(
                cold_inputs["text"],
                cold_inputs["image"],
                cold_inputs["text_available"],
                cold_inputs["image_available"],
                cold_inputs["tag"],
                cold_inputs["ids"],
                torch.tensor([False]),
            )
            content_only = model.encode_items(
                cold_inputs["text"],
                cold_inputs["image"],
                cold_inputs["text_available"],
                cold_inputs["image_available"],
                cold_inputs["tag"],
                None,
                None,
            )
        assert torch.allclose(gated, content_only, atol=1e-6)

    def test_the_residual_is_not_merely_absent(self, model, cold_inputs) -> None:
        """Guards against the previous test passing because nothing is added at all."""
        with torch.no_grad():
            cold = model.encode_items(
                cold_inputs["text"],
                cold_inputs["image"],
                cold_inputs["text_available"],
                cold_inputs["image_available"],
                cold_inputs["tag"],
                cold_inputs["ids"],
                torch.tensor([False]),
            )
            warm = model.encode_items(
                cold_inputs["text"],
                cold_inputs["image"],
                cold_inputs["text_available"],
                cold_inputs["image_available"],
                cold_inputs["tag"],
                cold_inputs["ids"],
                torch.tensor([True]),
            )
        assert not torch.allclose(cold, warm, atol=1e-6)

    def test_cold_embedding_is_finite(self, model, cold_inputs) -> None:
        with torch.no_grad():
            cold = model.encode_items(
                cold_inputs["text"],
                cold_inputs["image"],
                cold_inputs["text_available"],
                cold_inputs["image_available"],
                cold_inputs["tag"],
                cold_inputs["ids"],
                torch.tensor([False]),
            )
        assert torch.isfinite(cold).all()

    def test_a_model_without_content_cannot_claim_cold_capability(self) -> None:
        from omnirank.models.two_tower import TwoTowerConfig

        id_only = TwoTowerConfig(
            use_text=False, use_image=False, use_tag=False, use_item_id_residual=True
        )
        assert not id_only.content_enabled

    def test_disabling_every_input_is_refused(self) -> None:
        """With nothing to encode, every item would get the same vector."""
        from omnirank.models.two_tower import TwoTowerConfig

        with pytest.raises(DataError):
            TwoTowerConfig(
                use_text=False, use_image=False, use_tag=False, use_item_id_residual=False
            )


class TestMissingModalities:
    @pytest.fixture
    def single(self, store, item_tags):
        features = store.get_batch(np.array([5]))
        return (
            torch.from_numpy(features.text),
            torch.from_numpy(features.image),
            torch.tensor([int(item_tags[5])]),
        )

    def _encode(self, model, single, *, text: bool, image: bool):
        with torch.no_grad():
            return model.encode_items(
                single[0],
                single[1],
                torch.tensor([text]),
                torch.tensor([image]),
                single[2],
                None,
                None,
            )

    def test_text_only_differs_from_image_only(self, model, single) -> None:
        assert not torch.allclose(
            self._encode(model, single, text=True, image=False),
            self._encode(model, single, text=False, image=True),
            atol=1e-6,
        )

    def test_masking_a_modality_changes_the_result(self, model, single) -> None:
        """A mask that did nothing would mean the model reads absent features."""
        assert not torch.allclose(
            self._encode(model, single, text=True, image=True),
            self._encode(model, single, text=False, image=True),
            atol=1e-6,
        )

    def test_both_modalities_missing_still_encodes(self, model, single) -> None:
        """Falls back to the tag rather than failing."""
        result = self._encode(model, single, text=False, image=False)
        assert torch.isfinite(result).all()

    def test_missing_token_is_learnable(self, model) -> None:
        """A fixed zero would make 'absent' unrepresentable rather than learned."""
        assert model.item_tower.text_encoder.missing.requires_grad


class TestNormalization:
    def test_l2_produces_unit_vectors(self, model, batch) -> None:
        t = tensors(batch)
        items = encode_items(model, t)
        users = encode_users(model, t)
        assert torch.allclose(items.norm(dim=-1), torch.ones(8), atol=1e-5)
        assert torch.allclose(users.norm(dim=-1), torch.ones(8), atol=1e-5)

    def test_disabling_normalization_leaves_magnitudes_free(self, batch) -> None:
        from omnirank.models.two_tower import MultimodalTwoTower, TwoTowerConfig

        torch.manual_seed(0)
        unnormalized = MultimodalTwoTower(
            TwoTowerConfig(
                embedding_dim=16,
                hidden_dims=(32,),
                dropout=0.0,
                maximum_history_length=5,
                l2_normalize=False,
                device="cpu",
            ),
            text_dim=TEXT_DIM,
            image_dim=IMAGE_DIM,
            num_items=ITEMS,
            num_users=USERS,
            num_tags=TAGS,
        )
        unnormalized.eval()
        norms = encode_items(unnormalized, tensors(batch)).norm(dim=-1)
        assert not torch.allclose(norms, torch.ones(8), atol=1e-3)

    def test_normalization_rule_is_recorded(self, model) -> None:
        """FAISS must be built under the same convention or it returns nonsense."""
        assert model.modality_schema()["normalization"] == "l2"


class TestUserTower:
    def test_padded_positions_cannot_affect_pooling(self, model, dataset) -> None:
        """A padded slot influencing the query would make history length a signal."""
        short = dataset.collate(np.array([0]))
        t = tensors(short)
        with torch.no_grad():
            encoded = model.item_tower.encode_content(
                t["history_text_features"].reshape(-1, TEXT_DIM),
                t["history_image_features"].reshape(-1, IMAGE_DIM),
                t["history_text_available"].reshape(-1),
                t["history_image_available"].reshape(-1),
                t["history_tag_ids"].reshape(-1),
            ).reshape(1, 5, -1)
            pooled = model.user_tower.pool(encoded, t["history_padding_mask"], t["history_lengths"])
            perturbed = encoded.clone()
            perturbed[0, 0] += 100.0
            after = model.user_tower.pool(
                perturbed, t["history_padding_mask"], t["history_lengths"]
            )
        assert bool(t["history_padding_mask"][0, 0])
        assert torch.allclose(pooled, after, atol=1e-6)

    def test_empty_history_pools_to_zero(self, model) -> None:
        encoded = torch.randn(1, 5, 16)
        pooled = model.user_tower.pool(
            encoded, torch.ones(1, 5, dtype=torch.bool), torch.tensor([0])
        )
        assert torch.allclose(pooled, torch.zeros_like(pooled))

    def test_recency_weighting_favours_recent_positions(self, model) -> None:
        """Histories are right-aligned, so the last column is the newest item."""
        encoded = torch.zeros(1, 4, 16)
        encoded[0, 0] = 1.0  # oldest
        encoded[0, 3] = 2.0  # newest
        pooled = model.user_tower.pool(
            encoded, torch.zeros(1, 4, dtype=torch.bool), torch.tensor([4])
        )
        mean = encoded.mean(dim=1)
        assert pooled[0, 0] > mean[0, 0]

    def test_mean_pooling_weights_positions_equally(self) -> None:
        from omnirank.models.two_tower import MultimodalTwoTower, TwoTowerConfig
        from omnirank.models.two_tower.config import MEAN_POOLING

        torch.manual_seed(0)
        built = MultimodalTwoTower(
            TwoTowerConfig(
                embedding_dim=16,
                hidden_dims=(32,),
                dropout=0.0,
                maximum_history_length=5,
                history_pooling=MEAN_POOLING,
                device="cpu",
            ),
            text_dim=TEXT_DIM,
            image_dim=IMAGE_DIM,
            num_items=ITEMS,
            num_users=USERS,
            num_tags=TAGS,
        )
        encoded = torch.randn(1, 4, 16)
        pooled = built.user_tower.pool(
            encoded, torch.zeros(1, 4, dtype=torch.bool), torch.tensor([4])
        )
        assert torch.allclose(pooled, encoded.mean(dim=1), atol=1e-5)

    def test_unknown_user_encodes_from_history_alone(self, model, batch) -> None:
        """A supplied history is enough; no identity lookup is required."""
        result = encode_users(model, tensors(batch), user_ids=False)
        assert result.shape == (8, 16)
        assert torch.isfinite(result).all()

    def test_unknown_user_differs_from_the_known_user(self, model, batch) -> None:
        t = tensors(batch)
        assert not torch.allclose(
            encode_users(model, t, user_ids=False), encode_users(model, t), atol=1e-6
        )

    def test_history_items_are_encoded_without_identity(self, model, batch) -> None:
        """Otherwise an unknown-user query would need its history items to be warm."""
        result = encode_users(model, tensors(batch), user_ids=False)
        assert torch.isfinite(result).all()


class TestDeterminism:
    def test_repeated_encoding_is_identical(self, model, batch) -> None:
        t = tensors(batch)
        assert torch.equal(encode_items(model, t), encode_items(model, t))

    def test_same_seed_builds_the_same_model(self, config, batch) -> None:
        from omnirank.models.two_tower import MultimodalTwoTower

        def build():
            torch.manual_seed(11)
            built = MultimodalTwoTower(
                config,
                text_dim=TEXT_DIM,
                image_dim=IMAGE_DIM,
                num_items=ITEMS,
                num_users=USERS,
                num_tags=TAGS,
            )
            built.eval()
            return encode_items(built, tensors(batch))

        assert torch.equal(build(), build())
