"""End-to-end two-tower workflow over a deterministic synthetic fixture.

    synthetic features -> feature store -> training examples -> train
                       -> encode users -> encode warm items
                       -> encode a COLD item from content alone
                       -> save -> load -> verify identical

The cold item is the point. It appears in no history and is no user's target,
so a collaborative model has nothing to represent it with. This test asserts
that the two-tower model encodes it anyway, that the encoding uses no identity
residual, and that the property survives a save/load round trip -- because a
cold-start guarantee that holds only in the live model is not a guarantee.

Offline, CPU-only, seconds. No PixelRec download, no pretrained weights, no GPU,
no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="Two-tower requires the 'retrieval' extra")

from omnirank.features.multimodal_store import MultimodalFeatureStore  # noqa: E402
from omnirank.models.two_tower import (  # noqa: E402
    MultimodalTwoTower,
    TwoTowerConfig,
    TwoTowerTrainer,
    TwoTowerTrainingDataset,
    build_metadata,
    load,
    save,
)

pytestmark = pytest.mark.integration

ITEMS = 36
USERS = 24
TEXT_DIM = 12
IMAGE_DIM = 8
TAGS = 3
#: Never interacted with by anyone. Collaborative models cannot reach it.
COLD_ITEM = ITEMS - 1


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, object]:
    """Synthetic content, sequences and a real on-disk feature store."""
    rng = np.random.default_rng(7)
    features = tmp_path / "features"
    features.mkdir(parents=True)

    # Content is clustered by tag, so items in a block genuinely resemble each
    # other. Without that the model has nothing learnable and "loss decreased"
    # would only mean it memorised ids.
    tags = np.arange(ITEMS) % TAGS
    centres = rng.normal(size=(TAGS, TEXT_DIM)).astype("float32") * 3.0
    text = (centres[tags] + rng.normal(size=(ITEMS, TEXT_DIM)) * 0.3).astype("float32")
    image_centres = rng.normal(size=(TAGS, IMAGE_DIM)).astype("float32") * 3.0
    image = (image_centres[tags] + rng.normal(size=(ITEMS, IMAGE_DIM)) * 0.3).astype("float32")
    np.save(features / "text_features.npy", text)
    np.save(features / "image_features.npy", image)

    pd.DataFrame(
        {
            "internal_item_id": range(ITEMS),
            "external_item_id": [f"i{index}" for index in range(ITEMS)],
            "has_text_feature": np.ones(ITEMS, dtype=bool),
            "has_image_feature": np.ones(ITEMS, dtype=bool),
        }
    ).to_parquet(features / "modality_mask.parquet", index=False)

    (features / "multimodal_feature_manifest.json").write_text(
        json.dumps(
            {
                "feature_version": "1",
                "item_mapping_checksum": "integration-mapping",
                "catalogue_items": ITEMS,
                "modalities": {
                    "text": {
                        "available": True,
                        "matrix_file": "text_features.npy",
                        "dimension": TEXT_DIM,
                        "coverage": 1.0,
                        "rows_matched": ITEMS,
                    },
                    "image": {
                        "available": True,
                        "matrix_file": "image_features.npy",
                        "dimension": IMAGE_DIM,
                        "coverage": 1.0,
                        "rows_matched": ITEMS,
                    },
                },
            }
        )
    )

    # Each user consumes items from one tag block. The cold item is excluded
    # from every block by construction.
    rows = []
    for user in range(USERS):
        block = user % TAGS
        pool = [item for item in range(ITEMS - 1) if tags[item] == block]
        chain = rng.permutation(pool)[:5].tolist()
        for cut in range(2, 5):
            rows.append((user, chain[:cut], chain[cut]))
    sequences = pd.DataFrame(rows, columns=["internal_user_id", "item_sequence", "target_item"])

    return {
        "store": MultimodalFeatureStore(features),
        "sequences": sequences,
        "tags": tags,
        "root": tmp_path,
    }


@pytest.fixture
def trained(workspace):
    """A trained model plus the dataset it was fitted on."""
    dataset = TwoTowerTrainingDataset(
        workspace["sequences"],
        workspace["store"],
        num_items=ITEMS,
        num_users=USERS,
        maximum_history_length=4,
        item_tags=workspace["tags"],
        num_tags=TAGS,
    )
    config = TwoTowerConfig(
        embedding_dim=16,
        hidden_dims=(32,),
        dropout=0.0,
        maximum_history_length=4,
        batch_size=16,
        max_epochs=30,
        early_stopping_patience=30,
        learning_rate=0.01,
        temperature=0.2,
        device="cpu",
        seed=7,
    )
    torch.manual_seed(0)
    model = MultimodalTwoTower(
        config,
        text_dim=TEXT_DIM,
        image_dim=IMAGE_DIM,
        num_items=ITEMS,
        num_users=USERS,
        num_tags=TAGS,
    )
    history = TwoTowerTrainer(model, config, device="cpu").fit(dataset, dataset)
    model.eval()
    return model, dataset, history, workspace


def cold_arguments(workspace):
    """Encoder inputs for the cold item."""
    features = workspace["store"].get_batch(np.array([COLD_ITEM]))
    return (
        torch.from_numpy(features.text),
        torch.from_numpy(features.image),
        torch.tensor([True]),
        torch.tensor([True]),
        torch.tensor([int(workspace["tags"][COLD_ITEM])]),
    )


class TestWorkflow:
    def test_the_cold_item_is_genuinely_cold(self, trained) -> None:
        """Establishes the premise the rest of the test depends on."""
        _, dataset, _, _ = trained
        assert not dataset.warm_mask[COLD_ITEM]
        assert dataset.warm_mask.sum() > 0

    def test_training_runs_and_loss_decreases(self, trained) -> None:
        _, _, history, _ = trained
        assert len(history.train_loss) > 1
        assert history.train_loss[-1] < history.train_loss[0]
        assert all(np.isfinite(history.train_loss))

    def test_the_model_learns_the_content_clusters(self, trained) -> None:
        """A falling loss alone could just be id memorisation."""
        _, _, history, _ = trained
        assert history.in_batch_accuracy[-1] > history.in_batch_accuracy[0]

    def test_ran_on_cpu(self, trained) -> None:
        _, _, history, _ = trained
        assert history.device == "cpu"

    def test_users_and_warm_items_encode_into_one_space(self, trained) -> None:
        model, dataset, _, _ = trained
        batch = dataset.collate(np.arange(8))
        with torch.no_grad():
            users = model.encode_users(
                torch.from_numpy(batch.history_text_features),
                torch.from_numpy(batch.history_image_features),
                torch.from_numpy(batch.history_text_available),
                torch.from_numpy(batch.history_image_available),
                torch.from_numpy(batch.history_tag_ids),
                torch.from_numpy(batch.history_padding_mask),
                torch.from_numpy(batch.history_lengths),
                torch.from_numpy(batch.user_ids),
            )
            items = model.encode_items(
                torch.from_numpy(batch.positive_text_features),
                torch.from_numpy(batch.positive_image_features),
                torch.from_numpy(batch.positive_text_available),
                torch.from_numpy(batch.positive_image_available),
                torch.from_numpy(batch.positive_tag_ids),
                torch.from_numpy(batch.positive_item_ids),
                torch.from_numpy(batch.positive_warm_mask),
            )
        assert users.shape == items.shape == (8, 16)
        assert torch.isfinite(model.similarity(users, items)).all()


class TestColdItemEncoding:
    """Phase 5's reason to exist, end to end."""

    def test_the_cold_item_encodes_from_content(self, trained, workspace) -> None:
        model = trained[0]
        with torch.no_grad():
            cold = model.encode_items(
                *cold_arguments(workspace), torch.tensor([COLD_ITEM]), torch.tensor([False])
            )
        assert cold.shape == (1, 16)
        assert torch.isfinite(cold).all()
        assert float(cold.norm()) > 0.0

    def test_the_cold_encoding_uses_no_identity_residual(self, trained, workspace) -> None:
        """Exactly equal to the content-only path, not merely close to it."""
        model = trained[0]
        arguments = cold_arguments(workspace)
        with torch.no_grad():
            gated = model.encode_items(*arguments, torch.tensor([COLD_ITEM]), torch.tensor([False]))
            content_only = model.encode_items(*arguments, None, None)
        assert torch.allclose(gated, content_only, atol=1e-6)

    def test_a_trained_residual_would_otherwise_change_it(self, trained, workspace) -> None:
        """After training the residual is non-trivial, so the gate is doing work."""
        model = trained[0]
        arguments = cold_arguments(workspace)
        with torch.no_grad():
            cold = model.encode_items(*arguments, torch.tensor([COLD_ITEM]), torch.tensor([False]))
            as_if_warm = model.encode_items(*arguments, torch.tensor([0]), torch.tensor([True]))
        assert not torch.allclose(cold, as_if_warm, atol=1e-6)

    def test_the_cold_item_resembles_its_own_tag_block(self, trained, workspace) -> None:
        """Content encoding should place it near items it actually looks like."""
        model, _, _, _ = trained
        tags = workspace["tags"]
        same = [i for i in range(ITEMS - 1) if tags[i] == tags[COLD_ITEM]]
        other = [i for i in range(ITEMS - 1) if tags[i] != tags[COLD_ITEM]]
        features = workspace["store"].get_batch(np.arange(ITEMS))
        with torch.no_grad():
            everything = model.encode_items(
                torch.from_numpy(features.text),
                torch.from_numpy(features.image),
                torch.from_numpy(features.text_mask),
                torch.from_numpy(features.image_mask),
                torch.from_numpy(tags),
                None,
                None,
            )
        cold = everything[COLD_ITEM]
        same_similarity = float((everything[same] @ cold).mean())
        other_similarity = float((everything[other] @ cold).mean())
        assert same_similarity > other_similarity


class TestPersistenceRoundTrip:
    def test_every_embedding_survives_a_round_trip(self, trained, workspace) -> None:
        model, dataset, history, _ = trained
        store = workspace["store"]
        batch = dataset.collate(np.arange(8))
        arguments = cold_arguments(workspace)

        def encode(target):
            with torch.no_grad():
                return (
                    target.encode_users(
                        torch.from_numpy(batch.history_text_features),
                        torch.from_numpy(batch.history_image_features),
                        torch.from_numpy(batch.history_text_available),
                        torch.from_numpy(batch.history_image_available),
                        torch.from_numpy(batch.history_tag_ids),
                        torch.from_numpy(batch.history_padding_mask),
                        torch.from_numpy(batch.history_lengths),
                        torch.from_numpy(batch.user_ids),
                    ),
                    target.encode_items(
                        torch.from_numpy(batch.positive_text_features),
                        torch.from_numpy(batch.positive_image_features),
                        torch.from_numpy(batch.positive_text_available),
                        torch.from_numpy(batch.positive_image_available),
                        torch.from_numpy(batch.positive_tag_ids),
                        torch.from_numpy(batch.positive_item_ids),
                        torch.from_numpy(batch.positive_warm_mask),
                    ),
                    target.encode_items(
                        *arguments, torch.tensor([COLD_ITEM]), torch.tensor([False])
                    ),
                )

        before = encode(model)
        path = workspace["root"] / "artifact"
        save(
            model,
            path,
            metadata=build_metadata(
                model,
                feature_version=store.feature_version,
                feature_manifest_checksum=store.manifest_checksum(),
                mapping_checksum=store.mapping_checksum,
                training_history=history.to_dict(),
            ),
            training_history=history.to_dict(),
        )
        reloaded, metadata = load(
            path,
            device="cpu",
            expected_mapping_checksum=store.mapping_checksum,
            expected_feature_version=store.feature_version,
            expected_text_dim=TEXT_DIM,
            expected_image_dim=IMAGE_DIM,
        )
        reloaded.eval()
        after = encode(reloaded)

        assert torch.equal(before[0], after[0]), "user embeddings changed"
        assert torch.equal(before[1], after[1]), "warm item embeddings changed"
        assert torch.equal(before[2], after[2]), "cold item embedding changed"
        assert metadata["normalization"] == "l2"

    def test_the_cold_guarantee_holds_after_reload(self, trained, workspace) -> None:
        """A guarantee that only holds in the live model is not a guarantee."""
        model, _, history, _ = trained
        store = workspace["store"]
        path = workspace["root"] / "artifact-cold"
        save(
            model,
            path,
            metadata=build_metadata(
                model,
                feature_version=store.feature_version,
                feature_manifest_checksum=store.manifest_checksum(),
                mapping_checksum=store.mapping_checksum,
                training_history=history.to_dict(),
            ),
        )
        reloaded, _ = load(path, device="cpu")
        reloaded.eval()
        arguments = cold_arguments(workspace)
        with torch.no_grad():
            gated = reloaded.encode_items(
                *arguments, torch.tensor([COLD_ITEM]), torch.tensor([False])
            )
            content_only = reloaded.encode_items(*arguments, None, None)
        assert torch.allclose(gated, content_only, atol=1e-6)
