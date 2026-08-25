"""Two-tower persistence and identity enforcement.

A saved two-tower model is meaningless apart from the mapping and feature store
it was fitted against. Its item vectors are indexed by that mapping and derived
from those features; loaded beside different ones it does not fail, it returns
confident recommendations for the wrong items. So the round-trip tests check
exact equality, and the rejection tests check that a supplied identity mismatch
is fatal rather than a warning.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from omnirank.core.exceptions import ArtifactValidationError
from omnirank.models.two_tower import build_metadata, load, save
from omnirank.models.two_tower.persistence import (
    CONFIG_FILENAME,
    METADATA_FILENAME,
    STATE_FILENAME,
)

from .conftest import COLD_ITEM, IMAGE_DIM, TEXT_DIM


@pytest.fixture
def saved(model, store, dataset, tmp_path):
    """A saved model plus the embeddings it produced before saving."""
    batch = dataset.collate(np.arange(8))
    t = {
        name: torch.from_numpy(getattr(batch, name))
        for name in (
            "positive_text_features",
            "positive_image_features",
            "positive_text_available",
            "positive_image_available",
            "positive_tag_ids",
            "positive_item_ids",
            "positive_warm_mask",
            "history_text_features",
            "history_image_features",
            "history_text_available",
            "history_image_available",
            "history_tag_ids",
            "history_padding_mask",
            "history_lengths",
            "user_ids",
        )
    }
    with torch.no_grad():
        items = model.encode_items(
            t["positive_text_features"],
            t["positive_image_features"],
            t["positive_text_available"],
            t["positive_image_available"],
            t["positive_tag_ids"],
            t["positive_item_ids"],
            t["positive_warm_mask"],
        )
        users = model.encode_users(
            t["history_text_features"],
            t["history_image_features"],
            t["history_text_available"],
            t["history_image_available"],
            t["history_tag_ids"],
            t["history_padding_mask"],
            t["history_lengths"],
            t["user_ids"],
        )
    metadata = build_metadata(
        model,
        feature_version=store.feature_version,
        feature_manifest_checksum=store.manifest_checksum(),
        mapping_checksum=store.mapping_checksum,
        training_history={"best_epoch": 3, "epochs_run": 5, "device": "cpu"},
    )
    path = tmp_path / "model"
    save(model, path, metadata=metadata, training_history={"best_epoch": 3})
    return path, t, items, users, store


class TestRoundTrip:
    def test_required_files_are_written(self, saved) -> None:
        path = saved[0]
        for name in (STATE_FILENAME, CONFIG_FILENAME, METADATA_FILENAME):
            assert (path / name).is_file()

    def test_item_embeddings_are_identical(self, saved) -> None:
        path, t, items, _, _ = saved
        loaded, _ = load(path, device="cpu")
        loaded.eval()
        with torch.no_grad():
            after = loaded.encode_items(
                t["positive_text_features"],
                t["positive_image_features"],
                t["positive_text_available"],
                t["positive_image_available"],
                t["positive_tag_ids"],
                t["positive_item_ids"],
                t["positive_warm_mask"],
            )
        assert torch.equal(items, after)

    def test_user_embeddings_are_identical(self, saved) -> None:
        path, t, _, users, _ = saved
        loaded, _ = load(path, device="cpu")
        loaded.eval()
        with torch.no_grad():
            after = loaded.encode_users(
                t["history_text_features"],
                t["history_image_features"],
                t["history_text_available"],
                t["history_image_available"],
                t["history_tag_ids"],
                t["history_padding_mask"],
                t["history_lengths"],
                t["user_ids"],
            )
        assert torch.equal(users, after)

    def test_cold_item_encoding_survives_the_round_trip(self, saved, model, item_tags) -> None:
        """The cold guarantee must hold after reload, not only in the live model."""
        path, _, _, _, store = saved
        features = store.get_batch(np.array([COLD_ITEM]))
        args = (
            torch.from_numpy(features.text),
            torch.from_numpy(features.image),
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([int(item_tags[COLD_ITEM])]),
        )
        loaded, _ = load(path, device="cpu")
        loaded.eval()
        with torch.no_grad():
            before = model.encode_items(*args, torch.tensor([COLD_ITEM]), torch.tensor([False]))
            after = loaded.encode_items(*args, torch.tensor([COLD_ITEM]), torch.tensor([False]))
            content_only = loaded.encode_items(*args, None, None)
        assert torch.equal(before, after)
        assert torch.allclose(after, content_only, atol=1e-6)

    def test_scoring_semantics_are_recorded(self, saved) -> None:
        """FAISS must be built under the same normalisation rule."""
        _, metadata = load(saved[0], device="cpu")
        assert metadata["normalization"] == "l2"
        assert metadata["embedding_dim"] == 16

    def test_modality_and_cold_policy_are_recorded(self, saved) -> None:
        _, metadata = load(saved[0], device="cpu")
        schema = metadata["modality_schema"]
        assert schema["text_dim"] == TEXT_DIM
        assert "warm mask" in schema["cold_item_policy"]

    def test_provenance_is_recorded(self, saved) -> None:
        _, metadata = load(saved[0], device="cpu")
        for key in ("git_commit", "python_version", "torch_version", "seed", "created_at"):
            assert metadata[key]


class TestIdentityEnforcement:
    def test_matching_identity_loads(self, saved) -> None:
        path, _, _, _, store = saved
        load(
            path,
            device="cpu",
            expected_mapping_checksum=store.mapping_checksum,
            expected_feature_version=store.feature_version,
            expected_text_dim=TEXT_DIM,
            expected_image_dim=IMAGE_DIM,
        )

    def test_a_different_mapping_is_refused(self, saved) -> None:
        """Every recommended id would resolve to a different item."""
        with pytest.raises(ArtifactValidationError, match="mapping_checksum"):
            load(saved[0], device="cpu", expected_mapping_checksum="another-mapping")

    def test_a_different_feature_version_is_refused(self, saved) -> None:
        with pytest.raises(ArtifactValidationError, match="feature_version"):
            load(saved[0], device="cpu", expected_feature_version="99")

    @pytest.mark.parametrize("field", ["expected_text_dim", "expected_image_dim"])
    def test_a_different_feature_dimension_is_refused(self, saved, field: str) -> None:
        with pytest.raises(ArtifactValidationError):
            load(saved[0], device="cpu", **{field: 4096})

    def test_unsupplied_identities_are_not_invented(self, saved) -> None:
        """A caller that does not know an identity is not forced to guess one."""
        load(saved[0], device="cpu")


class TestCorruptArtifacts:
    def test_missing_files_are_reported(self, tmp_path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(ArtifactValidationError, match="incomplete"):
            load(tmp_path / "empty")

    def test_missing_metadata_is_reported(self, saved) -> None:
        (saved[0] / METADATA_FILENAME).unlink()
        with pytest.raises(ArtifactValidationError):
            load(saved[0])

    def test_invalid_json_is_reported(self, saved) -> None:
        (saved[0] / METADATA_FILENAME).write_text("{not json")
        with pytest.raises(ArtifactValidationError, match="not valid JSON"):
            load(saved[0])

    def test_an_artifact_from_another_model_is_refused(self, saved) -> None:
        path = saved[0]
        metadata = json.loads((path / METADATA_FILENAME).read_text())
        metadata["model"] = "lightgcn"
        (path / METADATA_FILENAME).write_text(json.dumps(metadata))
        with pytest.raises(ArtifactValidationError, match="different model type"):
            load(path)

    def test_an_unsupported_format_version_is_refused(self, saved) -> None:
        path = saved[0]
        metadata = json.loads((path / METADATA_FILENAME).read_text())
        metadata["format_version"] = 99
        (path / METADATA_FILENAME).write_text(json.dumps(metadata))
        with pytest.raises(ArtifactValidationError, match="format version"):
            load(path)

    def test_corrupt_weights_are_reported(self, saved) -> None:
        (saved[0] / STATE_FILENAME).write_bytes(b"not a checkpoint")
        with pytest.raises(ArtifactValidationError, match="corrupt"):
            load(saved[0])

    def test_mismatched_weights_are_not_silently_initialised(self, saved) -> None:
        """Loose loading would leave parameters random without saying so."""
        path = saved[0]
        config = json.loads((path / CONFIG_FILENAME).read_text())
        config["embedding_dim"] = 32
        (path / CONFIG_FILENAME).write_text(json.dumps(config))
        with pytest.raises(ArtifactValidationError, match="randomly"):
            load(path)
