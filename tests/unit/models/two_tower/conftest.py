"""Shared fixtures for the two-tower suite.

The synthetic feature store is the important one. It is written to disk and
loaded through the real :class:`MultimodalFeatureStore` rather than mocked, so
these tests exercise the same memory-mapping, masking and identity-checking code
that the real corpus goes through. A mocked store would let a shape or masking
bug pass here and fail only on 8.6 GiB of real vectors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="Two-tower requires the 'retrieval' extra")

from omnirank.features.multimodal_store import MultimodalFeatureStore

ITEMS = 40
USERS = 30
TEXT_DIM = 16
IMAGE_DIM = 12
TAGS = 4
#: Never appears in any history or target, so the dataset marks it cold.
COLD_ITEM = ITEMS - 1
#: Deliberately missing one modality each, so the masks are exercised.
NO_TEXT_ITEM = 0
NO_IMAGE_ITEM = 1


def write_feature_store(
    root: Path, *, items: int = ITEMS, text_dim: int = TEXT_DIM, image_dim: int = IMAGE_DIM
) -> MultimodalFeatureStore:
    """Build a real on-disk feature store with mixed modality availability."""
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "text_features.npy", rng.normal(size=(items, text_dim)).astype("float32"))
    np.save(root / "image_features.npy", rng.normal(size=(items, image_dim)).astype("float32"))

    has_text = np.ones(items, dtype=bool)
    has_image = np.ones(items, dtype=bool)
    has_text[NO_TEXT_ITEM] = False
    has_image[NO_IMAGE_ITEM] = False
    pd.DataFrame(
        {
            "internal_item_id": range(items),
            "external_item_id": [f"i{index}" for index in range(items)],
            "has_text_feature": has_text,
            "has_image_feature": has_image,
        }
    ).to_parquet(root / "modality_mask.parquet", index=False)

    (root / "multimodal_feature_manifest.json").write_text(
        json.dumps(
            {
                "feature_version": "1",
                "item_mapping_checksum": "fixture-mapping",
                "catalogue_items": items,
                "modalities": {
                    "text": {
                        "available": True,
                        "matrix_file": "text_features.npy",
                        "dimension": text_dim,
                        "coverage": 1.0,
                        "rows_matched": items,
                    },
                    "image": {
                        "available": True,
                        "matrix_file": "image_features.npy",
                        "dimension": image_dim,
                        "coverage": 1.0,
                        "rows_matched": items,
                    },
                },
            }
        )
    )
    return MultimodalFeatureStore(root)


def build_sequences(users: int = USERS) -> pd.DataFrame:
    """Block-preferring users. The cold item never appears in any history."""
    rng = np.random.default_rng(1)
    rows = []
    for user in range(users):
        block = user % 3
        # Distinct items per chain. Phase 2 deduplicates repeat events, so a
        # real target never reappears inside its own history -- verified against
        # 50,000 PixelRec rows -- and a fixture with repeats would trip the
        # dataset's leakage guard for a reason the real corpus never produces.
        chain = rng.permutation(np.arange(block * 10, block * 10 + 10))[:6].tolist()
        for cut in range(2, 6):
            rows.append((user, chain[:cut], chain[cut]))
    return pd.DataFrame(rows, columns=["internal_user_id", "item_sequence", "target_item"])


@pytest.fixture
def store(tmp_path: Path) -> MultimodalFeatureStore:
    return write_feature_store(tmp_path / "features")


@pytest.fixture
def sequences() -> pd.DataFrame:
    return build_sequences()


@pytest.fixture
def item_tags() -> np.ndarray:
    return np.arange(ITEMS) % TAGS


@pytest.fixture
def dataset(store, sequences, item_tags):
    from omnirank.models.two_tower import TwoTowerTrainingDataset

    return TwoTowerTrainingDataset(
        sequences,
        store,
        num_items=ITEMS,
        num_users=USERS,
        maximum_history_length=5,
        item_tags=item_tags,
        num_tags=TAGS,
    )


@pytest.fixture
def config():
    from omnirank.models.two_tower import TwoTowerConfig

    return TwoTowerConfig(
        embedding_dim=16,
        hidden_dims=(32,),
        dropout=0.0,
        maximum_history_length=5,
        batch_size=16,
        device="cpu",
    )


@pytest.fixture
def model(config):
    import torch

    from omnirank.models.two_tower import MultimodalTwoTower

    torch.manual_seed(0)
    built = MultimodalTwoTower(
        config,
        text_dim=TEXT_DIM,
        image_dim=IMAGE_DIM,
        num_items=ITEMS,
        num_users=USERS,
        num_tags=TAGS,
    )
    built.eval()
    return built
