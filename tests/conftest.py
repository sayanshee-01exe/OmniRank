"""Shared fixtures.

Every fixture here is offline, filesystem-local, and fast. No fixture starts a
container, downloads a model, opens a socket, or reads the developer's real
environment - which is what lets ``make test`` be the thing you run on every
save.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from omnirank.artifacts.metadata import ArtifactMetadata, ArtifactType, SupportedDevice
from omnirank.artifacts.registry import ArtifactRegistry
from omnirank.core.config import AppConfig, load_config
from omnirank.core.logging import reset_context

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"

# A fixed instant so nothing in the suite depends on the wall clock.
FROZEN_NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clean_log_context():
    """Prevent context variables leaking between tests."""
    reset_context()
    yield
    reset_context()


@pytest.fixture(scope="session")
def config_dir() -> Path:
    """The repository's real config directory."""
    return CONFIG_DIR


@pytest.fixture
def config(config_dir: Path) -> AppConfig:
    """The real configuration, isolated from the developer's environment.

    ``env={}`` and a non-existent dotenv path together guarantee this is exactly
    what ``configs/*.yaml`` says, so a test cannot pass or fail because of what
    is in someone's shell.
    """
    return load_config(config_dir, env={}, dotenv_path=config_dir / "__no_such_dotenv__")


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    """A throwaway artifact root laid out like the real one."""
    root = tmp_path / "artifacts"
    for child in ("metadata", "models", "embeddings", "indexes", "mappings"):
        (root / child).mkdir(parents=True)
    return root


@pytest.fixture
def registry(artifact_root: Path) -> ArtifactRegistry:
    """An empty registry over a temporary artifact root."""
    return ArtifactRegistry(artifact_root / "metadata", artifact_root=artifact_root)


@pytest.fixture
def sample_metadata() -> ArtifactMetadata:
    """A valid artifact manifest for a device-agnostic model."""
    return ArtifactMetadata(
        model_name="popularity",
        model_version="v1",
        model_type=ArtifactType.RANKER,
        created_at=FROZEN_NOW,
        training_data_version="ecommerce-subset@v0",
        feature_version="f1",
        configuration_hash="a" * 64,
        random_seed=42,
        framework_version={},
        python_version="3.11.15",
        git_commit=None,
        metrics={"recall@20": 0.11},
        supported_device=SupportedDevice.ANY,
    )


@pytest.fixture
def user_row() -> dict[str, object]:
    """A minimal valid user record."""
    return {"user_id": "u1", "created_at": "2026-01-01T00:00:00Z"}


@pytest.fixture
def item_row() -> dict[str, object]:
    """A minimal valid item record."""
    return {
        "item_id": "i1",
        "title": "Wireless Headphones",
        "category": "audio",
        "price": 99.5,
        "available": True,
        "created_at": "2026-01-01T00:00:00Z",
    }


@pytest.fixture
def interaction_row() -> dict[str, object]:
    """A minimal valid interaction record."""
    return {
        "interaction_id": "e1",
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "click",
        "timestamp": "2026-02-01T10:00:00Z",
    }


# --------------------------------------------------------------------------- #
# Phase 2: dataset pipeline fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def pixelrec_fixture_dir(tmp_path: Path) -> Path:
    """A synthetic PixelRec-shaped raw directory."""
    from tests.fixtures.pixelrec import write_fixture

    raw = tmp_path / "raw" / "pixelrec50k"
    write_fixture(raw)
    return raw


@pytest.fixture
def pixelrec_config(config_dir: Path, tmp_path: Path, pixelrec_fixture_dir: Path) -> AppConfig:
    """The real PixelRec profile, repointed at the synthetic fixture.

    Filtering thresholds are relaxed relative to the shipped profile because the
    fixture is tiny; the shipped thresholds would empty it, which is correct
    behaviour but useless for testing the stages that follow.
    """
    from omnirank.core.config import ENV_PREFIX

    return load_config(
        config_dir,
        data_profile="data/pixelrec50k.yaml",
        env={
            f"{ENV_PREFIX}DATA__DATASET__RAW_DIR": str(pixelrec_fixture_dir),
            f"{ENV_PREFIX}DATA__DATASET__INTERIM_DIR": str(tmp_path / "interim"),
            f"{ENV_PREFIX}DATA__DATASET__PROCESSED_DIR": str(tmp_path / "processed"),
            f"{ENV_PREFIX}DATA__FILTERING__MIN_INTERACTIONS_PER_USER": "2",
            f"{ENV_PREFIX}DATA__FILTERING__MIN_INTERACTIONS_PER_ITEM": "2",
            f"{ENV_PREFIX}DATA__SEQUENCES__MAX_LENGTH": "10",
        },
        dotenv_path=config_dir / "__no_such_dotenv__",
    )


@pytest.fixture
def split_frame() -> pd.DataFrame:
    """A tiny hand-built split-labelled frame with known, checkable properties.

    Three users: one with a full history, one with the bare minimum, one
    ineligible. Written by hand rather than generated so every expected number
    in the tests that use it can be verified by reading.
    """
    rows = [
        # user 0: 5 events -> 3 train, 1 validation, 1 test
        *[(0, item, order, "train") for item, order in [(10, 0), (11, 1), (12, 2)]],
        (0, 13, 3, "validation"),
        (0, 14, 4, "test"),
        # user 1: 3 events -> 1 train, 1 validation, 1 test
        (1, 10, 0, "train"),
        (1, 15, 1, "validation"),
        (1, 16, 2, "test"),
        # user 2: 2 events -> ineligible, all training
        (2, 10, 0, "train"),
        (2, 11, 1, "train"),
    ]
    frame = pd.DataFrame(
        rows, columns=["internal_user_id", "internal_item_id", "interaction_order", "split"]
    )
    frame["external_user_id"] = "u" + frame["internal_user_id"].astype(str)
    frame["external_item_id"] = "i" + frame["internal_item_id"].astype(str)
    frame["interaction_id"] = "e" + frame.index.astype(str)
    frame["timestamp"] = 1_640_995_200 + frame["interaction_order"] * 3600
    frame["event_type"] = "interaction"
    frame["interaction_weight"] = 1.0
    frame["source_row_id"] = frame.index
    return frame
