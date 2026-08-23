"""Shared fixtures.

Every fixture here is offline, filesystem-local, and fast. No fixture starts a
container, downloads a model, opens a socket, or reads the developer's real
environment - which is what lets ``make test`` be the thing you run on every
save.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

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
