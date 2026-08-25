"""Device-neutral persistence for the two-tower model.

A saved two-tower model is meaningless without the things it was fitted
against. Its item vectors are indexed by a mapping, derived from a specific
feature store, and scored under a specific normalisation rule. Load it beside a
different mapping and nothing raises -- every recommended id resolves to the
wrong item, confidently. So identity travels with the weights and is checked on
load, the same argument ADR-006 makes for indexes.

Weights are saved on CPU and loaded with ``weights_only=True``: a checkpoint is
data, and executing arbitrary pickle from one is a remote-code-execution path
that buys nothing here.
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import torch

from omnirank.artifacts.metadata import detect_git_commit
from omnirank.core.exceptions import ArtifactValidationError
from omnirank.core.logging import get_logger
from omnirank.models.two_tower.config import TwoTowerConfig
from omnirank.models.two_tower.model import MultimodalTwoTower

logger = get_logger(__name__)

FORMAT_VERSION: Final = 1
MODEL_NAME: Final = "two_tower"

STATE_FILENAME: Final = "model.pt"
CONFIG_FILENAME: Final = "config.json"
METADATA_FILENAME: Final = "metadata.json"
HISTORY_FILENAME: Final = "training_history.json"

REQUIRED_FILES: Final = (STATE_FILENAME, CONFIG_FILENAME, METADATA_FILENAME)


def build_metadata(
    model: MultimodalTwoTower,
    *,
    feature_version: str,
    feature_manifest_checksum: str,
    mapping_checksum: str,
    dataset_identity: dict[str, Any] | None = None,
    fit_splits: tuple[str, ...] = (),
    training_history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything needed to decide whether this artifact may be used."""
    history = training_history or {}
    return {
        "model": MODEL_NAME,
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": detect_git_commit() or "unknown",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        # Identity. Each of these, mismatched, produces wrong answers rather
        # than errors, which is why they are stored rather than inferred.
        "feature_version": feature_version,
        "feature_manifest_checksum": feature_manifest_checksum,
        "mapping_checksum": mapping_checksum,
        "dataset_identity": dataset_identity or {},
        "fit_splits": list(fit_splits),
        # Scoring semantics. FAISS must be built under the same rule.
        "embedding_dim": model.config.embedding_dim,
        "normalization": "l2" if model.config.l2_normalize else "none",
        "temperature": model.config.temperature,
        "history_pooling": model.config.history_pooling,
        "modality_schema": model.modality_schema(),
        "text_dim": model.text_dim,
        "image_dim": model.image_dim,
        "num_items": model.num_items,
        "num_users": model.num_users,
        "num_tags": model.num_tags,
        "seed": model.config.seed,
        "best_epoch": history.get("best_epoch", 0),
        "epochs_run": history.get("epochs_run", 0),
        "device_trained_on": history.get("device", "unknown"),
    }


def save(
    model: MultimodalTwoTower,
    path: str | Path,
    *,
    metadata: dict[str, Any],
    training_history: dict[str, Any] | None = None,
) -> Path:
    """Persist weights, configuration and identity to a directory."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    # CPU tensors, so an artifact trained on MPS loads anywhere.
    torch.save(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        target / STATE_FILENAME,
    )
    (target / CONFIG_FILENAME).write_text(
        json.dumps(model.config.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (target / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n"
    )
    if training_history is not None:
        (target / HISTORY_FILENAME).write_text(
            json.dumps(training_history, indent=2, sort_keys=True) + "\n"
        )
    logger.info(
        "two_tower.saved",
        path=str(target),
        embedding_dim=metadata.get("embedding_dim"),
        normalization=metadata.get("normalization"),
    )
    return target


def load(
    path: str | Path,
    *,
    device: str = "cpu",
    expected_mapping_checksum: str | None = None,
    expected_feature_version: str | None = None,
    expected_text_dim: int | None = None,
    expected_image_dim: int | None = None,
) -> tuple[MultimodalTwoTower, dict[str, Any]]:
    """Restore a saved model, refusing an incompatible one.

    Every ``expected_*`` argument is optional and checked only when supplied,
    so a caller that legitimately does not know an identity is not forced to
    invent one. What is *not* optional is failing loudly when a supplied
    identity disagrees.

    Raises:
        ArtifactValidationError: Files are missing or corrupt, the artifact
            belongs to a different model, or an identity mismatches.
    """
    source = Path(path)
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise ArtifactValidationError(
            "Two-tower artifact is incomplete", path=str(source), missing=missing
        )

    try:
        raw_config = json.loads((source / CONFIG_FILENAME).read_text())
        metadata: dict[str, Any] = json.loads((source / METADATA_FILENAME).read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError(
            "Two-tower artifact metadata is not valid JSON", path=str(source)
        ) from exc

    if metadata.get("model") != MODEL_NAME:
        raise ArtifactValidationError(
            "Artifact was written by a different model type",
            expected=MODEL_NAME,
            found=metadata.get("model"),
        )
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ArtifactValidationError(
            "Unsupported two-tower artifact format version",
            expected=FORMAT_VERSION,
            found=metadata.get("format_version"),
        )

    problems: list[str] = []
    if (
        expected_mapping_checksum
        and metadata.get("mapping_checksum")
        and expected_mapping_checksum != metadata["mapping_checksum"]
    ):
        problems.append("mapping_checksum differs; every item id would resolve differently")
    if (
        expected_feature_version
        and metadata.get("feature_version")
        and expected_feature_version != metadata["feature_version"]
    ):
        problems.append(
            f"feature_version {metadata['feature_version']!r} != {expected_feature_version!r}"
        )
    if expected_text_dim is not None and metadata.get("text_dim") != expected_text_dim:
        problems.append(f"text_dim {metadata.get('text_dim')} != {expected_text_dim}")
    if expected_image_dim is not None and metadata.get("image_dim") != expected_image_dim:
        problems.append(f"image_dim {metadata.get('image_dim')} != {expected_image_dim}")
    if problems:
        raise ArtifactValidationError(
            "Two-tower artifact is incompatible with the supplied environment. "
            "Loading it anyway would produce confident, wrong recommendations.",
            problems=problems,
        )

    try:
        # weights_only=True: a checkpoint is data. Executing pickle from one is
        # an arbitrary-code path that buys nothing.
        state = torch.load(source / STATE_FILENAME, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactValidationError(
            "Two-tower weights could not be read; the file may be corrupt",
            path=str(source / STATE_FILENAME),
            reason=str(exc)[:200],
        ) from exc

    model = MultimodalTwoTower(
        TwoTowerConfig.from_dict(raw_config),
        text_dim=int(metadata["text_dim"]),
        image_dim=int(metadata["image_dim"]),
        num_items=int(metadata["num_items"]),
        num_users=int(metadata["num_users"]),
        num_tags=int(metadata.get("num_tags", 1)),
    )
    try:
        # strict=True: a missing key means a parameter would keep its random
        # initialisation, which is silent and produces a subtly wrong model.
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ArtifactValidationError(
            "Saved weights do not match the model this configuration builds. "
            "Loading them loosely would leave some parameters randomly "
            "initialised without saying so.",
            reason=str(exc)[:300],
        ) from exc

    model.eval()
    from omnirank.models.baselines.bpr import resolve_torch_device

    model.to(resolve_torch_device(device))
    logger.info("two_tower.loaded", path=str(source), device=device)
    return model, metadata


def load_training_history(path: str | Path) -> dict[str, Any]:
    """Read the saved training history, or an empty record when absent."""
    history_path = Path(path) / HISTORY_FILENAME
    if not history_path.is_file():
        return {}
    payload: dict[str, Any] = json.loads(history_path.read_text())
    return payload


__all__ = [
    "CONFIG_FILENAME",
    "FORMAT_VERSION",
    "HISTORY_FILENAME",
    "METADATA_FILENAME",
    "MODEL_NAME",
    "REQUIRED_FILES",
    "STATE_FILENAME",
    "build_metadata",
    "load",
    "load_training_history",
    "save",
]
