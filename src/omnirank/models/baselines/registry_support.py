"""Artifact registration for the offline retrieval models.

Wraps the Phase 1 :class:`~omnirank.artifacts.registry.ArtifactRegistry` so
every model records the same manifest fields. Registration is deliberately
non-clobbering: re-registering a version requires an explicit flag, because
silently rewriting a version another run may already have evaluated is how
"the metrics changed but the model didn't" happens.

Named for the Phase 3 baselines it was written for; Phase 4's LightGCN and
SASRec register through it unchanged, which is the point -- a model with its own
registration path could record a different set of fields and quietly become
incomparable.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from omnirank.artifacts.metadata import (
    ArtifactMetadata,
    ArtifactType,
    SupportedDevice,
    build_metadata,
)
from omnirank.artifacts.registry import ArtifactRegistry
from omnirank.core.logging import get_logger
from omnirank.data.io import sha256_file

logger = get_logger(__name__)

#: Bumped when the ranking-feature definitions change. Phase 3 uses no ranking
#: features, so the models depend only on raw interactions.
FEATURE_VERSION = "phase3-raw-interactions"


def directory_checksum(path: Path) -> str:
    """Order-independent checksum over a saved artifact directory."""
    import hashlib

    digest = hashlib.sha256()
    for file in sorted(p for p in Path(path).rglob("*") if p.is_file()):
        digest.update(file.name.encode())
        digest.update(sha256_file(file).encode())
    return digest.hexdigest()


def register_baseline(
    registry: ArtifactRegistry,
    *,
    model_name: str,
    model_version: str,
    artifact_dir: Path,
    dataset_identity: dict[str, Any],
    configuration_hash: str,
    random_seed: int,
    device: str,
    metrics: dict[str, float],
    fit_splits: tuple[str, ...],
    evaluation_protocol: str,
    mapping_checksum: str,
    extra_notes: str | None = None,
    overwrite: bool = False,
) -> ArtifactMetadata:
    """Register a fitted retrieval model with the full manifest."""
    supported = {
        "cpu": SupportedDevice.CPU,
        "mps": SupportedDevice.MPS,
        "cuda": SupportedDevice.CUDA,
    }
    # Popularity has no device affinity at all; BPR factors are saved on CPU and
    # load anywhere, so both are portable. The training device is recorded in
    # `notes` rather than pinned, so an artifact is not locked to one host.
    metadata = build_metadata(
        model_name=model_name,
        model_version=model_version,
        model_type=ArtifactType.RETRIEVAL_MODEL,
        training_data_version=(
            f"{dataset_identity.get('dataset_name')}@{dataset_identity.get('dataset_version')}"
        ),
        feature_version=FEATURE_VERSION,
        configuration_hash=configuration_hash,
        random_seed=random_seed,
        supported_device=SupportedDevice.ANY,
        # Phase 3 retrieves by brute-force scoring, not a vector index, so there
        # is no index build to pair with. Recorded as 1 because the artifact type
        # requires it (ADR-006); the real pairing arrives with FAISS in Phase 4.
        required_index_version=1,
        metrics=metrics,
        artifact_path=str(artifact_dir),
        notes=(extra_notes or "")
        + f" | fit_splits={'+'.join(fit_splits)} | protocol={evaluation_protocol}"
        + f" | trained_on_device={device}"
        + f" | artifact_sha256={directory_checksum(artifact_dir)}"
        + f" | item_mapping_checksum={mapping_checksum}"
        + f" | split_version={dataset_identity.get('split_version')}"
        + f" | mapping_version={dataset_identity.get('mapping_version')}"
        + f" | dataset_manifest={str(dataset_identity.get('dataset_manifest_sha256'))[:16]}"
        + f" | python={platform.python_version()}",
        id_mapping_fingerprints={"item": mapping_checksum},
    )
    registry.register(metadata, overwrite=overwrite)
    logger.info(
        "baseline.registered",
        artifact=metadata.key,
        supported_devices=sorted(supported),
        metrics=sorted(metrics),
    )
    return metadata


__all__ = ["FEATURE_VERSION", "directory_checksum", "register_baseline"]
