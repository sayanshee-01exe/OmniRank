"""Filesystem-backed artifact registry - component 14.

The registry is the single source of truth for "what models exist and may this
process use them". It stores one JSON manifest per artifact version under
``artifacts/metadata/<model_name>/<model_version>.json`` and never touches the
payload files themselves - loading a checkpoint is the owning model's job.

Why a directory of JSON rather than MLflow in Phase 1: the registry has to work
offline, on a laptop, with no server running, and be readable by a human with
``cat``. MLflow is added in Phase 2 as an *additional* sink for experiment
tracking, not as a replacement for this (ADR-006).

Concurrency: writes are atomic per file (write-temp-then-replace), which is
sufficient for the single-writer training jobs of Phase 1-2. A multi-writer
setup would need the database registry table defined in
``docs/data/database_schema.md``.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from omnirank.artifacts.metadata import ArtifactMetadata, ArtifactType
from omnirank.core.exceptions import (
    ArtifactCompatibilityError,
    ArtifactNotFoundError,
    ArtifactValidationError,
)
from omnirank.core.logging import get_logger

logger = get_logger(__name__)


class ArtifactRegistry:
    """Reads and writes artifact manifests under a metadata root."""

    def __init__(
        self, metadata_root: Path | str, *, artifact_root: Path | str | None = None
    ) -> None:
        """Create a registry.

        Args:
            metadata_root: Directory holding the manifest tree. Created lazily
                on first write; a missing directory reads as "no artifacts".
            artifact_root: Directory the manifests' ``artifact_path`` values are
                relative to. Defaults to ``metadata_root``'s parent, matching
                the repository layout.
        """
        self.metadata_root = Path(metadata_root)
        self.artifact_root = (
            Path(artifact_root) if artifact_root is not None else self.metadata_root.parent
        )

    # -- paths -------------------------------------------------------------- #
    def _manifest_path(self, model_name: str, model_version: str) -> Path:
        return self.metadata_root / model_name / f"{model_version}.json"

    def payload_path(self, metadata: ArtifactMetadata) -> Path:
        """Absolute path to the artifact payload described by ``metadata``.

        Raises:
            ArtifactValidationError: The manifest records no payload path.
        """
        if metadata.artifact_path is None:
            raise ArtifactValidationError(
                "Artifact manifest has no artifact_path, so its payload cannot be located",
                artifact=metadata.key,
            )
        candidate = Path(metadata.artifact_path)
        return candidate if candidate.is_absolute() else self.artifact_root / candidate

    # -- writing ------------------------------------------------------------ #
    def register(self, metadata: ArtifactMetadata, *, overwrite: bool = False) -> Path:
        """Write a manifest.

        Args:
            metadata: The manifest to store.
            overwrite: Permit replacing an existing version. Off by default:
                silently rewriting a version that another process may already
                have loaded is how "the metrics changed but the model didn't"
                bugs happen.

        Returns:
            Path of the written manifest.

        Raises:
            ArtifactValidationError: The version already exists and ``overwrite``
                is not set.
        """
        target = self._manifest_path(metadata.model_name, metadata.model_version)
        if target.exists() and not overwrite:
            raise ArtifactValidationError(
                "Artifact version is already registered. Bump model_version, or "
                "pass overwrite=True if you are deliberately replacing it.",
                artifact=metadata.key,
                path=str(target),
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        payload = metadata.model_dump(mode="json")
        # Atomic replace: a reader never observes a half-written manifest.
        handle, temp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        try:
            with os.fdopen(handle, "w") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
            Path(temp_name).replace(target)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise

        logger.info(
            "artifact.registered",
            artifact=metadata.key,
            artifact_type=metadata.model_type.value,
            path=str(target),
        )
        return target

    # -- reading ------------------------------------------------------------ #
    def get(self, model_name: str, model_version: str) -> ArtifactMetadata:
        """Load one manifest.

        Raises:
            ArtifactNotFoundError: No such name/version.
            ArtifactValidationError: The manifest on disk is malformed.
        """
        path = self._manifest_path(model_name, model_version)
        if not path.is_file():
            raise ArtifactNotFoundError(
                "Artifact is not registered",
                model_name=model_name,
                model_version=model_version,
                searched=str(path),
            )
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError(
                "Artifact manifest is not valid JSON", path=str(path), reason=str(exc)
            ) from exc
        try:
            return ArtifactMetadata.model_validate(payload)
        except ValueError as exc:
            raise ArtifactValidationError(
                f"Artifact manifest failed validation: {exc}", path=str(path)
            ) from exc

    def versions(self, model_name: str) -> list[str]:
        """Registered versions of one model, sorted by creation time (oldest first)."""
        directory = self.metadata_root / model_name
        if not directory.is_dir():
            return []
        found = [self.get(model_name, path.stem) for path in sorted(directory.glob("*.json"))]
        return [item.model_version for item in sorted(found, key=lambda meta: meta.created_at)]

    def latest(self, model_name: str) -> ArtifactMetadata:
        """Most recently created version of ``model_name``.

        Raises:
            ArtifactNotFoundError: The model has no registered versions.
        """
        versions = self.versions(model_name)
        if not versions:
            raise ArtifactNotFoundError("Model has no registered versions", model_name=model_name)
        return self.get(model_name, versions[-1])

    def iter_all(self) -> Iterator[ArtifactMetadata]:
        """Yield every registered manifest. Malformed files are skipped, loudly."""
        if not self.metadata_root.is_dir():
            return
        for path in sorted(self.metadata_root.glob("*/*.json")):
            try:
                yield self.get(path.parent.name, path.stem)
            except ArtifactValidationError as exc:
                logger.error("artifact.manifest_unreadable", path=str(path), reason=str(exc))

    def list_all(self, *, artifact_type: ArtifactType | None = None) -> list[ArtifactMetadata]:
        """All manifests, newest first, optionally filtered by type."""
        found = list(self.iter_all())
        if artifact_type is not None:
            found = [item for item in found if item.model_type is artifact_type]
        return sorted(found, key=lambda meta: meta.created_at, reverse=True)

    # -- compatibility ------------------------------------------------------ #
    def require_compatible(
        self,
        metadata: ArtifactMetadata,
        *,
        device: str,
        index_version: int | None = None,
    ) -> ArtifactMetadata:
        """Return ``metadata`` if it may be used here, else raise.

        Raises:
            ArtifactCompatibilityError: Device or index version mismatch.
        """
        if not metadata.is_compatible_with(device=device, index_version=index_version):
            raise ArtifactCompatibilityError(
                "Artifact is registered but incompatible with this environment. "
                "Rebuild the index or re-export the model against the current "
                "index version (see ADR-006).",
                artifact=metadata.key,
                supported_device=metadata.supported_device.value,
                host_device=device,
                required_index_version=metadata.required_index_version,
                available_index_version=index_version,
            )
        return metadata

    # -- health ------------------------------------------------------------- #
    def is_ready(self) -> bool:
        """Whether at least one artifact is registered.

        Used by ``GET /ready``: a serving process with an empty registry cannot
        answer recommendation requests and must not be sent traffic.
        """
        return any(True for _ in self.iter_all())


__all__ = ["ArtifactRegistry"]
