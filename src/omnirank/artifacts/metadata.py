"""Artifact metadata contract - component 14.

Every artifact OmniRank produces - a trained model, an embedding matrix, a
vector index, an id mapping - ships with a metadata record. An artifact without
one is unusable by definition: the registry refuses to load it.

The record answers four questions that a bare ``.pt`` file cannot:

* *What produced this?* - code version, config hash, seed, framework versions.
* *What data is it about?* - training data version, feature version.
* *How good was it?* - the offline metrics measured at export time.
* *Where can it run, and with what?* - supported device, required index version.

The last one is the load-bearing part. A retrieval model and the FAISS index it
queries are only meaningful as a pair; ADR-006 makes that pairing explicit here
so a mismatched combination fails loudly at startup rather than quietly
returning nonsense at serving time.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

METADATA_FORMAT_VERSION = 1


class ArtifactType(StrEnum):
    """What kind of thing an artifact is.

    Determines how the registry resolves its path and which loader can read it.
    """

    MAPPING = "mapping"
    RETRIEVAL_MODEL = "retrieval_model"
    RANKER = "ranker"
    EMBEDDING = "embedding"
    INDEX = "index"
    FEATURE_SET = "feature_set"


class SupportedDevice(StrEnum):
    """Devices an artifact can be loaded onto.

    ``any`` marks artifacts with no device affinity at all (mappings, indexes,
    tree-based rankers). Neural artifacts declare a concrete device so that a
    checkpoint saved from an MPS run is not silently loaded somewhere it will
    produce different numerics.
    """

    ANY = "any"
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"


class ArtifactMetadata(BaseModel):
    """The manifest accompanying every artifact.

    All fields are required unless noted. Optional fields are optional because
    they genuinely do not apply to some artifact types (an id mapping has no
    metrics; a tree ranker has no vector index).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    # -- identity ----------------------------------------------------------- #
    model_name: str = Field(min_length=1, max_length=128)
    # Semantic-ish version chosen by the training run, e.g. "v3" or "2026-08-24".
    model_version: str = Field(min_length=1, max_length=64)
    model_type: ArtifactType

    # -- provenance --------------------------------------------------------- #
    created_at: datetime
    # Version of the dataset snapshot the artifact was trained on. Comes from
    # `data.dataset_version`; changing it invalidates the artifact.
    training_data_version: str = Field(min_length=1, max_length=64)
    # Version of the feature-generation logic. Bumped whenever a feature's
    # definition changes, which is the training/serving skew tripwire.
    feature_version: str = Field(min_length=1, max_length=64)
    # SHA-256 of the training-relevant configuration sections.
    configuration_hash: str = Field(min_length=8, max_length=64)
    random_seed: int = Field(ge=0)
    # e.g. {"torch": "2.3.1", "lightgbm": "4.3.0"}. Empty for pure-python artifacts.
    framework_version: dict[str, str] = Field(default_factory=dict)
    python_version: str = Field(min_length=1)
    # Repository commit, or None when built outside a git checkout.
    git_commit: str | None = Field(default=None, max_length=40)

    # -- quality ------------------------------------------------------------ #
    # Offline metrics measured at export time, e.g. {"recall@20": 0.14}.
    # Empty is legitimate (an id mapping has no metrics) but is never a claim
    # of quality - the registry does not invent numbers.
    metrics: dict[str, float] = Field(default_factory=dict)

    # -- compatibility ------------------------------------------------------ #
    supported_device: SupportedDevice = SupportedDevice.ANY
    # The index build version this artifact's embeddings must be paired with.
    # None for artifacts that do not participate in vector retrieval.
    required_index_version: int | None = Field(default=None, ge=1)

    # -- bookkeeping -------------------------------------------------------- #
    format_version: int = METADATA_FORMAT_VERSION
    # Path to the artifact payload, relative to the artifact root, so a registry
    # directory stays portable between machines.
    artifact_path: str | None = None
    # Fingerprints of the id mappings the artifact was built against. A model
    # whose recorded fingerprint differs from the loaded mapping would resolve
    # every dense index to the wrong entity.
    id_mapping_fingerprints: dict[str, str] = Field(default_factory=dict)
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("created_at")
    @classmethod
    def _created_at_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("metrics")
    @classmethod
    def _finite_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        for name, number in value.items():
            if number != number:  # NaN
                raise ValueError(f"metric {name!r} is NaN; refusing to record it")
        return value

    @model_validator(mode="after")
    def _check_retrieval_pairing(self) -> Self:
        """Retrieval artifacts must declare which index build they belong to."""
        needs_index = {ArtifactType.RETRIEVAL_MODEL, ArtifactType.EMBEDDING, ArtifactType.INDEX}
        if self.model_type in needs_index and self.required_index_version is None:
            raise ValueError(
                f"artifact type {self.model_type.value!r} participates in vector "
                "retrieval and must set required_index_version (see ADR-006)"
            )
        return self

    @property
    def key(self) -> str:
        """``name:version`` - the registry's primary key."""
        return f"{self.model_name}:{self.model_version}"

    def is_compatible_with(self, *, device: str, index_version: int | None = None) -> bool:
        """Whether this artifact may be loaded in the given environment.

        Args:
            device: The concrete device resolved for this host.
            index_version: The vector index version available, if any.
        """
        device_ok = (
            self.supported_device is SupportedDevice.ANY or self.supported_device.value == device
        )
        index_ok = (
            self.required_index_version is None or index_version == self.required_index_version
        )
        return device_ok and index_ok


def detect_git_commit(repo_root: Path | str | None = None) -> str | None:
    """Return the current commit SHA, or ``None`` outside a git checkout.

    Never raises: an artifact built from a tarball is legitimate, it just has no
    commit to record.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def detect_framework_versions(
    packages: tuple[str, ...] = ("torch", "lightgbm", "faiss"),
) -> dict[str, str]:
    """Report installed versions of the ML frameworks that are actually present.

    Absent packages are omitted rather than recorded as ``"not installed"``, so
    the metadata says what *was* used and nothing more.
    """
    from importlib.metadata import PackageNotFoundError, version

    found: dict[str, str] = {}
    for package in packages:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            continue
    return found


def build_metadata(
    *,
    model_name: str,
    model_version: str,
    model_type: ArtifactType,
    training_data_version: str,
    feature_version: str,
    configuration_hash: str,
    random_seed: int,
    supported_device: SupportedDevice = SupportedDevice.ANY,
    required_index_version: int | None = None,
    metrics: dict[str, float] | None = None,
    repo_root: Path | str | None = None,
    **extra: Any,
) -> ArtifactMetadata:
    """Construct metadata, filling in everything detectable from the environment.

    Training code calls this rather than instantiating :class:`ArtifactMetadata`
    directly, so that the environment fields cannot be forgotten or faked.
    """
    return ArtifactMetadata(
        model_name=model_name,
        model_version=model_version,
        model_type=model_type,
        created_at=datetime.now(tz=UTC),
        training_data_version=training_data_version,
        feature_version=feature_version,
        configuration_hash=configuration_hash,
        random_seed=random_seed,
        framework_version=detect_framework_versions(),
        python_version=platform.python_version() or sys.version.split()[0],
        git_commit=detect_git_commit(repo_root),
        metrics=metrics or {},
        supported_device=supported_device,
        required_index_version=required_index_version,
        **extra,
    )


__all__ = [
    "METADATA_FORMAT_VERSION",
    "ArtifactMetadata",
    "ArtifactType",
    "SupportedDevice",
    "build_metadata",
    "detect_framework_versions",
    "detect_git_commit",
]
