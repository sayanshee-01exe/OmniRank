"""Artifact metadata contract and the filesystem registry."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from omnirank.artifacts.metadata import (
    ArtifactMetadata,
    ArtifactType,
    SupportedDevice,
    build_metadata,
    detect_framework_versions,
    detect_git_commit,
)
from omnirank.core.exceptions import (
    ArtifactCompatibilityError,
    ArtifactNotFoundError,
    ArtifactValidationError,
)
from tests.conftest import FROZEN_NOW

REQUIRED_FIELDS = {
    "model_name",
    "model_version",
    "model_type",
    "created_at",
    "training_data_version",
    "feature_version",
    "configuration_hash",
    "random_seed",
    "framework_version",
    "python_version",
    "metrics",
    "supported_device",
    "required_index_version",
    "git_commit",
}


class TestMetadataContract:
    def test_every_mandated_field_exists(self):
        assert set(ArtifactMetadata.model_fields) >= REQUIRED_FIELDS

    def test_valid_metadata(self, sample_metadata):
        assert sample_metadata.key == "popularity:v1"

    def test_unknown_field_is_rejected(self, sample_metadata):
        with pytest.raises(ValidationError):
            ArtifactMetadata.model_validate({**sample_metadata.model_dump(mode="json"), "extra": 1})

    def test_naive_created_at_is_rejected(self, sample_metadata):
        payload = sample_metadata.model_dump()
        payload["created_at"] = datetime(2026, 1, 1)
        with pytest.raises(ValidationError):
            ArtifactMetadata.model_validate(payload)

    def test_nan_metric_is_rejected(self, sample_metadata):
        payload = sample_metadata.model_dump()
        payload["metrics"] = {"ndcg@10": float("nan")}
        with pytest.raises(ValidationError) as exc:
            ArtifactMetadata.model_validate(payload)
        assert "NaN" in str(exc.value)

    def test_negative_seed_is_rejected(self, sample_metadata):
        payload = sample_metadata.model_dump()
        payload["random_seed"] = -1
        with pytest.raises(ValidationError):
            ArtifactMetadata.model_validate(payload)

    def test_metadata_is_immutable(self, sample_metadata):
        with pytest.raises(ValidationError):
            sample_metadata.model_version = "v2"

    @pytest.mark.parametrize(
        "artifact_type",
        [ArtifactType.RETRIEVAL_MODEL, ArtifactType.EMBEDDING, ArtifactType.INDEX],
    )
    def test_retrieval_artifacts_must_declare_an_index_version(
        self, sample_metadata, artifact_type
    ):
        payload = sample_metadata.model_dump()
        payload["model_type"] = artifact_type
        payload["required_index_version"] = None
        with pytest.raises(ValidationError) as exc:
            ArtifactMetadata.model_validate(payload)
        assert "required_index_version" in str(exc.value)

    def test_non_retrieval_artifacts_need_no_index_version(self, sample_metadata):
        assert sample_metadata.required_index_version is None


class TestCompatibility:
    def test_device_agnostic_artifact_runs_anywhere(self, sample_metadata):
        assert sample_metadata.is_compatible_with(device="cpu")
        assert sample_metadata.is_compatible_with(device="mps")

    def test_device_specific_artifact_is_refused_elsewhere(self, sample_metadata):
        payload = sample_metadata.model_dump()
        payload["supported_device"] = SupportedDevice.CUDA
        cuda_only = ArtifactMetadata.model_validate(payload)
        assert not cuda_only.is_compatible_with(device="mps")
        assert cuda_only.is_compatible_with(device="cuda")

    def test_index_version_must_match(self, sample_metadata):
        payload = sample_metadata.model_dump()
        payload["model_type"] = ArtifactType.EMBEDDING
        payload["required_index_version"] = 2
        embedding = ArtifactMetadata.model_validate(payload)
        assert embedding.is_compatible_with(device="cpu", index_version=2)
        assert not embedding.is_compatible_with(device="cpu", index_version=1)
        assert not embedding.is_compatible_with(device="cpu", index_version=None)


class TestBuildMetadata:
    def test_fills_environment_fields(self):
        metadata = build_metadata(
            model_name="m",
            model_version="v1",
            model_type=ArtifactType.MAPPING,
            training_data_version="d@v0",
            feature_version="f1",
            configuration_hash="b" * 64,
            random_seed=0,
        )
        assert metadata.python_version.startswith("3.11")
        assert metadata.created_at.tzinfo is UTC

    def test_framework_versions_omit_absent_packages(self):
        # Phase 1 installs none of these, so the dict must be empty rather than
        # claiming a version it does not have.
        assert detect_framework_versions(("definitely_not_installed_xyz",)) == {}

    def test_git_commit_detection_never_raises(self, tmp_path):
        assert detect_git_commit(tmp_path) is None or isinstance(detect_git_commit(tmp_path), str)


class TestRegistry:
    def test_empty_registry_reports_nothing(self, registry):
        assert registry.list_all() == []
        assert registry.is_ready() is False
        assert registry.versions("anything") == []

    def test_register_then_get(self, registry, sample_metadata):
        registry.register(sample_metadata)
        loaded = registry.get("popularity", "v1")
        assert loaded == sample_metadata

    def test_registered_manifest_is_human_readable_json(self, registry, sample_metadata):
        path = registry.register(sample_metadata)
        payload = json.loads(path.read_text())
        assert payload["model_name"] == "popularity"

    def test_duplicate_version_is_refused(self, registry, sample_metadata):
        registry.register(sample_metadata)
        with pytest.raises(ArtifactValidationError) as exc:
            registry.register(sample_metadata)
        assert "already registered" in str(exc.value)

    def test_overwrite_is_explicit(self, registry, sample_metadata):
        registry.register(sample_metadata)
        registry.register(sample_metadata, overwrite=True)

    def test_missing_artifact_raises(self, registry):
        with pytest.raises(ArtifactNotFoundError):
            registry.get("absent", "v1")

    def test_latest_uses_creation_time_not_filename(self, registry, sample_metadata):
        payload = sample_metadata.model_dump()
        # "v10" sorts before "v2" lexicographically; created_at must win.
        payload.update(model_version="v10", created_at=FROZEN_NOW.replace(year=2027))
        newer = ArtifactMetadata.model_validate(payload)
        registry.register(sample_metadata)
        registry.register(newer)
        assert registry.latest("popularity").model_version == "v10"

    def test_latest_on_an_unknown_model_raises(self, registry):
        with pytest.raises(ArtifactNotFoundError):
            registry.latest("absent")

    def test_list_all_is_newest_first(self, registry, sample_metadata):
        payload = sample_metadata.model_dump()
        payload.update(model_version="v2", created_at=FROZEN_NOW.replace(year=2027))
        registry.register(sample_metadata)
        registry.register(ArtifactMetadata.model_validate(payload))
        assert [m.model_version for m in registry.list_all()] == ["v2", "v1"]

    def test_list_all_filters_by_type(self, registry, sample_metadata):
        registry.register(sample_metadata)
        assert registry.list_all(artifact_type=ArtifactType.RANKER)
        assert registry.list_all(artifact_type=ArtifactType.INDEX) == []

    def test_corrupt_manifest_is_skipped_not_fatal(self, registry, sample_metadata):
        registry.register(sample_metadata)
        broken = registry.metadata_root / "broken" / "v1.json"
        broken.parent.mkdir(parents=True)
        broken.write_text("{ not json")
        # The good one still lists; the bad one is logged and skipped.
        assert [m.key for m in registry.list_all()] == ["popularity:v1"]

    def test_is_ready_flips_once_something_is_registered(self, registry, sample_metadata):
        assert registry.is_ready() is False
        registry.register(sample_metadata)
        assert registry.is_ready() is True

    def test_require_compatible_passes_and_raises(self, registry, sample_metadata):
        assert registry.require_compatible(sample_metadata, device="cpu") is sample_metadata

        payload = sample_metadata.model_dump()
        payload["supported_device"] = SupportedDevice.CUDA
        cuda_only = ArtifactMetadata.model_validate(payload)
        with pytest.raises(ArtifactCompatibilityError) as exc:
            registry.require_compatible(cuda_only, device="mps")
        assert "ADR-006" in str(exc.value)

    def test_payload_path_requires_a_recorded_path(self, registry, sample_metadata):
        with pytest.raises(ArtifactValidationError):
            registry.payload_path(sample_metadata)

    def test_payload_path_resolves_against_the_artifact_root(self, registry, sample_metadata):
        payload = sample_metadata.model_dump()
        payload["artifact_path"] = "models/popularity/v1.pkl"
        metadata = ArtifactMetadata.model_validate(payload)
        assert (
            registry.payload_path(metadata) == registry.artifact_root / "models/popularity/v1.pkl"
        )
