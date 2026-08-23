"""Model-registry payloads for ``GET /v1/models``."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from omnirank.api.schemas.common import ApiModel
from omnirank.artifacts.metadata import ArtifactMetadata


class ModelSummary(ApiModel):
    """One registered artifact, as exposed over HTTP.

    A projection of :class:`~omnirank.artifacts.metadata.ArtifactMetadata`
    rather than the whole record: filesystem paths and id-mapping fingerprints
    are operational detail that clients have no use for.
    """

    model_name: str
    model_version: str
    model_type: str
    created_at: datetime
    training_data_version: str
    feature_version: str
    supported_device: str
    required_index_version: int | None = None
    # Offline metrics as recorded at export time. Empty when the artifact type
    # has none - it is never populated with placeholder numbers.
    metrics: dict[str, float] = Field(default_factory=dict)
    # Whether this artifact can be loaded on this host right now.
    compatible: bool = True
    incompatibility_reason: str | None = None

    @classmethod
    def from_metadata(
        cls,
        metadata: ArtifactMetadata,
        *,
        device: str,
        index_version: int | None = None,
    ) -> ModelSummary:
        """Project registry metadata, evaluating host compatibility."""
        compatible = metadata.is_compatible_with(device=device, index_version=index_version)
        reason: str | None = None
        if not compatible:
            if (
                metadata.supported_device.value != "any"
                and metadata.supported_device.value != device
            ):
                reason = (
                    f"artifact requires device {metadata.supported_device.value!r}, "
                    f"host resolved to {device!r}"
                )
            else:
                reason = (
                    f"artifact requires index version {metadata.required_index_version}, "
                    f"available: {index_version}"
                )
        return cls(
            model_name=metadata.model_name,
            model_version=metadata.model_version,
            model_type=metadata.model_type.value,
            created_at=metadata.created_at,
            training_data_version=metadata.training_data_version,
            feature_version=metadata.feature_version,
            supported_device=metadata.supported_device.value,
            required_index_version=metadata.required_index_version,
            metrics=metadata.metrics,
            compatible=compatible,
            incompatibility_reason=reason,
        )


class ModelListResponse(ApiModel):
    """Everything the registry currently holds."""

    models: list[ModelSummary] = Field(default_factory=list)
    count: int = 0
    device: str = Field(description="Compute device resolved for this host.")
    # True when at least one compatible model exists; mirrors /ready.
    serving_ready: bool = False


__all__ = ["ModelListResponse", "ModelSummary"]
