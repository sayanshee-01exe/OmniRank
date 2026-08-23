"""Project exception hierarchy.

Every error OmniRank raises on purpose derives from :class:`OmniRankError`, so
callers (the API error handler, CLI entrypoints, batch jobs) can distinguish
"this system said no, and here is why" from an unexpected crash.

Each exception carries a stable ``code`` used verbatim as the ``error.code``
field in API responses and as a log key, so alerting can key on it without
string-matching human-readable messages.
"""

from __future__ import annotations

from typing import Any


class OmniRankError(Exception):
    """Base class for all deliberate OmniRank failures."""

    code: str = "omnirank_error"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        # Structured detail attached to logs and API responses. Must never hold
        # secrets or raw user attributes - see core.logging redaction.
        self.context: dict[str, Any] = context

    def __str__(self) -> str:
        if not self.context:
            return self.message
        rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(self.context.items()))
        return f"{self.message} ({rendered})"

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-serialisable payload for API responses."""
        return {"code": self.code, "message": self.message, "context": self.context}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class ConfigurationError(OmniRankError):
    """Configuration is missing, malformed, or internally inconsistent.

    Raised during startup so the process dies immediately with an actionable
    message rather than failing later inside a request.
    """

    code = "configuration_error"


class ConfigFileNotFoundError(ConfigurationError):
    """A referenced YAML config file does not exist on disk."""

    code = "config_file_not_found"


class ConfigValidationError(ConfigurationError):
    """Merged configuration failed schema validation."""

    code = "config_validation_error"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class DataError(OmniRankError):
    """Base class for dataset-level problems."""

    code = "data_error"


class SchemaValidationError(DataError):
    """A record or batch violated a declared data contract."""

    code = "schema_validation_error"


class DataSourceError(DataError):
    """A raw data source is unreadable, missing, or in an unexpected format."""

    code = "data_source_error"


class IdMappingError(DataError):
    """A raw identifier could not be mapped to (or from) a dense index."""

    code = "id_mapping_error"


# --------------------------------------------------------------------------- #
# Artifacts and models
# --------------------------------------------------------------------------- #
class ArtifactError(OmniRankError):
    """Base class for artifact registry / loading problems."""

    code = "artifact_error"


class ArtifactNotFoundError(ArtifactError):
    """The requested artifact name/version is not registered."""

    code = "artifact_not_found"


class ArtifactValidationError(ArtifactError):
    """Artifact metadata is malformed or incomplete."""

    code = "artifact_validation_error"


class ArtifactCompatibilityError(ArtifactError):
    """An artifact is registered but cannot be used in this environment.

    Covers the two failure modes ADR-006 exists to prevent: a model paired with
    an index built by an incompatible procedure, and a model whose supported
    device is unavailable on this host.
    """

    code = "artifact_compatibility_error"


class ModelNotFittedError(OmniRankError):
    """``recommend``/``score``/``rank`` was called before ``fit`` or ``load``."""

    code = "model_not_fitted"


class VectorIndexError(OmniRankError):
    """A vector index could not be built, loaded, or queried."""

    code = "vector_index_error"


# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #
class ServingError(OmniRankError):
    """Base class for request-time failures."""

    code = "serving_error"


class NotImplementedYetError(ServingError):
    """A declared contract has no implementation in the current phase.

    Deliberately distinct from the builtin ``NotImplementedError``: this is a
    documented, expected state that maps to HTTP 501 with a pointer to the
    phase that will deliver it, not a bug.
    """

    code = "not_implemented_yet"

    def __init__(self, feature: str, phase: int, message: str | None = None) -> None:
        super().__init__(
            message or f"{feature!r} is a defined contract with no implementation yet.",
            feature=feature,
            planned_phase=phase,
        )
        self.feature = feature
        self.phase = phase


class ServiceNotReadyError(ServingError):
    """A dependency required to answer this request is not available."""

    code = "service_not_ready"


class FallbackExhaustedError(ServingError):
    """Every stage of the fallback chain failed.

    This is the only path on which the recommendation API may fail to return a
    non-empty payload, and it is therefore alert-worthy.
    """

    code = "fallback_exhausted"


class UnsupportedDeviceError(OmniRankError):
    """The requested compute device is unavailable on this host."""

    code = "unsupported_device"


__all__ = [
    "ArtifactCompatibilityError",
    "ArtifactError",
    "ArtifactNotFoundError",
    "ArtifactValidationError",
    "ConfigFileNotFoundError",
    "ConfigValidationError",
    "ConfigurationError",
    "DataError",
    "DataSourceError",
    "FallbackExhaustedError",
    "IdMappingError",
    "ModelNotFittedError",
    "NotImplementedYetError",
    "OmniRankError",
    "SchemaValidationError",
    "ServiceNotReadyError",
    "ServingError",
    "UnsupportedDeviceError",
    "VectorIndexError",
]
