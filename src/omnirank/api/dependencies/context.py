"""Application context and FastAPI dependency providers.

One :class:`AppContext` is built at startup and stored on ``app.state``. Route
handlers receive it through a dependency rather than importing a module-level
singleton, which is what makes the whole API testable against an arbitrary
configuration without touching the environment.

Everything expensive to construct - configuration, the artifact registry, the
resolved device - is built once here. Nothing that talks to a network is built
at import time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Request

from omnirank import __version__
from omnirank.artifacts.registry import ArtifactRegistry
from omnirank.core.config import AppConfig
from omnirank.core.exceptions import UnsupportedDeviceError
from omnirank.core.logging import get_logger

if TYPE_CHECKING:
    from omnirank.artifacts.metadata import ArtifactMetadata

logger = get_logger(__name__)


@dataclass(slots=True)
class AppContext:
    """Long-lived state shared by every request."""

    config: AppConfig
    registry: ArtifactRegistry
    # Concrete device this host resolved to, or a description of why it could
    # not be resolved. Never raises at startup: an unusable device makes the
    # service *not ready*, which is recoverable, rather than crash-looping.
    device: str
    device_error: str | None = None
    started_at: float = field(default_factory=time.monotonic)
    version: str = __version__

    @property
    def uptime_seconds(self) -> float:
        """Seconds since this context was created."""
        return max(0.0, time.monotonic() - self.started_at)

    @property
    def now(self) -> datetime:
        """Current UTC time. Centralised so tests can substitute a clock."""
        return datetime.now(tz=UTC)

    def available_index_version(self) -> int | None:
        """Index build version this deployment can supply, from configuration."""
        return self.config.models.index.index_version

    def compatible_artifacts(self) -> list[ArtifactMetadata]:
        """Registered artifacts that could actually be loaded on this host."""
        index_version = self.available_index_version()
        return [
            metadata
            for metadata in self.registry.list_all()
            if metadata.is_compatible_with(device=self.device, index_version=index_version)
        ]

    def serving_ready(self) -> bool:
        """Whether the recommendation pipeline could serve a request.

        Phase 1 answer is honestly ``False`` on a fresh checkout: no artifacts
        are registered because no model has been trained. That is the correct
        readiness signal, not a bug to paper over.
        """
        if self.device_error is not None:
            return False
        return bool(self.compatible_artifacts())


def build_context(config: AppConfig) -> AppContext:
    """Construct the application context from a validated configuration."""
    paths = config.paths.resolved(Path.cwd())
    registry = ArtifactRegistry(
        metadata_root=paths["metadata_dir"], artifact_root=paths["artifact_root"]
    )

    device = "unavailable"
    device_error: str | None = None
    try:
        device = config.device.resolve()
    except UnsupportedDeviceError as exc:
        # Recorded, not raised: the service starts and reports itself unready.
        device_error = str(exc)
        logger.error("startup.device_unavailable", reason=device_error)

    context = AppContext(config=config, registry=registry, device=device, device_error=device_error)
    logger.info(
        "startup.context_built",
        environment=config.environment,
        device=device,
        artifacts_registered=len(registry.list_all()),
        serving_ready=context.serving_ready(),
    )
    return context


def get_context(request: Request) -> AppContext:
    """FastAPI dependency: the context stored on ``app.state`` at startup."""
    context = getattr(request.app.state, "context", None)
    if context is None:  # pragma: no cover - only reachable via manual app wiring
        raise RuntimeError(
            "Application context is missing. Build the app with "
            "omnirank.api.app.create_app() rather than instantiating FastAPI directly."
        )
    return context  # type: ignore[no-any-return]


def get_config_dependency(request: Request) -> AppConfig:
    """FastAPI dependency: the validated configuration."""
    return get_context(request).config


ContextDep = Annotated[AppContext, Depends(get_context)]
ConfigDep = Annotated[AppConfig, Depends(get_config_dependency)]

__all__ = [
    "AppContext",
    "ConfigDep",
    "ContextDep",
    "build_context",
    "get_config_dependency",
    "get_context",
]
