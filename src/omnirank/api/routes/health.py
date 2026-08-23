"""Liveness and readiness endpoints - the two routes that are fully implemented.

Both answer from real state. ``/health`` reports process facts; ``/ready``
actually inspects the artifact registry and the resolved device, and returns 503
when the service could not serve a recommendation. On a fresh checkout that is
exactly what happens, because no model has been trained yet - and reporting
"ready" there would be the single most misleading thing this API could do.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from omnirank.api.dependencies.context import ContextDep
from omnirank.api.schemas.health import (
    DependencyStatus,
    HealthResponse,
    ReadinessResponse,
)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Reports that the process is running. Performs no dependency checks, so a "
        "degraded backing service cannot cause a liveness probe to restart a "
        "container that is working fine."
    ),
)
def health(context: ContextDep) -> HealthResponse:
    """Return process liveness."""
    return HealthResponse(
        service=context.config.project_name,
        version=context.version,
        environment=context.config.environment,
        uptime_seconds=round(context.uptime_seconds, 3),
        timestamp=context.now,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    responses={503: {"description": "The service cannot serve recommendations yet."}},
    description=(
        "Reports whether this process can serve recommendation traffic. Returns 503 "
        "with a per-dependency breakdown when it cannot."
    ),
)
def ready(context: ContextDep, response: Response) -> ReadinessResponse:
    """Return readiness, setting 503 when the service cannot serve."""
    dependencies: list[DependencyStatus] = []

    # 1. Compute device. Required: no model loads without one.
    dependencies.append(
        DependencyStatus(
            name="device",
            ready=context.device_error is None,
            detail=context.device_error or f"resolved to {context.device}",
            required=True,
        )
    )

    # 2. Artifact registry. Required: an empty registry means nothing to serve.
    compatible = context.compatible_artifacts()
    total = len(context.registry.list_all())
    if compatible:
        registry_detail = f"{len(compatible)} of {total} registered artifacts usable on this host"
    elif total:
        registry_detail = (
            f"{total} artifacts registered, none compatible with device "
            f"{context.device!r} and index version {context.available_index_version()}"
        )
    else:
        registry_detail = (
            "no artifacts registered; train and register a model before serving "
            "(Phase 1 ships no trained models)"
        )
    dependencies.append(
        DependencyStatus(
            name="artifact_registry",
            ready=bool(compatible),
            detail=registry_detail,
            required=True,
        )
    )

    # 3. Configuration. Reaching this line means it validated at startup.
    dependencies.append(
        DependencyStatus(
            name="configuration",
            ready=True,
            detail=f"loaded for environment {context.config.environment!r}",
            required=True,
        )
    )

    # PostgreSQL and Redis are intentionally absent: no client is wired in Phase 1,
    # and reporting an unchecked dependency as "ready" would be a false claim.
    # They join this list in Phase 2 alongside their clients.

    is_ready = all(dep.ready for dep in dependencies if dep.required)
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        ready=is_ready,
        service=context.config.project_name,
        version=context.version,
        environment=context.config.environment,
        device=context.device,
        dependencies=dependencies,
        timestamp=context.now,
    )


__all__ = ["router"]
