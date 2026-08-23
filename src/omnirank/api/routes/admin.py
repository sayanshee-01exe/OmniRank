"""Administrative endpoints - declared contract, no hot-swap yet.

Reload semantics are specified now (atomic swap, dry-run, per-artifact report)
because they constrain how the serving pipeline may hold model references: a
handler that captures a model object at import time can never be hot-swapped.
Writing this contract before the pipeline exists is what keeps that mistake out.

Authentication is out of scope for Phase 1 and the endpoint is unimplemented, so
nothing is currently exposed. Before this route does anything, it needs an
auth dependency - it is the one endpoint that changes server behaviour.
"""

from __future__ import annotations

from fastapi import APIRouter

from omnirank.api.dependencies.context import ContextDep
from omnirank.api.schemas.admin import ReloadArtifactsRequest, ReloadArtifactsResponse
from omnirank.api.schemas.common import NotImplementedDetail
from omnirank.core.exceptions import NotImplementedYetError

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.post(
    "/reload-artifacts",
    response_model=ReloadArtifactsResponse,
    responses={
        501: {
            "model": NotImplementedDetail,
            "description": "Contract is defined; hot reload is not built yet.",
        }
    },
    summary="Reload model artifacts without restarting",
    description=(
        "Re-reads the artifact registry and atomically swaps the serving model set. "
        "The new set is fully loaded and compatibility-checked before it replaces "
        "the live one; a failure leaves the previous set serving. `dry_run` reports "
        "what would change without swapping.\n\n"
        "**Not implemented in Phase 1.** Returns 501. Requires an authentication "
        "dependency before it is enabled."
    ),
)
def reload_artifacts(
    payload: ReloadArtifactsRequest,
    context: ContextDep,
) -> ReloadArtifactsResponse:
    """Contract for artifact hot-reload. Raises 501 until Phase 5."""
    raise NotImplementedYetError(
        feature="POST /v1/admin/reload-artifacts",
        phase=5,
        message=(
            "Hot reload requires a live serving model set to swap. Phase 1 loads no "
            "models. This endpoint also requires authentication before being enabled."
        ),
    )


__all__ = ["router"]
