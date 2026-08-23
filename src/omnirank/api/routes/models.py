"""Model registry endpoint - implemented, and honest about an empty registry.

Reads the real filesystem registry and reports what is there. On a fresh
checkout that is an empty list with ``serving_ready: false``, which is the
truthful answer.
"""

from __future__ import annotations

from fastapi import APIRouter

from omnirank.api.dependencies.context import ContextDep
from omnirank.api.schemas.models import ModelListResponse, ModelSummary

router = APIRouter(prefix="/v1", tags=["models"])


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List registered model artifacts",
    description=(
        "Returns every artifact in the local registry, newest first, each flagged "
        "with whether it can be loaded on this host's device and index version."
    ),
)
def list_models(context: ContextDep) -> ModelListResponse:
    """List registered artifacts with host-compatibility flags."""
    index_version = context.available_index_version()
    summaries = [
        ModelSummary.from_metadata(metadata, device=context.device, index_version=index_version)
        for metadata in context.registry.list_all()
    ]
    return ModelListResponse(
        models=summaries,
        count=len(summaries),
        device=context.device,
        serving_ready=context.serving_ready(),
    )


__all__ = ["router"]
