"""Recommendation endpoints - declared contracts, no implementation yet.

Each route below has a complete, validated request/response schema and appears
in the OpenAPI document, so a client can be written against it today. Each one
raises :class:`~omnirank.core.exceptions.NotImplementedYetError`, which the
error handler renders as HTTP 501 naming the phase that will deliver it.

This is a deliberate choice over the alternative of returning random or
hard-coded items. Fake recommendations are indistinguishable from real ones to
a caller, they get screenshotted into status updates, and they make integration
tests pass against a system that does not work. A 501 cannot be mistaken for
anything.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Path, Query

from omnirank.api.dependencies.context import ContextDep
from omnirank.api.schemas.common import NotImplementedDetail
from omnirank.api.schemas.recommendations import (
    RecommendationResponse,
    SessionRecommendationRequest,
)
from omnirank.core.exceptions import NotImplementedYetError

router = APIRouter(prefix="/v1/recommendations", tags=["recommendations"])

_NOT_IMPLEMENTED_RESPONSE: dict[int | str, dict[str, Any]] = {
    501: {
        "model": NotImplementedDetail,
        "description": "Contract is defined; the pipeline behind it is not built yet.",
    }
}

UserIdPath = Annotated[str, Path(min_length=1, max_length=128, description="Opaque user id.")]
ItemIdPath = Annotated[str, Path(min_length=1, max_length=128, description="Opaque item id.")]


@router.get(
    "/users/{user_id}",
    response_model=RecommendationResponse,
    responses=_NOT_IMPLEMENTED_RESPONSE,
    summary="Personalised recommendations for a user",
    description=(
        "Runs the full pipeline: candidate generation across every enabled generator, "
        "aggregation, ranking, post-ranking filters, and diversity reranking. Falls "
        "back to category and then global popularity if the models are unavailable, "
        "so a 200 response is always non-empty.\n\n"
        "**Not implemented in Phase 1.** Returns 501."
    ),
)
def recommend_for_user(
    user_id: UserIdPath,
    context: ContextDep,
    k: Annotated[int, Query(ge=1, le=200, description="Number of items to return.")] = 20,
    category: Annotated[str | None, Query(description="Restrict to one category.")] = None,
    exclude_seen: Annotated[bool, Query(description="Drop already-seen items.")] = True,
) -> RecommendationResponse:
    """Contract for user recommendations. Raises 501 until Phase 5."""
    raise NotImplementedYetError(
        feature="GET /v1/recommendations/users/{user_id}",
        phase=5,
        message=(
            "User recommendations require candidate generators, a ranker, and the "
            "fallback chain. None are built yet. See docs/api/api_contracts.md."
        ),
    )


@router.get(
    "/similar/{item_id}",
    response_model=RecommendationResponse,
    responses=_NOT_IMPLEMENTED_RESPONSE,
    summary="Items similar to a given item",
    description=(
        "Item-to-item retrieval from the vector index. The `space` parameter selects "
        "content embeddings (works for cold items), collaborative embeddings (does "
        "not), or a hybrid.\n\n**Not implemented in Phase 1.** Returns 501."
    ),
)
def similar_items(
    item_id: ItemIdPath,
    context: ContextDep,
    k: Annotated[int, Query(ge=1, le=200)] = 20,
    space: Annotated[str, Query(pattern="^(content|collaborative|hybrid)$")] = "hybrid",
) -> RecommendationResponse:
    """Contract for item-to-item similarity. Raises 501 until Phase 4."""
    raise NotImplementedYetError(
        feature="GET /v1/recommendations/similar/{item_id}",
        phase=4,
        message=(
            "Similar-item retrieval requires item embeddings and a built vector "
            "index. Neither exists yet. See docs/api/api_contracts.md."
        ),
    )


@router.post(
    "/session",
    response_model=RecommendationResponse,
    responses=_NOT_IMPLEMENTED_RESPONSE,
    summary="Recommendations from a live session",
    description=(
        "Session-based recommendations for anonymous or cold-start users, driven by "
        "the item sequence supplied in the request body rather than by stored "
        "history.\n\n**Not implemented in Phase 1.** Returns 501."
    ),
)
def recommend_for_session(
    payload: SessionRecommendationRequest,
    context: ContextDep,
) -> RecommendationResponse:
    """Contract for session recommendations. Raises 501 until Phase 5."""
    raise NotImplementedYetError(
        feature="POST /v1/recommendations/session",
        phase=5,
        message=(
            "Session recommendations require the sequential retrieval model "
            "(SASRec, Phase 3) and the serving pipeline (Phase 5)."
        ),
    )


__all__ = ["router"]
