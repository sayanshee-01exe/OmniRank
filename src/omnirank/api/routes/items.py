"""Item lookup endpoint - declared contract, no catalogue store yet."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path

from omnirank.api.dependencies.context import ContextDep
from omnirank.api.schemas.common import NotImplementedDetail
from omnirank.api.schemas.items import ItemResponse
from omnirank.core.exceptions import NotImplementedYetError

router = APIRouter(prefix="/v1", tags=["items"])


@router.get(
    "/items/{item_id}",
    response_model=ItemResponse,
    responses={
        404: {"description": "No such item."},
        501: {
            "model": NotImplementedDetail,
            "description": "Contract is defined; the catalogue store is not wired yet.",
        },
    },
    summary="Fetch one catalogue item",
    description=(
        "Hydrates an item id - typically one returned by a recommendation - into "
        "displayable fields.\n\n**Not implemented in Phase 1.** Returns 501."
    ),
)
def get_item(
    item_id: Annotated[str, Path(min_length=1, max_length=128)],
    context: ContextDep,
) -> ItemResponse:
    """Contract for item lookup. Raises 501 until Phase 2."""
    raise NotImplementedYetError(
        feature="GET /v1/items/{item_id}",
        phase=2,
        message=(
            "Item lookup requires the catalogue repository backed by PostgreSQL, "
            "which is defined but not implemented."
        ),
    )


__all__ = ["router"]
