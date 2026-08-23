"""Interaction ingestion endpoint - declared contract, no persistence yet.

The schema is complete and the validation semantics are specified (partial
acceptance, idempotent duplicates, per-event rejection reasons). What is missing
is the write path, which needs the PostgreSQL repository from Phase 2.

Accepting events into a black hole would be worse than refusing them: a client
would believe its data was recorded.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from omnirank.api.dependencies.context import ContextDep
from omnirank.api.schemas.common import NotImplementedDetail
from omnirank.api.schemas.interactions import (
    InteractionBatchRequest,
    InteractionBatchResponse,
)
from omnirank.core.exceptions import NotImplementedYetError

router = APIRouter(prefix="/v1", tags=["interactions"])


@router.post(
    "/interactions",
    response_model=InteractionBatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        501: {
            "model": NotImplementedDetail,
            "description": "Contract is defined; the write path is not built yet.",
        }
    },
    summary="Submit a batch of interaction events",
    description=(
        "Validates and appends events to the interaction log. Partial success is "
        "normal: valid events are accepted and invalid ones are reported per index "
        "with a rule identifier. Re-sending an event with the same `interaction_id` "
        "is idempotent.\n\n**Not implemented in Phase 1.** Returns 501."
    ),
)
def ingest_interactions(
    payload: InteractionBatchRequest,
    context: ContextDep,
) -> InteractionBatchResponse:
    """Contract for interaction ingestion. Raises 501 until Phase 2."""
    raise NotImplementedYetError(
        feature="POST /v1/interactions",
        phase=2,
        message=(
            "Interaction ingestion requires the PostgreSQL repository. The schema "
            "exists (src/omnirank/database/schema.sql) but no client is wired yet."
        ),
    )


__all__ = ["router"]
