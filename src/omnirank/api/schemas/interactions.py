"""Interaction ingestion contracts - component 17's HTTP surface."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from omnirank.api.schemas.common import ApiModel
from omnirank.api.schemas.recommendations import IdField
from omnirank.data.schemas import EventType


class InteractionEvent(ApiModel):
    """One event submitted by a client.

    Mirrors :class:`~omnirank.data.schemas.Interaction` but with two differences
    appropriate to an ingress boundary: ``interaction_id`` is optional (the
    server assigns one when the client has no idempotency key), and
    ``timestamp`` defaults to server receipt time.
    """

    user_id: IdField
    item_id: IdField
    event_type: EventType
    timestamp: datetime | None = Field(
        default=None,
        description="Event time, UTC. Defaults to server receipt time when omitted.",
    )
    session_id: IdField | None = None
    event_value: float | None = Field(
        default=None, description="Rating value, order value, or dwell time."
    )
    # Client-supplied idempotency key. Re-sending the same id is a no-op, which
    # is what makes at-least-once delivery from a client safe.
    interaction_id: IdField | None = None
    # Set when this event followed a recommendation, enabling attribution.
    request_id: str | None = None


class InteractionBatchRequest(ApiModel):
    """Body for ``POST /v1/interactions``.

    Batched because clients buffer events, and one request per click would make
    ingestion the most expensive endpoint in the system.
    """

    events: list[InteractionEvent] = Field(min_length=1, max_length=500)


class InteractionRejection(ApiModel):
    """One event that was not accepted, and why."""

    index: int = Field(ge=0, description="Position in the submitted batch.")
    rule: str = Field(description="Validation rule identifier, e.g. 'unknown_event_type'.")
    message: str


class InteractionBatchResponse(ApiModel):
    """Per-batch ingestion outcome.

    Partial success is the norm: a batch with two bad rows out of fifty is
    accepted for the forty-eight, and the rejections are reported rather than
    silently dropped. A client can then fix and resend only what failed.
    """

    accepted: int = Field(ge=0)
    rejected: int = Field(ge=0)
    duplicates: int = Field(
        ge=0, description="Events already recorded; ignored idempotently, not an error."
    )
    rejections: list[InteractionRejection] = Field(default_factory=list)
    request_id: str | None = None


__all__ = [
    "InteractionBatchRequest",
    "InteractionBatchResponse",
    "InteractionEvent",
    "InteractionRejection",
]
