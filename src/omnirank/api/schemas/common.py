"""Shared API response primitives."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    """Base for every API payload: strict in, stable out."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ErrorDetail(ApiModel):
    """Machine-readable error body.

    ``code`` comes from the raised :class:`~omnirank.core.exceptions.OmniRankError`
    subclass and is stable across releases, so clients and alerting can branch on
    it without parsing prose.
    """

    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable explanation.")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured detail. Never contains secrets or raw user attributes.",
    )
    request_id: str | None = Field(
        default=None, description="Correlates this response with server logs."
    )


class ErrorResponse(ApiModel):
    """Envelope returned for every non-2xx response."""

    error: ErrorDetail


class NotImplementedDetail(ApiModel):
    """Body of a 501, naming the phase that will deliver the endpoint.

    A 501 here is a *contract that exists but has no implementation yet*, not a
    bug and not a permanent absence. Returning this rather than fabricating
    plausible-looking recommendations is deliberate: a client that receives
    invented data cannot tell it is invented.
    """

    code: str = "not_implemented_yet"
    message: str
    feature: str
    planned_phase: int
    documentation: str = Field(
        default="docs/api/api_contracts.md",
        description="Where the full contract for this endpoint is specified.",
    )


__all__ = ["ApiModel", "ErrorDetail", "ErrorResponse", "NotImplementedDetail"]
