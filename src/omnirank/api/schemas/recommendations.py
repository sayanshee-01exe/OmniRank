"""Recommendation request and response contracts.

The response shape is the most important contract in the project: it is the one
thing every client depends on, and changing it later is expensive. Three
decisions are baked in deliberately.

**``sources`` is a list.** An item can be nominated by several generators, and
knowing which ones is what makes a result debuggable after the fact. Collapsing
to a single string loses that permanently.

**``fallback_used`` is mandatory, not optional.** A degraded response that looks
identical to a healthy one is how quality regressions go unnoticed for weeks.
The flag is always present, and clients are expected to log it.

**``reason`` is nullable.** A reason is emitted only when the pipeline can
actually justify the item from its sources and features. There is no default
string, because a fabricated explanation is worse than none.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints

from omnirank.api.schemas.common import ApiModel

IdField = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]


class RecommendationItem(ApiModel):
    """One recommended item."""

    item_id: IdField
    rank: int = Field(ge=1, description="1-based position in the returned list.")
    score: float = Field(description="Final relevance score after ranking and reranking.")
    sources: list[str] = Field(
        default_factory=list,
        description="Candidate generators that nominated this item, e.g. ['lightgcn', 'sasrec'].",
    )
    reason: str | None = Field(
        default=None,
        description=(
            "Human-readable justification, derived from sources and features. "
            "Null when the pipeline cannot justify the item; never a placeholder."
        ),
    )


class RecommendationResponse(ApiModel):
    """The canonical recommendation payload."""

    user_id: str | None = Field(
        default=None, description="Null for anonymous session-based requests."
    )
    model_version: str = Field(
        description="Version of the serving pipeline that produced this response."
    )
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    fallback_used: bool = Field(
        description="True when one or more fallback stages produced this response."
    )
    fallback_stage: str | None = Field(
        default=None,
        description="Which fallback stage answered, e.g. 'global_popularity'. Null when unused.",
    )
    latency_ms: int = Field(ge=0, description="Server-side wall time for the whole pipeline.")
    request_id: str | None = Field(
        default=None, description="Correlates this response with server logs."
    )
    generated_at: datetime | None = None

    model_config = ApiModel.model_config | {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user_123",
                    "model_version": "v1",
                    "recommendations": [
                        {
                            "item_id": "item_456",
                            "rank": 1,
                            "score": 0.87,
                            "sources": ["lightgcn", "sasrec"],
                            "reason": "Recommended from your recent activity",
                        }
                    ],
                    "fallback_used": False,
                    "latency_ms": 42,
                }
            ]
        }
    }


class UserRecommendationQuery(ApiModel):
    """Query parameters for ``GET /v1/recommendations/users/{user_id}``."""

    k: int = Field(default=20, ge=1, le=200, description="How many items to return.")
    category: str | None = Field(
        default=None, description="Restrict candidates to a single category."
    )
    exclude_seen: bool = Field(
        default=True,
        description="Drop items the user has already interacted with.",
    )
    include_reasons: bool = Field(
        default=True, description="Populate the per-item `reason` field when available."
    )


class SimilarItemsQuery(ApiModel):
    """Query parameters for ``GET /v1/recommendations/similar/{item_id}``."""

    k: int = Field(default=20, ge=1, le=200)
    # Which embedding space to compare in. 'content' uses text/image embeddings
    # and works for cold items; 'collaborative' uses co-interaction and does not.
    space: Literal["content", "collaborative", "hybrid"] = "hybrid"


class SessionRecommendationRequest(ApiModel):
    """Body for ``POST /v1/recommendations/session``.

    Session recommendations exist for the anonymous and cold-start case: no
    ``user_id``, just what happened in the last few minutes. This is the request
    shape SASRec is built to serve.
    """

    session_id: IdField
    # Items seen this session, oldest first. The ordering is meaningful.
    item_ids: list[IdField] = Field(
        default_factory=list, max_length=200, description="Session history, oldest first."
    )
    user_id: IdField | None = Field(
        default=None, description="Present when the session belongs to a known user."
    )
    k: int = Field(default=20, ge=1, le=200)
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional request-time signals (locale, device, surface).",
    )


__all__ = [
    "RecommendationItem",
    "RecommendationResponse",
    "SessionRecommendationRequest",
    "SimilarItemsQuery",
    "UserRecommendationQuery",
]
