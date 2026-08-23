"""Pydantic request/response contracts for every endpoint.

Every schema here is complete and validated. Whether the endpoint behind it is
implemented is a separate question, answered per-route in
``docs/api/api_contracts.md`` and by the 501 responses the unimplemented ones
return.
"""

from __future__ import annotations

from omnirank.api.schemas.admin import (
    ReloadArtifactsRequest,
    ReloadArtifactsResponse,
    ReloadedArtifact,
)
from omnirank.api.schemas.common import (
    ApiModel,
    ErrorDetail,
    ErrorResponse,
    NotImplementedDetail,
)
from omnirank.api.schemas.health import (
    DependencyStatus,
    HealthResponse,
    ReadinessResponse,
)
from omnirank.api.schemas.interactions import (
    InteractionBatchRequest,
    InteractionBatchResponse,
    InteractionEvent,
    InteractionRejection,
)
from omnirank.api.schemas.items import ItemResponse
from omnirank.api.schemas.models import ModelListResponse, ModelSummary
from omnirank.api.schemas.recommendations import (
    RecommendationItem,
    RecommendationResponse,
    SessionRecommendationRequest,
    SimilarItemsQuery,
    UserRecommendationQuery,
)

__all__ = [
    "ApiModel",
    "DependencyStatus",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "InteractionBatchRequest",
    "InteractionBatchResponse",
    "InteractionEvent",
    "InteractionRejection",
    "ItemResponse",
    "ModelListResponse",
    "ModelSummary",
    "NotImplementedDetail",
    "ReadinessResponse",
    "RecommendationItem",
    "RecommendationResponse",
    "ReloadArtifactsRequest",
    "ReloadArtifactsResponse",
    "ReloadedArtifact",
    "SessionRecommendationRequest",
    "SimilarItemsQuery",
    "UserRecommendationQuery",
]
