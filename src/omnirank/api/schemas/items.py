"""Item lookup contracts for ``GET /v1/items/{item_id}``.

Exposed so a client that received a recommendation can hydrate it into
something displayable without a second system. Deliberately a projection of
:class:`~omnirank.data.schemas.Item`: internal ``attributes`` are not dumped
wholesale, because the catalogue's free-form field is where vertical-specific
and occasionally sensitive data ends up.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from omnirank.api.schemas.common import ApiModel
from omnirank.api.schemas.recommendations import IdField


class ItemResponse(ApiModel):
    """Public view of one catalogue item."""

    item_id: IdField
    title: str
    description: str | None = None
    category: str | None = None
    brand: str | None = None
    price: float | None = Field(default=None, ge=0.0)
    image_id: str | None = None
    available: bool
    created_at: datetime


__all__ = ["ItemResponse"]
