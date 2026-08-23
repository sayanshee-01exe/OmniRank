"""Administrative contracts for ``POST /v1/admin/reload-artifacts``.

Reloading is how a newly trained model reaches a running server without a
restart. Two properties are specified up front because retrofitting them is
painful:

* **Atomic swap.** The new artifact set is fully loaded and compatibility-checked
  *before* it replaces the live one. A failed reload leaves the previous set
  serving, and says so in the response.
* **Reported, not assumed.** The response names exactly which artifacts changed,
  so a deploy pipeline can assert on it rather than trusting a 200.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from omnirank.api.schemas.common import ApiModel


class ReloadArtifactsRequest(ApiModel):
    """Body for a reload request."""

    # None reloads every model currently in the serving set.
    model_names: list[str] | None = Field(
        default=None, description="Restrict the reload to these models. Null means all."
    )
    # Force reload even when the registry reports the same versions already
    # loaded. Used after replacing a payload file in place.
    force: bool = False
    # Verify compatibility and report what *would* change, without swapping.
    dry_run: bool = False


class ReloadedArtifact(ApiModel):
    """One artifact's reload outcome."""

    model_name: str
    previous_version: str | None = None
    new_version: str | None = None
    status: str = Field(description="loaded | unchanged | failed | skipped")
    detail: str | None = None


class ReloadArtifactsResponse(ApiModel):
    """Aggregate reload outcome."""

    reloaded: list[ReloadedArtifact] = Field(default_factory=list)
    # False when any required artifact failed; the previous set stays live.
    success: bool = True
    dry_run: bool = False
    serving_ready: bool = False
    completed_at: datetime
    request_id: str | None = None


__all__ = ["ReloadArtifactsRequest", "ReloadArtifactsResponse", "ReloadedArtifact"]
