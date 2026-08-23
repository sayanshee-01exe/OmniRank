"""Health and readiness payloads.

The distinction matters operationally:

* ``/health`` - is this process alive? Answered from in-process state only, with
  no dependency checks, so a Redis outage does not get the container killed by a
  liveness probe.
* ``/ready`` - can this process serve useful traffic? Answered by checking that
  the artifacts needed to produce recommendations are actually loadable. A
  process with an empty registry is alive but must not receive traffic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from omnirank.api.schemas.common import ApiModel


class HealthResponse(ApiModel):
    """Liveness. Cheap, dependency-free, always answers."""

    status: Literal["ok"] = "ok"
    service: str = Field(description="Service name from configuration.")
    version: str = Field(description="Package version.")
    environment: str = Field(description="local | ci | staging | production.")
    uptime_seconds: float = Field(ge=0.0, description="Seconds since app startup.")
    timestamp: datetime


class DependencyStatus(ApiModel):
    """Readiness verdict for one dependency."""

    name: str
    ready: bool
    # Why it is not ready, or a short note when it is. Never contains a DSN,
    # password, or host credential.
    detail: str | None = None
    # False for dependencies whose absence degrades but does not disable serving
    # (e.g. the cache). Only required dependencies can make the service unready.
    required: bool = True


class ReadinessResponse(ApiModel):
    """Aggregate readiness. Served with HTTP 503 when ``ready`` is false."""

    ready: bool
    service: str
    version: str
    environment: str
    device: str = Field(description="Compute device resolved for this host.")
    dependencies: list[DependencyStatus]
    timestamp: datetime


__all__ = ["DependencyStatus", "HealthResponse", "ReadinessResponse"]
