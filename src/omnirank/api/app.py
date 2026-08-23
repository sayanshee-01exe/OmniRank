"""FastAPI application factory - component 16's entry point.

``create_app`` takes an explicit configuration, which is what makes the whole
API testable: a test builds an app against a temporary artifact root and a
synthetic config without touching the environment or the real filesystem.

There is exactly one application. OmniRank is a modular monolith (ADR-001): the
retrieval, ranking, and serving components are separate *modules* with explicit
contracts, deployed as one process. Splitting them into services buys network
hops and distributed failure modes in exchange for independent scaling that a
single-machine system does not need.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from omnirank import __version__
from omnirank.api.dependencies.context import build_context
from omnirank.api.errors import register_error_handlers
from omnirank.api.middleware import RequestContextMiddleware
from omnirank.api.routes import admin, health, interactions, items, models, recommendations
from omnirank.core.config import AppConfig, load_config
from omnirank.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

DESCRIPTION = """
Multi-stage, multimodal recommendation API.

**Implementation status (Phase 1).** `GET /health` and `GET /ready` answer from
real service state, and `GET /v1/models` reads the real artifact registry.
Every other endpoint is a *defined contract with no implementation* and returns
**501** naming the phase that will deliver it. No endpoint returns fabricated
recommendations.

Full contracts: `docs/api/api_contracts.md`.
"""


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Build the application.

    Args:
        config: Validated configuration. Loaded from ``configs/`` when omitted.

    Returns:
        A configured :class:`fastapi.FastAPI` instance with the application
        context attached to ``app.state.context``.

    Raises:
        ConfigurationError: Configuration is missing or invalid. Deliberately
            fatal: a service that starts with a broken config fails later, in a
            request, where the cause is much harder to see.
    """
    settings = config or load_config()
    configure_logging(settings.logging, force=True)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Build the context on startup; log a clean shutdown."""
        application.state.context = build_context(settings)
        logger.info(
            "api.startup",
            version=__version__,
            environment=settings.environment,
            serving_ready=application.state.context.serving_ready(),
        )
        yield
        logger.info("api.shutdown")

    app = FastAPI(
        title=settings.api.title,
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        # Keep the interactive docs on: the OpenAPI document is the deliverable
        # a Phase 1 API contract is judged by.
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(RequestContextMiddleware, log_requests=settings.api.log_requests)
    register_error_handlers(app)

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(recommendations.router)
    app.include_router(interactions.router)
    app.include_router(items.router)
    app.include_router(admin.router)

    return app


__all__ = ["create_app"]
