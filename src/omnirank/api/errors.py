"""Exception handlers mapping project errors onto HTTP responses.

Registered once in :func:`omnirank.api.app.create_app`. Two properties matter:

* **Every error response has the same shape** (:class:`ErrorResponse`), so a
  client needs one error-parsing path rather than one per failure mode.
* **Nothing leaks.** Unexpected exceptions are logged with a traceback and
  answered with a generic 500 carrying only the request id. The client gets
  enough to report the problem; the details stay server-side.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from omnirank.api.schemas.common import ErrorDetail, ErrorResponse
from omnirank.core.exceptions import (
    ArtifactCompatibilityError,
    ArtifactNotFoundError,
    ConfigurationError,
    NotImplementedYetError,
    OmniRankError,
    SchemaValidationError,
    ServiceNotReadyError,
)
from omnirank.core.logging import get_logger

logger = get_logger(__name__)

# Which HTTP status each deliberate error maps to. Anything not listed becomes
# a 500, which is the correct default for an error we did not plan for.
_STATUS_BY_TYPE: tuple[tuple[type[OmniRankError], int], ...] = (
    (NotImplementedYetError, status.HTTP_501_NOT_IMPLEMENTED),
    (ArtifactNotFoundError, status.HTTP_404_NOT_FOUND),
    # 422 written as a literal: Starlette renamed its constant, and pinning to
    # either spelling breaks on the other side of that rename.
    (SchemaValidationError, 422),
    (ServiceNotReadyError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (ArtifactCompatibilityError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (ConfigurationError, status.HTTP_500_INTERNAL_SERVER_ERROR),
)


def status_for(error: OmniRankError) -> int:
    """Return the HTTP status for a project error."""
    for error_type, code in _STATUS_BY_TYPE:
        if isinstance(error, error_type):
            return code
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


async def omnirank_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Render an :class:`OmniRankError` as a structured error response."""
    assert isinstance(exc, OmniRankError)  # noqa: S101 - handler is registered per-type
    http_status = status_for(exc)
    request_id = _request_id(request)

    # 5xx is our fault and gets a stack trace; 4xx is the caller's and does not.
    log = logger.error if http_status >= 500 else logger.warning
    log(
        "api.error",
        error_code=exc.code,
        http_status=http_status,
        path=request.url.path,
        request_id=request_id,
        **exc.context,
    )

    payload = ErrorResponse(
        error=ErrorDetail(
            code=exc.code, message=exc.message, context=exc.context, request_id=request_id
        )
    )
    return JSONResponse(status_code=http_status, content=payload.model_dump(mode="json"))


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: log fully, disclose nothing."""
    request_id = _request_id(request)
    logger.exception(
        "api.unhandled_error",
        path=request.url.path,
        request_id=request_id,
        error_type=type(exc).__name__,
    )
    payload = ErrorResponse(
        error=ErrorDetail(
            code="internal_error",
            message="An unexpected error occurred. Quote the request_id when reporting it.",
            request_id=request_id,
        )
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload.model_dump(mode="json")
    )


def register_error_handlers(app: FastAPI) -> None:
    """Attach both handlers to the application."""
    app.add_exception_handler(OmniRankError, omnirank_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)


__all__ = [
    "omnirank_error_handler",
    "register_error_handlers",
    "status_for",
    "unhandled_error_handler",
]
