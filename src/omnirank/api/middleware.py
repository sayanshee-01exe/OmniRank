"""Request correlation and access logging.

Every request gets an identifier that is bound into the logging context, echoed
in the ``X-Request-ID`` response header, and included in error bodies. That one
value is what turns "a user reported a bad recommendation" into a retrievable
trace through validation, retrieval, ranking, and cache lookup.

What is deliberately *not* logged: query strings and request bodies. Both carry
user and item identifiers, and an access log is the easiest place to
accidentally build a permanent record of individual browsing behaviour. The path
template, status, and duration are enough to operate the service.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from omnirank.core.logging import bound_context, get_logger, new_run_id

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

CallNext = Callable[[Request], Awaitable[Response]]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, binds it to logs, and records latency."""

    def __init__(self, app: ASGIApp, *, log_requests: bool = True) -> None:
        super().__init__(app)
        self.log_requests = log_requests

    async def dispatch(self, request: Request, call_next: CallNext) -> Response:
        """Wrap one request."""
        # Honour a caller-supplied id so a trace can span several services,
        # but bound its length: it ends up in every log line.
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request_id = incoming[:64] if incoming else new_run_id()
        request.state.request_id = request_id

        started = time.perf_counter()
        with bound_context(request_id=request_id):
            try:
                response = await call_next(request)
            except Exception:
                # Logged by the exception handler; timing is still useful.
                elapsed_ms = (time.perf_counter() - started) * 1000
                logger.warning(
                    "api.request_failed",
                    method=request.method,
                    path=request.url.path,
                    duration_ms=round(elapsed_ms, 2),
                )
                raise

            elapsed_ms = (time.perf_counter() - started) * 1000
            response.headers[REQUEST_ID_HEADER] = request_id
            # Let handlers report the server-side cost they measured.
            response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.2f}"

            if self.log_requests:
                logger.info(
                    "api.request",
                    method=request.method,
                    # Route template, not the raw path: an id-bearing path would
                    # both explode log cardinality and record who asked for what.
                    path=_route_template(request),
                    status_code=response.status_code,
                    duration_ms=round(elapsed_ms, 2),
                )
            return response


def _route_template(request: Request) -> str:
    """Return the matched route pattern, falling back to the raw path."""
    route = request.scope.get("route")
    path_format = getattr(route, "path_format", None) or getattr(route, "path", None)
    return str(path_format) if path_format else request.url.path


__all__ = ["REQUEST_ID_HEADER", "RequestContextMiddleware"]
