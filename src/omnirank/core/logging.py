"""Structured logging.

One call to :func:`configure_logging` at process start wires structlog and the
stdlib ``logging`` module together, so third-party libraries' log records flow
through the same processors and renderer as ours.

Two guarantees this module provides:

* **Correlation.** Every event emitted while handling a request or a batch run
  carries that run's identifier, bound via context variables. Nothing has to
  thread a logger through call signatures.
* **Redaction.** A processor drops the value of any key whose name matches the
  configured sensitive-key list, at any nesting depth, before rendering. This
  is a backstop, not a licence: do not log secrets in the first place.

Library modules never call ``print()``. They call ``get_logger(__name__)``.
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, unbind_contextvars

from omnirank.core.config import LoggingConfig

REDACTED = "***redacted***"

_configured = False


def _redactor(sensitive_keys: frozenset[str]) -> Any:
    """Build a structlog processor that masks sensitive values recursively."""

    def _mask(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: REDACTED if key.lower() in sensitive_keys else _mask(inner)
                for key, inner in value.items()
            }
        if isinstance(value, (list, tuple)):
            return type(value)(_mask(item) for item in value)
        return value

    def processor(
        _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        for key in list(event_dict):
            if key.lower() in sensitive_keys:
                event_dict[key] = REDACTED
            else:
                event_dict[key] = _mask(event_dict[key])
        return event_dict

    return processor


def configure_logging(config: LoggingConfig | None = None, *, force: bool = False) -> None:
    """Configure structlog and the stdlib root logger.

    Idempotent: repeated calls are no-ops unless ``force=True``. That matters
    because both the API app factory and CLI entrypoints call it, and reloading
    under uvicorn can import the app twice.

    Args:
        config: Logging settings. Defaults to :class:`LoggingConfig` defaults so
            that early failures (including config loading itself) can still log.
        force: Reconfigure even if already configured.
    """
    global _configured
    if _configured and not force:
        return

    settings = config or LoggingConfig()
    sensitive = frozenset(key.lower() for key in settings.redact_keys)

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
    ]
    if settings.include_timestamp:
        shared.append(structlog.processors.TimeStamper(fmt="iso", utc=True))
    shared.extend(
        [
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _redactor(sensitive),
        ]
    )

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.format == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Route stdlib records (uvicorn, sqlalchemy, ...) through the same pipeline.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(processor=renderer, foreign_pre_chain=shared)
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.level)

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger. Call with ``__name__`` from module scope."""
    if not _configured:
        # A module that logs before startup finished still gets usable output
        # rather than a silent drop.
        configure_logging()
    return structlog.stdlib.get_logger(name)


def new_run_id() -> str:
    """Generate a short identifier for a training run or serving request."""
    return uuid.uuid4().hex[:12]


@contextmanager
def bound_context(**values: Any) -> Iterator[None]:
    """Bind key/values onto every log event emitted inside the block.

    Example:
        >>> with bound_context(run_id="abc123", stage="training"):
        ...     get_logger(__name__).info("started")
    """
    bind_contextvars(**values)
    try:
        yield
    finally:
        unbind_contextvars(*values)


@contextmanager
def run_context(run_id: str | None = None, **values: Any) -> Iterator[str]:
    """Bind a ``run_id`` (generated when omitted) for the duration of the block.

    Yields:
        The run identifier, so callers can record it alongside their outputs.
    """
    identifier = run_id or new_run_id()
    with bound_context(run_id=identifier, **values):
        yield identifier


def reset_context() -> None:
    """Clear all bound context variables. Used between requests and in tests."""
    clear_contextvars()


__all__ = [
    "REDACTED",
    "bound_context",
    "configure_logging",
    "get_logger",
    "new_run_id",
    "reset_context",
    "run_context",
]
