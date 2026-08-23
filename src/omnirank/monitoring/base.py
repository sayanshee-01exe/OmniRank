"""Monitoring contracts - component 20.

Phase 1 provides the *seam*, not the system. Serving and training code emits
through :class:`MetricsSink`, and the only implementation right now is
:class:`LoggingMetricsSink`, which writes structured log events. Wiring a
Prometheus client in a later phase is then a one-line swap in the dependency
wiring rather than an edit to every call site - and until then, the numbers are
not lost, they are in the logs.

The metric names below are fixed now because renaming a metric after dashboards
exist is expensive.

PHASE 1 STATUS: interface plus the logging sink. Prometheus and Grafana are
deferred; see ``docs/phase_reports/phase_01_report.md``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omnirank.core.logging import get_logger

logger = get_logger(__name__)

# --- Canonical metric names ------------------------------------------------ #
# Serving
REQUEST_LATENCY_MS = "omnirank_request_latency_ms"
REQUEST_TOTAL = "omnirank_requests_total"
FALLBACK_TOTAL = "omnirank_fallback_total"
CANDIDATES_GENERATED = "omnirank_candidates_generated"
EMPTY_RESPONSE_TOTAL = "omnirank_empty_responses_total"
CACHE_HIT_TOTAL = "omnirank_cache_hits_total"
CACHE_MISS_TOTAL = "omnirank_cache_misses_total"
# Offline
TRAINING_DURATION_SECONDS = "omnirank_training_duration_seconds"
VALIDATION_REJECTED_TOTAL = "omnirank_validation_rejected_total"


@runtime_checkable
class MetricsSink(Protocol):
    """Where counters and timings go."""

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Add to a counter."""
        ...

    def observe(self, name: str, value: float, **labels: str) -> None:
        """Record a single observation of a distribution (latency, list size)."""
        ...

    def gauge(self, name: str, value: float, **labels: str) -> None:
        """Set a point-in-time value (artifacts loaded, index size)."""
        ...


class LoggingMetricsSink:
    """Emits metrics as structured log events.

    Deliberately the Phase 1 default: it needs no server, works identically in
    tests and in CI, and produces output that is already correlated with the
    request that generated it via the bound ``run_id``.
    """

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Log a counter increment."""
        logger.info("metric.counter", metric=name, value=value, **labels)

    def observe(self, name: str, value: float, **labels: str) -> None:
        """Log a distribution observation."""
        logger.info("metric.observation", metric=name, value=value, **labels)

    def gauge(self, name: str, value: float, **labels: str) -> None:
        """Log a gauge reading."""
        logger.info("metric.gauge", metric=name, value=value, **labels)


__all__ = [
    "CACHE_HIT_TOTAL",
    "CACHE_MISS_TOTAL",
    "CANDIDATES_GENERATED",
    "EMPTY_RESPONSE_TOTAL",
    "FALLBACK_TOTAL",
    "REQUEST_LATENCY_MS",
    "REQUEST_TOTAL",
    "TRAINING_DURATION_SECONDS",
    "VALIDATION_REJECTED_TOTAL",
    "LoggingMetricsSink",
    "MetricsSink",
]
