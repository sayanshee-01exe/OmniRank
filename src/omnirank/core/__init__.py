"""Cross-cutting concerns: configuration, logging, device selection, errors.

Nothing in ``core`` may import from any other OmniRank subpackage. Every other
subpackage may import from ``core``. That one-way rule is what keeps the
modular monolith from turning into a cycle (ADR-001).
"""

from __future__ import annotations

from omnirank.core.config import AppConfig, get_config, load_config
from omnirank.core.device import DeviceType, resolve_device
from omnirank.core.exceptions import OmniRankError
from omnirank.core.logging import configure_logging, get_logger, run_context

__all__ = [
    "AppConfig",
    "DeviceType",
    "OmniRankError",
    "configure_logging",
    "get_config",
    "get_logger",
    "load_config",
    "resolve_device",
    "run_context",
]
