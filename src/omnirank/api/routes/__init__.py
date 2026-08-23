"""HTTP routes.

Implemented in Phase 1: ``health`` and ``models``, both answering from real
state. Every other module declares its contract and returns 501 - see each
module's docstring for what it is waiting on.
"""

from __future__ import annotations

from omnirank.api.routes import admin, health, interactions, items, models, recommendations

__all__ = ["admin", "health", "interactions", "items", "models", "recommendations"]
