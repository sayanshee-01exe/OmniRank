"""HTTP serving layer.

``create_app`` is imported lazily by callers rather than re-exported eagerly,
so importing ``omnirank.api.schemas`` in a test does not construct an
application.
"""

from __future__ import annotations

from omnirank.api.app import create_app

__all__ = ["create_app"]
