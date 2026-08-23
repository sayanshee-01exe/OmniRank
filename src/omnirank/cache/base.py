"""Caching contracts - component 18.

Redis holds only data that can be regenerated: rendered recommendation
responses, hot item features, session histories. Nothing here is a source of
truth, which is the property that makes "flush the cache" a safe operation and
keeps ADR-005's split between PostgreSQL and Redis meaningful.

Two rules encoded in the interface:

* **Every write carries a TTL.** There is no unbounded ``set``. A cache entry
  with no expiry is a memory leak with extra steps, and a stale recommendation
  that never expires is worse than a slow one.
* **A cache failure is never a request failure.** Implementations must catch
  their own transport errors and degrade to a miss. The recommendation path must
  survive Redis being down entirely.

Cache keys embed the serving model version, so deploying a new model
invalidates the old responses without an explicit flush.

PHASE 1 STATUS: contracts and the key-building convention only.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CacheBackend(Protocol):
    """Key-value cache with mandatory expiry."""

    def get(self, key: str) -> bytes | None:
        """Return the cached value, or ``None`` on miss *or* on backend failure."""
        ...

    def set(self, key: str, value: bytes, *, ttl_seconds: int) -> bool:
        """Store a value with an expiry. Returns False when the write failed.

        A False return is informational, not an error to propagate: the caller
        has already computed the value and must serve it regardless.
        """
        ...

    def delete(self, key: str) -> bool:
        """Remove a key. Returns whether anything was removed."""
        ...

    def ping(self) -> bool:
        """Whether the backend is reachable. Used by ``GET /ready``."""
        ...


class CacheKey:
    """Builds namespaced cache keys.

    Centralised so that the key format is defined once. Two properties matter:
    the configured prefix keeps environments from colliding in a shared Redis,
    and the embedded ``model_version`` makes a deploy self-invalidating.
    """

    def __init__(self, prefix: str = "omnirank:") -> None:
        self.prefix = prefix

    def user_recommendations(self, user_id: str, k: int, model_version: str) -> str:
        """Key for a user's rendered top-k response."""
        return f"{self.prefix}rec:user:{model_version}:{user_id}:{k}"

    def similar_items(self, item_id: str, k: int, model_version: str) -> str:
        """Key for an item-to-item similarity response."""
        return f"{self.prefix}rec:similar:{model_version}:{item_id}:{k}"

    def item_features(self, item_id: str, feature_version: str) -> str:
        """Key for one item's cached feature vector."""
        return f"{self.prefix}feat:item:{feature_version}:{item_id}"

    def session_history(self, session_id: str) -> str:
        """Key for a live session's item history."""
        return f"{self.prefix}session:{session_id}"


__all__ = ["CacheBackend", "CacheKey"]
