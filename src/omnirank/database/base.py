"""Database access contracts - component 19.

The repository interfaces below are what the rest of OmniRank depends on; no
module outside this package imports SQLAlchemy, psycopg, or writes SQL. That is
what allows Phase 1 to ship a validated schema and a real DDL file without
taking a driver dependency, and Phase 2 to add the implementation without
touching a single caller.

Schema: ``schema.sql`` in this package is the authoritative definition, applied
by docker-compose on first start. See ``docs/data/database_schema.md`` for the
partitioning and retention rationale.

Migrations: none yet, deliberately. Phase 1 has no deployed database to migrate
and no prior schema to migrate from, so Alembic would add a version table and a
toolchain in exchange for nothing. It is adopted in Phase 2, at the first change
to a schema that already holds data - recorded in ADR-005.

PHASE 1 STATUS: contracts and DDL only.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from omnirank.data.schemas import Interaction, Item, User

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def read_schema_sql() -> str:
    """Return the DDL text.

    Used by docker-compose (via a bind mount) and by integration tests that
    provision a throwaway database.
    """
    return SCHEMA_PATH.read_text()


@runtime_checkable
class UserRepository(Protocol):
    """Durable access to users."""

    def get(self, user_id: str) -> User | None:
        """Return one user, or ``None`` when unknown.

        Unknown is a normal outcome - anonymous and brand-new users are the
        common case - so it is not an exception.
        """
        ...

    def upsert_many(self, users: Sequence[User]) -> int:
        """Insert or update users. Returns the number of rows written."""
        ...


@runtime_checkable
class ItemRepository(Protocol):
    """Durable access to the catalogue."""

    def get(self, item_id: str) -> Item | None:
        """Return one item, or ``None`` when unknown."""
        ...

    def get_many(self, item_ids: Sequence[str]) -> dict[str, Item]:
        """Batch fetch, keyed by id. Missing ids are simply absent.

        Batched because hydrating a 200-item candidate list one row at a time
        would dominate the serving latency budget.
        """
        ...

    def upsert_many(self, items: Sequence[Item]) -> int:
        """Insert or update items. Returns the number of rows written."""
        ...


@runtime_checkable
class InteractionRepository(Protocol):
    """Append and read the event log - component 17."""

    def append_many(self, interactions: Sequence[Interaction]) -> int:
        """Append events idempotently.

        Implementations must rely on the ``uq_interactions_event`` business-key
        index and ignore conflicts, so that a retried delivery does not
        double-count. Returns the number of rows actually inserted.
        """
        ...

    def recent_for_user(
        self,
        user_id: str,
        *,
        limit: int,
        before: datetime | None = None,
    ) -> list[Interaction]:
        """Most recent events for a user, newest first.

        ``before`` bounds the read to events strictly earlier than the given
        instant, which is how an offline backfill reproduces exactly what
        serving would have seen at that moment.
        """
        ...


__all__ = [
    "SCHEMA_PATH",
    "InteractionRepository",
    "ItemRepository",
    "UserRepository",
    "read_schema_sql",
]
