"""PostgreSQL access contracts and the authoritative schema DDL."""

from __future__ import annotations

from omnirank.database.base import (
    SCHEMA_PATH,
    InteractionRepository,
    ItemRepository,
    UserRepository,
    read_schema_sql,
)

__all__ = [
    "SCHEMA_PATH",
    "InteractionRepository",
    "ItemRepository",
    "UserRepository",
    "read_schema_sql",
]
