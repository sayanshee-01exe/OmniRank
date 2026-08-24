"""Canonical data contracts.

These three entities - :class:`User`, :class:`Item`, :class:`Interaction` - are
the interface between raw data of any vertical and everything downstream. A new
domain (courses, jobs, movies, ...) is onboarded by writing a loader that emits
these records; no model, feature, or serving code changes.

Two rules keep the contract honest:

* **Identifiers are opaque strings.** Dense integer indices belong to
  :mod:`omnirank.data.id_mapping` and never leak into a contract, an API
  payload, or the database. See ADR-002 and ADR-006.
* **Timestamps are timezone-aware UTC.** Naive datetimes are rejected rather
  than assumed local, because a temporal split silently corrupted by mixed
  offsets is the single easiest way to leak the future into training (ADR-002).

Domain-specific attributes travel in the ``attributes`` free-form mapping rather
than as new required columns, so the core schema stays vertical-agnostic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

# An identifier that is non-empty after stripping and bounded in length. Bounding
# matters because these become cache keys and database indexes.
IdStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]


class EventType(StrEnum):
    """Interaction event vocabulary supported out of the box.

    Ordered weakest to strongest intent. A domain profile assigns each a weight
    in ``configs/data/<domain>.yaml``; the enum fixes the *names* so that
    feature code and evaluation can reason about intent across verticals.
    """

    # Generic implicit feedback: the source records that a user engaged with an
    # item, and nothing finer. Datasets like PixelRec provide exactly this, and
    # calling it a "click" or a "purchase" would assert intent the source never
    # measured. Kept first because it is the weakest possible claim.
    INTERACTION = "interaction"
    VIEW = "view"
    CLICK = "click"
    WISHLIST = "wishlist"
    CART = "cart"
    PURCHASE = "purchase"
    RATING = "rating"


class _Record(BaseModel):
    """Shared record behaviour: strict, immutable, whitespace-normalised."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


def _require_utc(value: datetime, field_name: str) -> datetime:
    """Reject naive datetimes; normalise aware ones to UTC."""
    if value.tzinfo is None:
        raise ValueError(
            f"{field_name} must be timezone-aware. A naive timestamp cannot be "
            "ordered safely against other sources, and temporal splitting depends "
            "on a single global ordering."
        )
    return value.astimezone(UTC)


class User(_Record):
    """A recommendation subject.

    Deliberately thin. Demographics are optional and land in ``attributes``,
    because most verticals have none, and because anything personal that is not
    needed for ranking should not be copied into the feature store at all.
    """

    user_id: IdStr
    # Optional: many datasets expose no user table at all, only an interaction
    # log from which users are derived. Inventing a signup date would be
    # fabrication, and a fabricated date silently corrupts any recency feature
    # built on it.
    created_at: datetime | None = None
    # Free-form, domain-specific context (locale, segment, cohort, ...).
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _require_utc(value, "user.created_at")


class Item(_Record):
    """A recommendable object.

    ``title``/``description`` feed text embeddings and ``image_id`` feeds image
    embeddings; both are optional because catalogues are incomplete in practice
    and a missing modality must degrade rather than crash (ADR-003).
    """

    item_id: IdStr
    # Optional, but never empty when present: a catalogue with untitled rows is
    # normal (0.23% of PixelRec items have none), while a whitespace-only title
    # is a data error worth rejecting.
    title: str | None = Field(default=None, min_length=1, max_length=1024)
    description: str | None = Field(default=None, max_length=20_000)
    category: str | None = Field(default=None, max_length=256)
    brand: str | None = Field(default=None, max_length=256)
    # Non-negative and finite. Currency is a domain-profile concern, not per-row.
    price: float | None = Field(default=None, ge=0.0)
    # Reference to the image asset, resolved by the embedding pipeline. Kept as
    # an identifier rather than a filesystem path so that artifacts stay
    # portable between a laptop and a cloud training host.
    image_id: str | None = Field(default=None, max_length=512)
    available: bool = True
    # Optional for the same reason as `User.created_at`: catalogues frequently
    # ship without a publication date, and a fabricated one would make an item
    # look newer or older than it is to every freshness feature downstream.
    created_at: datetime | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _require_utc(value, "item.created_at")

    @field_validator("price")
    @classmethod
    def _finite_price(cls, value: float | None) -> float | None:
        if value is not None and (value != value or value in (float("inf"), float("-inf"))):
            raise ValueError("item.price must be a finite number")
        return value


class Interaction(_Record):
    """A single observed user-item event.

    ``event_value`` carries the payload the event type implies: a star rating
    for ``rating``, an order value for ``purchase``, ``None`` for a bare view.
    ``weight`` overrides the domain profile's default weight for this event type
    when a source provides one (e.g. an already-decayed implicit signal).
    """

    interaction_id: IdStr
    user_id: IdStr
    item_id: IdStr
    event_type: EventType
    timestamp: datetime
    session_id: IdStr | None = None
    event_value: float | None = None
    weight: float | None = Field(default=None, ge=0.0)

    @field_validator("timestamp")
    @classmethod
    def _timestamp_utc(cls, value: datetime) -> datetime:
        return _require_utc(value, "interaction.timestamp")

    @model_validator(mode="after")
    def _check_event_value(self) -> Self:
        """A rating without a value is unusable; catch it at the boundary."""
        if self.event_type is EventType.RATING and self.event_value is None:
            raise ValueError("interaction.event_value is required when event_type is 'rating'")
        if self.event_value is not None and self.event_value != self.event_value:
            raise ValueError("interaction.event_value must not be NaN")
        return self

    @property
    def dedup_key(self) -> tuple[str, str, str, datetime]:
        """Identity used for duplicate detection.

        Two events from the same user on the same item of the same type at the
        same instant are the same event, however many times the pipeline saw it.
        ``interaction_id`` is deliberately excluded: upstream systems routinely
        re-emit an event with a fresh id.
        """
        return (self.user_id, self.item_id, self.event_type.value, self.timestamp)


__all__ = ["EventType", "IdStr", "Interaction", "Item", "User"]
