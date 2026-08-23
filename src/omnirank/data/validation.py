"""Batch schema validation - component 2.

:mod:`omnirank.data.schemas` validates a record in isolation. This module
validates a *batch*, which is where the interesting failures live: duplicates,
dangling references, and timestamps that are individually well-formed but
collectively impossible.

The contract is deliberately non-throwing by default. A real catalogue always
contains some bad rows; failing the whole job on the first one is useless.
Instead every problem is collected into a :class:`ValidationReport` that says
what was rejected and why, and the caller decides whether the rejection rate is
acceptable. ``strict=True`` restores fail-fast behaviour for CI fixtures.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import ValidationError

from omnirank.core.config import DataConfig
from omnirank.core.exceptions import SchemaValidationError
from omnirank.core.logging import get_logger
from omnirank.data.schemas import Interaction, Item, User

logger = get_logger(__name__)


class ValidationRule(StrEnum):
    """Stable identifiers for every rejection reason.

    These are counted, logged, and surfaced in run reports, so they are part of
    the observable contract and must not be renamed casually.
    """

    MISSING_ID = "missing_id"
    MALFORMED_RECORD = "malformed_record"
    INVALID_TIMESTAMP = "invalid_timestamp"
    FUTURE_TIMESTAMP = "future_timestamp"
    INVALID_PRICE = "invalid_price"
    INVALID_RATING = "invalid_rating"
    UNKNOWN_EVENT_TYPE = "unknown_event_type"
    DUPLICATE_EVENT = "duplicate_event"
    UNKNOWN_ITEM_REFERENCE = "unknown_item_reference"
    UNKNOWN_USER_REFERENCE = "unknown_user_reference"
    INVALID_AVAILABILITY = "invalid_availability"
    DUPLICATE_ENTITY_ID = "duplicate_entity_id"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One rejected record."""

    rule: ValidationRule
    message: str
    # Position in the input batch, so an operator can find the offending row.
    index: int | None = None
    identifier: str | None = None


@dataclass(slots=True)
class ValidationReport:
    """Outcome of validating one batch."""

    entity: str
    total: int = 0
    valid: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def rejected(self) -> int:
        """Number of records that did not survive validation."""
        return self.total - self.valid

    @property
    def rejection_rate(self) -> float:
        """Fraction of the batch rejected; 0.0 for an empty batch."""
        return self.rejected / self.total if self.total else 0.0

    @property
    def counts_by_rule(self) -> dict[str, int]:
        """How many records each rule rejected, for logging and run reports."""
        return dict(Counter(issue.rule.value for issue in self.issues))

    @property
    def ok(self) -> bool:
        """True when nothing was rejected."""
        return not self.issues

    def record(
        self,
        rule: ValidationRule,
        message: str,
        *,
        index: int | None = None,
        identifier: str | None = None,
    ) -> None:
        """Append an issue."""
        self.issues.append(
            ValidationIssue(rule=rule, message=message, index=index, identifier=identifier)
        )

    def raise_if_failed(self) -> None:
        """Raise :class:`SchemaValidationError` when any record was rejected."""
        if self.issues:
            raise SchemaValidationError(
                f"{self.rejected} of {self.total} {self.entity} records failed validation",
                entity=self.entity,
                counts_by_rule=self.counts_by_rule,
                first_issues=[issue.message for issue in self.issues[:5]],
            )

    def summary(self) -> dict[str, Any]:
        """Compact, log-safe view. Contains counts only - never record content."""
        return {
            "entity": self.entity,
            "total": self.total,
            "valid": self.valid,
            "rejected": self.rejected,
            "rejection_rate": round(self.rejection_rate, 6),
            "counts_by_rule": self.counts_by_rule,
        }


@dataclass(frozen=True, slots=True)
class ValidatedBatch:
    """Records that passed, paired with the report explaining what did not."""

    users: tuple[User, ...]
    items: tuple[Item, ...]
    interactions: tuple[Interaction, ...]
    reports: tuple[ValidationReport, ...]

    @property
    def ok(self) -> bool:
        """True when every entity validated cleanly."""
        return all(report.ok for report in self.reports)


def _pydantic_issue(exc: ValidationError) -> tuple[ValidationRule, str]:
    """Map a pydantic failure onto the coarse rule vocabulary."""
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first["loc"])
    message = f"{location}: {first['msg']}"
    if "timestamp" in location or "created_at" in location:
        return ValidationRule.INVALID_TIMESTAMP, message
    if "price" in location:
        return ValidationRule.INVALID_PRICE, message
    if "event_type" in location:
        return ValidationRule.UNKNOWN_EVENT_TYPE, message
    if "available" in location:
        return ValidationRule.INVALID_AVAILABILITY, message
    if first["type"] == "missing" and location.endswith("_id"):
        return ValidationRule.MISSING_ID, message
    if location.endswith("_id"):
        return ValidationRule.MISSING_ID, message
    return ValidationRule.MALFORMED_RECORD, message


def _now() -> datetime:
    return datetime.now(tz=UTC)


def validate_users(
    rows: Iterable[Any], config: DataConfig, *, now: datetime | None = None
) -> tuple[list[User], ValidationReport]:
    """Parse and validate user rows.

    Args:
        rows: Mappings or already-built :class:`User` instances.
        config: Domain profile supplying the timestamp bounds.
        now: Injectable clock for deterministic tests.
    """
    report = ValidationReport(entity="users")
    reference_now = now or _now()
    seen: set[str] = set()
    valid: list[User] = []

    for index, row in enumerate(rows):
        report.total += 1
        try:
            user = row if isinstance(row, User) else User.model_validate(row)
        except ValidationError as exc:
            rule, message = _pydantic_issue(exc)
            report.record(rule, message, index=index)
            continue

        if user.user_id in seen:
            report.record(
                ValidationRule.DUPLICATE_ENTITY_ID,
                "user_id appears more than once in the batch",
                index=index,
                identifier=user.user_id,
            )
            continue
        if not _timestamp_in_bounds(user.created_at, config, reference_now, report, index):
            continue

        seen.add(user.user_id)
        valid.append(user)

    report.valid = len(valid)
    return valid, report


def validate_items(
    rows: Iterable[Any], config: DataConfig, *, now: datetime | None = None
) -> tuple[list[Item], ValidationReport]:
    """Parse and validate item rows, enforcing the domain's price bounds."""
    report = ValidationReport(entity="items")
    reference_now = now or _now()
    rules = config.validation
    seen: set[str] = set()
    valid: list[Item] = []

    for index, row in enumerate(rows):
        report.total += 1
        try:
            item = row if isinstance(row, Item) else Item.model_validate(row)
        except ValidationError as exc:
            rule, message = _pydantic_issue(exc)
            report.record(rule, message, index=index)
            continue

        if item.item_id in seen:
            report.record(
                ValidationRule.DUPLICATE_ENTITY_ID,
                "item_id appears more than once in the batch",
                index=index,
                identifier=item.item_id,
            )
            continue
        if item.price is not None and not rules.min_price <= item.price <= rules.max_price:
            report.record(
                ValidationRule.INVALID_PRICE,
                f"price {item.price} outside [{rules.min_price}, {rules.max_price}]",
                index=index,
                identifier=item.item_id,
            )
            continue
        if not _timestamp_in_bounds(item.created_at, config, reference_now, report, index):
            continue

        seen.add(item.item_id)
        valid.append(item)

    report.valid = len(valid)
    return valid, report


def validate_interactions(
    rows: Iterable[Any],
    config: DataConfig,
    *,
    known_user_ids: set[str] | None = None,
    known_item_ids: set[str] | None = None,
    now: datetime | None = None,
) -> tuple[list[Interaction], ValidationReport]:
    """Parse and validate interaction rows.

    Beyond per-record parsing this enforces the three batch-level rules that
    matter most downstream: the event type must be declared by the domain
    profile, references must resolve, and duplicates must not inflate counts.

    Args:
        rows: Mappings or :class:`Interaction` instances.
        config: Domain profile: declared event types, bounds, dedup policy.
        known_user_ids: When given, interactions referencing an unknown user are
            rejected. Pass ``None`` to skip referential checking (e.g. when
            users are streamed separately).
        known_item_ids: As above, for items.
        now: Injectable clock.
    """
    report = ValidationReport(entity="interactions")
    reference_now = now or _now()
    rules = config.validation
    declared_events = set(config.event_types)
    seen_keys: set[tuple[str, str, str, datetime]] = set()
    valid: list[Interaction] = []

    for index, row in enumerate(rows):
        report.total += 1
        try:
            event = row if isinstance(row, Interaction) else Interaction.model_validate(row)
        except ValidationError as exc:
            rule, message = _pydantic_issue(exc)
            report.record(rule, message, index=index)
            continue

        if event.event_type.value not in declared_events:
            report.record(
                ValidationRule.UNKNOWN_EVENT_TYPE,
                f"event_type {event.event_type.value!r} is not declared for domain "
                f"{config.domain!r}",
                index=index,
                identifier=event.interaction_id,
            )
            continue

        if not _timestamp_in_bounds(event.timestamp, config, reference_now, report, index):
            continue

        if (
            event.event_type.value == "rating"
            and event.event_value is not None
            and not rules.min_rating <= event.event_value <= rules.max_rating
        ):
            report.record(
                ValidationRule.INVALID_RATING,
                f"rating {event.event_value} outside [{rules.min_rating}, {rules.max_rating}]",
                index=index,
                identifier=event.interaction_id,
            )
            continue

        if known_user_ids is not None and event.user_id not in known_user_ids:
            report.record(
                ValidationRule.UNKNOWN_USER_REFERENCE,
                "user_id does not resolve to a known user",
                index=index,
                identifier=event.interaction_id,
            )
            continue

        if known_item_ids is not None and event.item_id not in known_item_ids:
            report.record(
                ValidationRule.UNKNOWN_ITEM_REFERENCE,
                "item_id does not resolve to a known item",
                index=index,
                identifier=event.interaction_id,
            )
            continue

        if rules.drop_duplicate_events:
            key = event.dedup_key
            if key in seen_keys:
                report.record(
                    ValidationRule.DUPLICATE_EVENT,
                    "identical (user, item, event_type, timestamp) already seen",
                    index=index,
                    identifier=event.interaction_id,
                )
                continue
            seen_keys.add(key)

        valid.append(event)

    report.valid = len(valid)
    return valid, report


def _timestamp_in_bounds(
    value: datetime,
    config: DataConfig,
    now: datetime,
    report: ValidationReport,
    index: int,
) -> bool:
    """Check a timestamp against the domain's window; record the issue if not."""
    rules = config.validation
    if value < rules.min_timestamp:
        report.record(
            ValidationRule.INVALID_TIMESTAMP,
            f"timestamp {value.isoformat()} precedes the configured floor "
            f"{rules.min_timestamp.isoformat()}",
            index=index,
        )
        return False
    if not rules.allow_future_timestamps and value > now:
        report.record(
            ValidationRule.FUTURE_TIMESTAMP,
            f"timestamp {value.isoformat()} is in the future; a clock skew here "
            "would leak future events into the training window",
            index=index,
        )
        return False
    return True


def validate_batch(
    users: Sequence[Any],
    items: Sequence[Any],
    interactions: Sequence[Any],
    config: DataConfig,
    *,
    check_references: bool = True,
    strict: bool = False,
    now: datetime | None = None,
) -> ValidatedBatch:
    """Validate all three entities together, resolving cross-entity references.

    Args:
        users: Raw or parsed user rows.
        items: Raw or parsed item rows.
        interactions: Raw or parsed interaction rows.
        config: Domain profile.
        check_references: Reject interactions pointing at users/items that did
            not survive their own validation.
        strict: Raise on any rejection instead of reporting it.
        now: Injectable clock.

    Returns:
        A :class:`ValidatedBatch` holding survivors and per-entity reports.

    Raises:
        SchemaValidationError: ``strict`` is set and something was rejected.
    """
    reference_now = now or _now()
    valid_users, user_report = validate_users(users, config, now=reference_now)
    valid_items, item_report = validate_items(items, config, now=reference_now)

    valid_interactions, interaction_report = validate_interactions(
        interactions,
        config,
        known_user_ids={user.user_id for user in valid_users} if check_references else None,
        known_item_ids={item.item_id for item in valid_items} if check_references else None,
        now=reference_now,
    )

    reports = (user_report, item_report, interaction_report)
    for report in reports:
        log = logger.warning if report.issues else logger.info
        log("data.validation.completed", **report.summary())

    if strict:
        for report in reports:
            report.raise_if_failed()

    return ValidatedBatch(
        users=tuple(valid_users),
        items=tuple(valid_items),
        interactions=tuple(valid_interactions),
        reports=reports,
    )


__all__ = [
    "ValidatedBatch",
    "ValidationIssue",
    "ValidationReport",
    "ValidationRule",
    "validate_batch",
    "validate_interactions",
    "validate_items",
    "validate_users",
]
