"""Concrete recommendation storage.

Implements the :class:`~omnirank.evaluation.base.Recommendations` protocol with
the invariants an evaluator depends on:

* **Order is meaningful** and is preserved exactly as the model produced it.
* **Duplicates are rejected, never silently removed.** A model that recommends
  the same item twice has a bug; deduplicating it here would hide the bug and
  quietly inflate every rank-sensitive metric.
* **An empty list is a legitimate state.** A model that cannot serve a user
  scores zero for that user - it is not dropped from the denominator.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omnirank.core.exceptions import DataError


@dataclass(frozen=True, slots=True)
class UserRecommendations:
    """One user's ordered recommendations."""

    user_id: str
    item_ids: tuple[str, ...]
    scores: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(set(self.item_ids)) != len(self.item_ids):
            duplicates = sorted({item for item in self.item_ids if self.item_ids.count(item) > 1})
            raise DataError(
                "Recommendation list contains duplicate items. Deduplicating here "
                "would hide a model bug and inflate rank-sensitive metrics.",
                user_id=self.user_id,
                duplicates=duplicates[:5],
            )
        if self.scores is not None and len(self.scores) != len(self.item_ids):
            raise DataError(
                "Scores and item ids must be the same length",
                user_id=self.user_id,
                items=len(self.item_ids),
                scores=len(self.scores),
            )

    def __len__(self) -> int:
        return len(self.item_ids)


class RecommendationSet:
    """Recommendations for a population of users.

    Backed by a dict keyed on user id, so ``items_for`` is O(1) - the evaluator
    calls it once per user per metric per cut-off, and a linear scan there turns
    a 50,000-user evaluation into minutes of nothing.
    """

    __slots__ = ("_by_user", "model_name", "model_version")

    def __init__(
        self,
        entries: Iterable[UserRecommendations] = (),
        *,
        model_name: str = "unknown",
        model_version: str = "unknown",
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version
        self._by_user: dict[str, UserRecommendations] = {}
        for entry in entries:
            self.add(entry)

    # -- construction ------------------------------------------------------ #
    def add(self, entry: UserRecommendations) -> None:
        """Record one user's list.

        Raises:
            DataError: The user already has recommendations. Two lists for one
                user means the caller looped twice, and silently keeping either
                one would make the result depend on iteration order.
        """
        if entry.user_id in self._by_user:
            raise DataError("Duplicate recommendations for one user", user_id=entry.user_id)
        self._by_user[entry.user_id] = entry

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Sequence[str]],
        *,
        scores: Mapping[str, Sequence[float]] | None = None,
        model_name: str = "unknown",
        model_version: str = "unknown",
    ) -> RecommendationSet:
        """Build from ``{user_id: [item_id, ...]}``."""
        return cls(
            (
                UserRecommendations(
                    user_id=user_id,
                    item_ids=tuple(items),
                    scores=tuple(scores[user_id]) if scores and user_id in scores else None,
                )
                for user_id, items in mapping.items()
            ),
            model_name=model_name,
            model_version=model_version,
        )

    # -- protocol ---------------------------------------------------------- #
    def users(self) -> Sequence[str]:
        """Users that received recommendations, in insertion order."""
        return tuple(self._by_user)

    def items_for(self, user_id: str) -> Sequence[str]:
        """Recommended item ids, best first. Empty tuple for an unknown user."""
        entry = self._by_user.get(user_id)
        return entry.item_ids if entry else ()

    def scores_for(self, user_id: str) -> Sequence[float] | None:
        """Scores aligned with :meth:`items_for`, when the model supplied them."""
        entry = self._by_user.get(user_id)
        return entry.scores if entry else None

    # -- introspection ----------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._by_user)

    def __contains__(self, user_id: object) -> bool:
        return user_id in self._by_user

    @property
    def total_recommendations(self) -> int:
        """Recommended items across all users, counting repeats across users."""
        return sum(len(entry) for entry in self._by_user.values())

    @property
    def users_with_no_recommendations(self) -> tuple[str, ...]:
        """Users present but holding an empty list - a real, reportable state."""
        return tuple(user for user, entry in self._by_user.items() if not entry.item_ids)

    def exposure_counts(self) -> dict[str, int]:
        """How many users each item was shown to. Feeds coverage and Gini."""
        counts: dict[str, int] = {}
        for entry in self._by_user.values():
            for item_id in entry.item_ids:
                counts[item_id] = counts.get(item_id, 0) + 1
        return counts

    def truncated(self, k: int) -> RecommendationSet:
        """A copy with every list cut to at most ``k`` items."""
        if k < 1:
            raise DataError("Truncation length must be >= 1", k=k)
        return RecommendationSet(
            (
                UserRecommendations(
                    user_id=entry.user_id,
                    item_ids=entry.item_ids[:k],
                    scores=entry.scores[:k] if entry.scores is not None else None,
                )
                for entry in self._by_user.values()
            ),
            model_name=self.model_name,
            model_version=self.model_version,
        )

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        """Deterministic, JSON-ready representation.

        Users are sorted so two runs producing the same recommendations produce
        byte-identical files, which is what makes save/load equality testable.
        """
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "users": [
                {
                    "user_id": entry.user_id,
                    "item_ids": list(entry.item_ids),
                    "scores": list(entry.scores) if entry.scores is not None else None,
                }
                for entry in sorted(self._by_user.values(), key=lambda item: item.user_id)
            ],
        }

    def save(self, path: Path | str) -> Path:
        """Write as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False))
        return target

    @classmethod
    def load(cls, path: Path | str) -> RecommendationSet:
        """Read a previously saved set.

        Raises:
            DataError: The file is missing or malformed.
        """
        source = Path(path)
        if not source.is_file():
            raise DataError("Recommendation file not found", path=str(source))
        try:
            payload = json.loads(source.read_text())
        except json.JSONDecodeError as exc:
            raise DataError(
                "Recommendation file is not valid JSON", path=str(source), reason=str(exc)[:200]
            ) from exc
        return cls(
            (
                UserRecommendations(
                    user_id=row["user_id"],
                    item_ids=tuple(row["item_ids"]),
                    scores=tuple(row["scores"]) if row.get("scores") is not None else None,
                )
                for row in payload["users"]
            ),
            model_name=payload.get("model_name", "unknown"),
            model_version=payload.get("model_version", "unknown"),
        )


__all__ = ["RecommendationSet", "UserRecommendations"]
