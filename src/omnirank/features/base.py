"""Feature and sequence generation contracts - components 6 and 7.

Two producers feed the models:

* :class:`FeatureStore` - static-ish user and item features, computed offline
  and read at request time. The read path is deliberately keyed by entity id
  only: anything that needs a join at request time belongs in the ranking
  feature builder, not here.
* :class:`SequenceBuilder` - per-user chronological item sequences for the
  sequential models.

The one rule that matters in both: **a feature may only use information that
existed at the timestamp it is attached to.** Computing "user's total purchase
count" over the whole log and attaching it to a training row from six months ago
leaks the future, inflates offline metrics, and produces a model that
underperforms online. ``as_of`` is threaded through every signature here so that
respecting it is the default and violating it is visible.

PHASE 1 STATUS: contracts only. Implementations land in Phase 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import Protocol, runtime_checkable

from omnirank.data.schemas import Interaction


@dataclass(frozen=True, slots=True)
class UserSequence:
    """One user's chronological item history.

    ``item_ids`` are ordered oldest to newest and truncated to the configured
    ``data.sequences.max_length``, keeping the most recent events - the ones a
    sequential model actually attends to.
    """

    user_id: str
    item_ids: tuple[str, ...]
    timestamps: tuple[datetime, ...]

    def __post_init__(self) -> None:
        if len(self.item_ids) != len(self.timestamps):
            raise ValueError("UserSequence item_ids and timestamps must be the same length")
        if any(later < earlier for earlier, later in pairwise(self.timestamps)):
            raise ValueError(
                "UserSequence timestamps must be non-decreasing; an out-of-order "
                "history would train a sequential model on a future-to-past signal"
            )

    def __len__(self) -> int:
        return len(self.item_ids)


@runtime_checkable
class FeatureStore(Protocol):
    """Read access to precomputed entity features.

    Backed by parquet files in Phase 2 and by PostgreSQL plus a Redis read-through
    cache once serving needs it (ADR-005). The interface is identical either way,
    which is what allows the switch without touching model code.
    """

    @property
    def feature_version(self) -> str:
        """Version of the feature definitions stored here."""
        ...

    def user_features(self, user_id: str, as_of: datetime | None = None) -> dict[str, float]:
        """Features for one user, as they stood at ``as_of``.

        ``as_of=None`` means "latest", which is correct at serving time and
        wrong during training-set construction.
        """
        ...

    def item_features(self, item_id: str, as_of: datetime | None = None) -> dict[str, float]:
        """Features for one item, as they stood at ``as_of``."""
        ...


class SequenceBuilder(ABC):
    """Builds per-user item sequences for sequential models."""

    @abstractmethod
    def build(
        self,
        interactions: Sequence[Interaction],
        *,
        max_length: int,
        min_length: int,
    ) -> list[UserSequence]:
        """Group interactions into per-user chronological sequences.

        Args:
            interactions: Events to build from. Must already be restricted to
                the training window; passing the full log here is exactly the
                leak this contract exists to prevent.
            max_length: Keep at most this many most-recent events per user.
            min_length: Drop users with fewer events than this.

        Implementations must sort by timestamp with a deterministic tiebreak
        (``interaction_id``), so that two runs over the same data produce
        identical sequences.
        """


__all__ = ["FeatureStore", "SequenceBuilder", "UserSequence"]
