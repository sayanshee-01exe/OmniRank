"""Ranking-feature contract - component 12's input side.

The :class:`~omnirank.models.base.Ranker` interface itself lives in
``models.base``, next to :class:`~omnirank.models.base.CandidateGenerator`,
because they are peer model contracts. What lives here is the thing that makes
ranking work or fail in production: how a (user, candidate) pair becomes a
feature row.

The single most valuable property this contract enforces is that **the same
code produces features offline and online**. Training/serving skew in a ranker
is nearly invisible - metrics look fine, production quality is quietly worse -
and the usual cause is a feature computed one way in a training notebook and
another way in a request handler. :class:`FeatureBuilder` is therefore the only
sanctioned place to compute ranking features, and ``feature_version`` is
recorded in artifact metadata so a mismatch is detectable.

PHASE 1 STATUS: contract only. The builder and the LightGBM ranker land in
Phase 5.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from omnirank.models.base import Candidate


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """Features for one (user, candidate) pair.

    Values are plain floats keyed by feature name rather than a positional
    vector, so a reordered feature set cannot silently misalign with a trained
    model. The builder converts to a matrix once, in one place, using
    ``feature_names`` as the authoritative order.
    """

    user_id: str
    item_id: str
    values: dict[str, float]


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    """Feature rows for one user's full candidate list."""

    rows: tuple[FeatureRow, ...]
    feature_names: tuple[str, ...]
    # Bumped whenever any feature's *definition* changes. Recorded in artifact
    # metadata; a ranker loaded against a different feature_version is refused.
    feature_version: str

    def __post_init__(self) -> None:
        for row in self.rows:
            missing = set(self.feature_names) - set(row.values)
            if missing:
                raise ValueError(
                    f"feature row for item {row.item_id!r} is missing declared "
                    f"features: {sorted(missing)}"
                )


class FeatureBuilder(ABC):
    """Builds ranking features. The only sanctioned source of them.

    Implementations must be callable identically from a training job and from a
    request handler, must not read the wall clock except through an injected
    ``as_of`` timestamp, and must not consult any data source that is unavailable
    at serving time. Every one of those rules exists to prevent a specific,
    common form of training/serving skew.
    """

    @property
    @abstractmethod
    def feature_version(self) -> str:
        """Version of the feature definitions this builder implements."""

    @property
    @abstractmethod
    def feature_names(self) -> tuple[str, ...]:
        """Authoritative feature order, matching the trained model's columns."""

    @abstractmethod
    def build(
        self,
        user_id: str,
        candidates: Sequence[Candidate],
        context: dict[str, Any] | None = None,
    ) -> FeatureBatch:
        """Compute features for one user's candidates.

        Args:
            user_id: Opaque user identifier.
            candidates: The aggregated candidate list.
            context: Request-time signals. Must include ``as_of`` (a UTC
                timestamp) when any feature is time-dependent, so that offline
                backfills reproduce online values exactly.
        """


__all__ = ["FeatureBatch", "FeatureBuilder", "FeatureRow"]
