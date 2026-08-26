"""Popularity candidate generator - the baseline everything else must beat.

Two variants:

* ``global_count`` - score is the item's fit-interaction count.
* ``time_decay``  - each interaction contributes ``0.5 ** (age_days / half_life)``,
  where age is measured against the **latest timestamp in the fit data**, not the
  wall clock. Using "now" would make a model's scores change every time it was
  loaded, which is both irreproducible and a subtle form of leakage when the fit
  window ended long ago.

This model matters for two reasons beyond being a yardstick. It is the terminal
stage of the serving fallback chain, so it must never be unavailable; and it is
the number that decides whether a personalised model is earning its complexity.

**Engagement counters are deliberately unused.** PixelRec ships `view_number`,
`thumbup_number` and friends, which look like ideal popularity features. They
are platform-wide lifetime totals carrying no timestamp, so they cannot be
bounded to the fit window - using them would attach post-training-window
popularity to a training-window model. See ``docs/models/popularity.md``.

Requires only numpy and pandas: popularity must work without the modelling
extra installed, because it is the fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Self, cast

import numpy as np
import pandas as pd

from omnirank.core.exceptions import (
    ArtifactValidationError,
    DataError,
)
from omnirank.core.logging import get_logger
from omnirank.models.base import Candidate, CandidateGenerator, ScoredCandidate

logger = get_logger(__name__)

GLOBAL_COUNT: Final = "global_count"
TIME_DECAY: Final = "time_decay"
VARIANTS: Final = (GLOBAL_COUNT, TIME_DECAY)

SECONDS_PER_DAY: Final = 86_400.0

#: Persisted-format version. Bumped when the on-disk layout changes in a way
#: that an older loader would misread.
FORMAT_VERSION: Final = 1

_STATE_FILENAME: Final = "state.npz"
_CONFIG_FILENAME: Final = "config.json"


@dataclass(frozen=True, slots=True)
class PopularityConfig:
    """Popularity hyperparameters."""

    variant: str = TIME_DECAY
    half_life_days: float = 365.0

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise DataError(
                "Unknown popularity variant", variant=self.variant, available=list(VARIANTS)
            )
        if self.variant == TIME_DECAY and self.half_life_days <= 0:
            raise DataError(
                "half_life_days must be positive for the time_decay variant",
                half_life_days=self.half_life_days,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable payload."""
        return {"variant": self.variant, "half_life_days": self.half_life_days}


class PopularityRecommender(CandidateGenerator):
    """Non-personalised recommender scoring items by fit-window popularity.

    Public methods take and return **external** ids; internal dense indices are
    used only inside, and only against the mapping the model was fitted with.
    """

    name = "popularity"

    def __init__(self, config: PopularityConfig | None = None) -> None:
        super().__init__()
        self.config = config or PopularityConfig()
        self._ranked_items: np.ndarray = np.empty(0, dtype="int64")
        self._scores: np.ndarray = np.empty(0, dtype="float64")
        self._score_by_internal: dict[int, float] = {}
        self._internal_to_external: dict[int, str] = {}
        self._external_to_internal: dict[str, int] = {}
        self._seen_by_user: dict[int, set[int]] = {}
        self._external_to_internal_user: dict[str, int] = {}
        self._reference_timestamp: int = 0
        self._mapping_checksum: str = ""
        self._dataset_identity: dict[str, Any] = {}

    # -- fitting ------------------------------------------------------------ #
    def fit(self, data: Any) -> None:
        """Fit from a :class:`PopularityFitData` bundle.

        Args:
            data: Fit interactions plus the mappings they are expressed in.
        """
        if not isinstance(data, PopularityFitData):
            raise DataError(
                "PopularityRecommender.fit expects a PopularityFitData bundle",
                received=type(data).__name__,
            )
        interactions = data.interactions
        if interactions.empty:
            raise DataError("Cannot fit popularity on an empty interaction set")

        item_ids = interactions["internal_item_id"].to_numpy(dtype="int64")
        if self.config.variant == GLOBAL_COUNT:
            weights = np.ones(len(item_ids), dtype="float64")
            reference = int(interactions["timestamp"].max())
        else:
            timestamps = interactions["timestamp"].to_numpy(dtype="int64")
            # The reference is the newest event in the *fit* data, so the model
            # is a pure function of what it was given.
            reference = int(timestamps.max())
            age_days = (reference - timestamps) / SECONDS_PER_DAY
            weights = np.exp(-np.log(2.0) * age_days / self.config.half_life_days)

        catalogue = np.unique(item_ids)
        totals = np.zeros(int(catalogue.max()) + 1, dtype="float64")
        np.add.at(totals, item_ids, weights)
        scores = totals[catalogue]

        # Sort by descending score, ties broken by ascending internal item id.
        # Deterministic ordering is what makes two runs comparable at all.
        order = np.lexsort((catalogue, -scores))
        self._ranked_items = catalogue[order]
        self._scores = scores[order]
        self._score_by_internal = dict(
            zip(self._ranked_items.tolist(), self._scores.tolist(), strict=True)
        )
        self._reference_timestamp = reference
        self._internal_to_external = data.internal_to_external_item
        self._external_to_internal = {v: k for k, v in data.internal_to_external_item.items()}
        self._external_to_internal_user = data.external_to_internal_user
        self._seen_by_user = data.seen_by_user
        self._mapping_checksum = data.mapping_checksum
        self._dataset_identity = data.dataset_identity
        self._fitted = True

        logger.info(
            "popularity.fitted",
            variant=self.config.variant,
            half_life_days=self.config.half_life_days,
            catalogue_size=len(self._ranked_items),
            reference_timestamp=reference,
            top_score=float(self._scores[0]) if len(self._scores) else 0.0,
        )

    # -- inference ---------------------------------------------------------- #
    def recommend(
        self, user_id: str, k: int, context: dict[str, Any] | None = None
    ) -> list[Candidate]:
        """Top-``k`` items for a user, best first.

        An **unknown user** receives the ordinary popularity list. That is the
        correct behaviour for a non-personalised fallback: it has nothing
        user-specific to lose, and returning nothing would leave the fallback
        chain with no terminal stage.
        """
        self.ensure_fitted()
        if k < 1:
            raise DataError("k must be >= 1", k=k)
        filter_seen = True if context is None else bool(context.get("filter_seen", True))
        internal_user = self._external_to_internal_user.get(user_id)
        seen = (
            self._seen_by_user.get(internal_user, set())
            if (filter_seen and internal_user is not None)
            else set()
        )

        candidates: list[Candidate] = []
        for internal_item, score in zip(self._ranked_items, self._scores, strict=True):
            item = int(internal_item)
            if item in seen:
                continue
            candidates.append(
                Candidate(
                    item_id=self._internal_to_external[item],
                    score=float(score),
                    sources=(self.name,),
                    source_scores={self.name: float(score)},
                )
            )
            if len(candidates) == k:
                break
        return candidates

    def recommend_batch(
        self, user_ids: list[str], k: int, *, filter_seen: bool = True
    ) -> dict[str, list[str]]:
        """Top-``k`` external item ids for many users.

        Far cheaper than looping :meth:`recommend`: the ranking is global and
        computed once, so each user costs only their seen-set scan. The scan
        depth is bounded by ``k + |seen|``, not the catalogue.
        """
        self.ensure_fitted()
        if k < 1:
            raise DataError("k must be >= 1", k=k)
        ranked = self._ranked_items.tolist()
        results: dict[str, list[str]] = {}
        for user_id in user_ids:
            internal_user = self._external_to_internal_user.get(user_id)
            seen = (
                self._seen_by_user.get(internal_user, set())
                if (filter_seen and internal_user is not None)
                else set()
            )
            picked: list[str] = []
            for internal_item in ranked:
                if internal_item in seen:
                    continue
                picked.append(self._internal_to_external[internal_item])
                if len(picked) == k:
                    break
            results[user_id] = picked
        return results

    def recommend_batch_scored(
        self, user_ids: list[str], k: int, *, filter_seen: bool = True
    ) -> dict[str, list[ScoredCandidate]]:
        """Batch retrieval keeping the decayed-popularity score for each item.

        The score is the same quantity that produced the global ordering, so it
        is genuinely the model's own output rather than a restatement of the
        rank. It is constant across users -- popularity is not personalised --
        which is itself a signal a ranker can use.
        """
        self.ensure_fitted()
        if k < 1:
            raise DataError("k must be >= 1", k=k)
        ranked = self._ranked_items.tolist()
        results: dict[str, list[ScoredCandidate]] = {}
        for user_id in user_ids:
            internal_user = self._external_to_internal_user.get(user_id)
            seen = (
                self._seen_by_user.get(internal_user, set())
                if (filter_seen and internal_user is not None)
                else set()
            )
            picked: list[ScoredCandidate] = []
            for internal_item in ranked:
                if internal_item in seen:
                    continue
                picked.append(
                    ScoredCandidate(
                        item_id=self._internal_to_external[internal_item],
                        rank=len(picked) + 1,
                        score=float(self._score_by_internal[internal_item]),
                        source=self.name,
                    )
                )
                if len(picked) == k:
                    break
            results[user_id] = picked
        return results

    def score(self, user_id: str, item_ids: list[str]) -> list[float]:
        """Score specific items, in the order given.

        Unknown items score ``0.0`` rather than raising: a cold item is a
        legitimate input here, and the ranking stage needs a value for every
        candidate. Scores are user-independent by construction.
        """
        self.ensure_fitted()
        return [
            self._score_by_internal.get(self._external_to_internal.get(item, -1), 0.0)
            for item in item_ids
        ]

    # -- introspection ------------------------------------------------------ #
    @property
    def fit_item_catalogue(self) -> set[int]:
        """Internal ids of every item this model can recommend."""
        return set(self._ranked_items.tolist())

    @property
    def reference_timestamp(self) -> int:
        """Epoch seconds the time decay was measured against."""
        return self._reference_timestamp

    def metadata(self) -> dict[str, Any]:
        """Configuration and fit provenance, for the artifact manifest."""
        return {
            "model": self.name,
            "format_version": FORMAT_VERSION,
            "config": self.config.to_dict(),
            "reference_timestamp": self._reference_timestamp,
            "catalogue_size": len(self._ranked_items),
            "mapping_checksum": self._mapping_checksum,
            "dataset_identity": self._dataset_identity,
        }

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        """Persist to a directory.

        JSON for configuration, NPZ for arrays. No pickle: an artifact loaded
        from disk should not be able to execute code.
        """
        self.ensure_fitted()
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)

        seen_users = np.array(sorted(self._seen_by_user), dtype="int64")
        seen_lengths = np.array(
            [len(self._seen_by_user[user]) for user in seen_users.tolist()], dtype="int64"
        )
        seen_flat = np.array(
            [item for user in seen_users.tolist() for item in sorted(self._seen_by_user[user])],
            dtype="int64",
        )
        np.savez_compressed(
            target / _STATE_FILENAME,
            ranked_items=self._ranked_items,
            scores=self._scores,
            seen_users=seen_users,
            seen_lengths=seen_lengths,
            seen_flat=seen_flat,
            mapping_internal=np.array(sorted(self._internal_to_external), dtype="int64"),
            mapping_external=np.array(
                [self._internal_to_external[key] for key in sorted(self._internal_to_external)],
                dtype=object,
            ),
            user_external=np.array(sorted(self._external_to_internal_user), dtype=object),
            user_internal=np.array(
                [
                    self._external_to_internal_user[key]
                    for key in sorted(self._external_to_internal_user)
                ],
                dtype="int64",
            ),
            allow_pickle=True,
        )
        (target / _CONFIG_FILENAME).write_text(
            json.dumps(self.metadata(), indent=2, sort_keys=True)
        )
        logger.info("popularity.saved", path=str(target))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Restore a saved model.

        Raises:
            ArtifactValidationError: Files are missing, malformed, or were
                written by a different model type or format version.
        """
        source = Path(path)
        state_path, config_path = source / _STATE_FILENAME, source / _CONFIG_FILENAME
        for candidate in (state_path, config_path):
            if not candidate.is_file():
                raise ArtifactValidationError(
                    "Popularity artifact is incomplete", missing=str(candidate)
                )
        try:
            metadata = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError(
                "Popularity config is not valid JSON",
                path=str(config_path),
                reason=str(exc)[:200],
            ) from exc

        if metadata.get("model") != cls.name:
            raise ArtifactValidationError(
                "Artifact was written by a different model type",
                expected=cls.name,
                found=metadata.get("model"),
            )
        if metadata.get("format_version") != FORMAT_VERSION:
            raise ArtifactValidationError(
                "Unsupported popularity artifact format version",
                expected=FORMAT_VERSION,
                found=metadata.get("format_version"),
            )
        try:
            state = np.load(state_path, allow_pickle=True)
        except Exception as exc:
            raise ArtifactValidationError(
                "Popularity state file could not be read; it may be corrupted",
                path=str(state_path),
                reason=str(exc)[:200],
            ) from exc

        model = cls(PopularityConfig(**metadata["config"]))
        model._ranked_items = state["ranked_items"].astype("int64")
        model._scores = state["scores"].astype("float64")
        model._score_by_internal = dict(
            zip(model._ranked_items.tolist(), model._scores.tolist(), strict=True)
        )
        model._internal_to_external = {
            int(key): str(value)
            for key, value in zip(state["mapping_internal"], state["mapping_external"], strict=True)
        }
        model._external_to_internal = {v: k for k, v in model._internal_to_external.items()}
        model._external_to_internal_user = {
            str(key): int(value)
            for key, value in zip(state["user_external"], state["user_internal"], strict=True)
        }
        offset = 0
        seen: dict[int, set[int]] = {}
        flat = state["seen_flat"].tolist()
        for user, length in zip(
            state["seen_users"].tolist(), state["seen_lengths"].tolist(), strict=True
        ):
            seen[int(user)] = set(flat[offset : offset + int(length)])
            offset += int(length)
        model._seen_by_user = seen
        model._reference_timestamp = int(metadata["reference_timestamp"])
        model._mapping_checksum = metadata.get("mapping_checksum", "")
        model._dataset_identity = metadata.get("dataset_identity", {})
        model._fitted = True
        logger.info("popularity.loaded", path=str(source), catalogue=len(model._ranked_items))
        return model

    def require_mapping(self, mapping_checksum: str) -> None:
        """Assert this model was fitted against the given item mapping.

        Raises:
            ArtifactValidationError: Checksums differ. A model paired with the
                wrong mapping resolves every dense index to the wrong item and
                fails silently, so this is a hard error.
        """
        if self._mapping_checksum and mapping_checksum != self._mapping_checksum:
            raise ArtifactValidationError(
                "Item mapping checksum does not match the one this model was "
                "fitted against. Every recommended id would resolve to the wrong item.",
                expected=self._mapping_checksum,
                found=mapping_checksum,
            )


@dataclass(frozen=True, slots=True)
class PopularityFitData:
    """Everything :meth:`PopularityRecommender.fit` needs.

    Bundled rather than passed positionally because the fit boundary - which
    splits count as "seen" - is the easiest thing to get wrong, and naming it
    makes the caller state it.
    """

    interactions: pd.DataFrame
    internal_to_external_item: dict[int, str]
    external_to_internal_user: dict[str, int]
    seen_by_user: dict[int, set[int]]
    mapping_checksum: str = ""
    dataset_identity: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = {"internal_item_id", "timestamp"} - set(self.interactions.columns)
        if missing:
            raise DataError(
                "Fit interactions are missing required columns", missing=sorted(missing)
            )


def build_seen_by_user(interactions: pd.DataFrame) -> dict[int, set[int]]:
    """Map each user to the set of items they interacted with in the fit data."""
    grouped = interactions.groupby("internal_user_id", observed=True)["internal_item_id"]
    return {int(cast("int", user)): set(items.tolist()) for user, items in grouped}


__all__ = [
    "FORMAT_VERSION",
    "GLOBAL_COUNT",
    "TIME_DECAY",
    "VARIANTS",
    "PopularityConfig",
    "PopularityFitData",
    "PopularityRecommender",
    "build_seen_by_user",
]
