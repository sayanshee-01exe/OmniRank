"""Rolling-origin temporal validation folds.

Phase 3 found that validation and test targets differ materially in
distribution: test targets are roughly twice as concentrated on trending items
and considerably more recent, because each user's *last* interaction is more
likely to be currently-popular content than their second-to-last. A model
selected on a single validation fold inherits that fold's idiosyncrasies, and
Phase 3's ranking duly reversed on test.

Rolling folds do not fix the distribution gap - nothing at selection time can,
without touching test - but they make a configuration's ranking robust to *which*
held-out position it is measured at, and they expose instability instead of
hiding it behind one number.

Layout, per user, by distance from the end of their history::

    offset 1   last interaction        RESERVED - the official test target
    offset 2   second-to-last          fold target
    offset 3   third-to-last           fold target
    ...        earlier                 fold history

**Offset 1 is never used for selection.** :data:`RESERVED_TEST_OFFSET` is
excluded by :func:`build_fold`, which raises rather than quietly proceeding.

Folds are built **lazily** from the Phase 2 interaction log rather than written
out as duplicate parquet files: at 976k rows a fold is a boolean mask, and
materialising two more copies of the dataset would cost 70 MB to save a fraction
of a second. Reproducibility comes from the manifest, which records the offsets,
the row counts, and a content checksum of each fold's assignment.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import pandas as pd

from omnirank.core.exceptions import DataError
from omnirank.core.logging import get_logger

logger = get_logger(__name__)

#: The official test target. Never a selection fold, by construction.
RESERVED_TEST_OFFSET: Final = 1

#: Default selection folds: the two positions immediately before the test target.
DEFAULT_TARGET_OFFSETS: Final = (3, 2)

FOLD_HISTORY: Final = "history"
FOLD_TARGET: Final = "target"
FOLD_EXCLUDED: Final = "excluded"


@dataclass(slots=True)
class RollingFold:
    """One rolling-origin fold: a history/target partition at a fixed offset."""

    offset: int
    #: The full interaction log with a `fold_role` column added.
    interactions: pd.DataFrame
    eligible_users: int
    excluded_users: int
    minimum_history: int

    @property
    def name(self) -> str:
        """Stable fold identifier, e.g. ``fold_offset_3``."""
        return f"fold_offset_{self.offset}"

    @property
    def history(self) -> pd.DataFrame:
        """Interactions a model fitted on this fold may see."""
        return self.interactions[self.interactions["fold_role"] == FOLD_HISTORY]

    @property
    def targets(self) -> pd.DataFrame:
        """The held-out target rows for this fold."""
        return self.interactions[self.interactions["fold_role"] == FOLD_TARGET]

    @property
    def checksum(self) -> str:
        """Content hash of the fold assignment.

        Covers the (user, order, role) triples, so two builds of the same fold
        over the same data hash identically and a changed split does not.
        """
        frame = self.interactions.loc[:, ["internal_user_id", "interaction_order", "fold_role"]]
        ordered = frame.sort_values(["internal_user_id", "interaction_order"])
        digest = hashlib.sha256()
        digest.update(
            pd.util.hash_pandas_object(ordered, index=False).to_numpy(dtype="uint64").tobytes()
        )
        return digest.hexdigest()

    def describe(self) -> dict[str, Any]:
        """Manifest-ready description."""
        targets = self.targets
        history = self.history
        return {
            "fold": self.name,
            "target_offset": self.offset,
            "eligible_users": self.eligible_users,
            "excluded_users": self.excluded_users,
            "minimum_history": self.minimum_history,
            "history_rows": len(history),
            "history_users": int(history["internal_user_id"].nunique()),
            "history_items": int(history["internal_item_id"].nunique()),
            "target_rows": len(targets),
            "target_users": int(targets["internal_user_id"].nunique()),
            "target_items": int(targets["internal_item_id"].nunique()),
            "checksum": self.checksum,
        }


@dataclass(slots=True)
class RollingValidation:
    """A set of rolling folds plus the manifest describing them."""

    folds: list[RollingFold]
    target_offsets: tuple[int, ...]
    dataset_identity: dict[str, Any] = field(default_factory=dict)

    def fold(self, offset: int) -> RollingFold:
        """One fold by offset.

        Raises:
            DataError: No fold was built at that offset.
        """
        for candidate in self.folds:
            if candidate.offset == offset:
                return candidate
        raise DataError(
            "No rolling fold at that offset",
            requested=offset,
            available=[item.offset for item in self.folds],
        )

    def manifest(self) -> dict[str, Any]:
        """The manifest written beside the reports."""
        return {
            "target_offsets": list(self.target_offsets),
            "reserved_test_offset": RESERVED_TEST_OFFSET,
            "construction": (
                "Lazy per-user rolling-origin folds over the Phase 2 interaction "
                "log. No Phase 2 file is modified or duplicated; each fold is an "
                "assignment of existing rows, checksummed below."
            ),
            "dataset_identity": self.dataset_identity,
            "folds": [item.describe() for item in self.folds],
        }


def build_fold(
    interactions: pd.DataFrame,
    *,
    offset: int,
    minimum_history: int = 1,
) -> RollingFold:
    """Partition the log into history and target at one rolling origin.

    Args:
        interactions: The full Phase 2 interaction log, with
            ``internal_user_id`` and ``interaction_order``.
        offset: Distance from the end of each user's history. ``2`` targets the
            second-to-last interaction, ``3`` the third-to-last.
        minimum_history: Interactions a user must retain before the target to be
            eligible. Users with fewer contribute history only and are excluded
            from the fold's evaluation population - not dropped from the data,
            because their interactions still inform the model.

    Returns:
        A :class:`RollingFold`.

    Raises:
        DataError: The offset is the reserved test position, is not positive,
            or the log is empty or missing required columns.

    Every row at a position *later* than the target is marked ``excluded`` and is
    visible to nothing: it is the future relative to this fold's origin.
    """
    if offset == RESERVED_TEST_OFFSET:
        raise DataError(
            "Offset 1 is the official test target and is reserved. Using it for "
            "selection would tune against the final benchmark.",
            offset=offset,
            reserved=RESERVED_TEST_OFFSET,
        )
    if offset < 1:
        raise DataError("Rolling offset must be positive", offset=offset)
    if interactions.empty:
        raise DataError("Cannot build a rolling fold from an empty log")
    missing = {"internal_user_id", "interaction_order"} - set(interactions.columns)
    if missing:
        raise DataError("Interaction log is missing columns", missing=sorted(missing))

    frame = interactions.sort_values(
        ["internal_user_id", "interaction_order"], kind="mergesort"
    ).reset_index(drop=True)

    per_user = frame.groupby("internal_user_id", observed=True)["interaction_order"].transform(
        "size"
    )
    # 0 is the most recent interaction for that user.
    from_end = (
        per_user
        - 1
        - frame.groupby("internal_user_id", observed=True)["interaction_order"]
        .rank(method="first")
        .astype("int64")
        + 1
    )

    # A user needs `minimum_history` rows before the target, plus the target,
    # plus every position after it that this fold must ignore.
    required = offset + minimum_history
    eligible = per_user >= required

    role = pd.Series(FOLD_EXCLUDED, index=frame.index, dtype="object")
    role[from_end > offset - 1] = FOLD_HISTORY
    role[eligible & (from_end == offset - 1)] = FOLD_TARGET
    # Ineligible users contribute all their pre-target rows as history and have
    # no target, so they inform the model without being evaluated.
    role[~eligible & (from_end > offset - 1)] = FOLD_HISTORY
    frame["fold_role"] = role

    total_users = int(frame["internal_user_id"].nunique())
    eligible_users = int(frame.loc[eligible, "internal_user_id"].nunique())

    fold = RollingFold(
        offset=offset,
        interactions=frame,
        eligible_users=eligible_users,
        excluded_users=total_users - eligible_users,
        minimum_history=minimum_history,
    )
    logger.info("rolling.fold_built", **fold.describe())
    return fold


def build_rolling_validation(
    interactions: pd.DataFrame,
    *,
    target_offsets: Sequence[int] = DEFAULT_TARGET_OFFSETS,
    minimum_history: int = 1,
    dataset_identity: dict[str, Any] | None = None,
) -> RollingValidation:
    """Build every configured rolling fold.

    Raises:
        DataError: No offsets were given, or one of them is reserved.
    """
    offsets = tuple(int(value) for value in target_offsets)
    if not offsets:
        raise DataError("At least one rolling target offset is required")
    if RESERVED_TEST_OFFSET in offsets:
        raise DataError(
            "Offset 1 is the reserved test target and cannot be a selection fold",
            requested=list(offsets),
        )
    if len(set(offsets)) != len(offsets):
        raise DataError("Rolling target offsets must be distinct", requested=list(offsets))

    folds = [
        build_fold(interactions, offset=offset, minimum_history=minimum_history)
        for offset in offsets
    ]
    validation = RollingValidation(
        folds=folds, target_offsets=offsets, dataset_identity=dataset_identity or {}
    )
    logger.info(
        "rolling.built",
        offsets=list(offsets),
        folds=len(folds),
        eligible={item.name: item.eligible_users for item in folds},
    )
    return validation


def check_fold_integrity(fold: RollingFold) -> None:
    """Assert the invariants that make a fold trustworthy.

    Checks, in order:

    1. Every history row precedes its user's target.
    2. No user has more than one target.
    3. The target item is absent from that user's history for the same fold.
    4. No row after the target is visible as history.

    Raises:
        DataError: Any invariant is violated, naming which and giving examples.
    """
    history = fold.history
    targets = fold.targets

    duplicated = targets.groupby("internal_user_id", observed=True).size()
    offenders = duplicated[duplicated > 1]
    if len(offenders):
        raise DataError(
            "A user has more than one target in a single rolling fold",
            fold=fold.name,
            users=offenders.index[:5].tolist(),
        )

    history_max = history.groupby("internal_user_id", observed=True)["interaction_order"].max()
    target_order = targets.set_index("internal_user_id")["interaction_order"]
    shared = history_max.index.intersection(target_order.index)
    late = shared[history_max[shared] >= target_order[shared]]
    if len(late):
        raise DataError(
            "Rolling fold history does not strictly precede its target",
            fold=fold.name,
            offending_users=len(late),
            examples=late[:5].tolist(),
        )

    if "internal_item_id" in fold.interactions.columns:
        seen = history.groupby("internal_user_id", observed=True)["internal_item_id"].agg(set)
        leaked = [
            int(user)
            for user, item in zip(
                targets["internal_user_id"], targets["internal_item_id"], strict=True
            )
            if item in seen.get(user, set())
        ]
        if leaked:
            # Not fatal in itself - a user genuinely can re-watch an item - but
            # PixelRec50K has no repeated (user, item) pairs, so any occurrence
            # here means the fold was built wrong.
            raise DataError(
                "A fold target item also appears in that user's fold history. "
                "PixelRec50K contains no repeated (user, item) pairs, so this "
                "indicates the fold was constructed incorrectly.",
                fold=fold.name,
                offending_users=len(leaked),
                examples=leaked[:5],
            )
    logger.debug("rolling.fold_integrity_ok", fold=fold.name)


def check_no_reserved_offset_used(validation: RollingValidation) -> None:
    """Assert the official test target never entered a selection fold.

    Raises:
        DataError: A fold targets or exposes the reserved offset.
    """
    if RESERVED_TEST_OFFSET in validation.target_offsets:
        raise DataError(
            "A rolling fold targets the reserved test offset",
            offsets=list(validation.target_offsets),
        )
    for fold in validation.folds:
        frame = fold.interactions
        per_user = frame.groupby("internal_user_id", observed=True)["interaction_order"].transform(
            "max"
        )
        # The user's final interaction must never be history or a target.
        final_rows = frame["interaction_order"] == per_user
        roles = set(frame.loc[final_rows, "fold_role"].unique())
        if roles - {FOLD_EXCLUDED}:
            raise DataError(
                "The final interaction of some user is visible inside a selection "
                "fold. That row is the official test target and must stay excluded.",
                fold=fold.name,
                roles=sorted(roles),
            )
    logger.info(
        "rolling.reserved_offset_protected",
        offsets=list(validation.target_offsets),
        reserved=RESERVED_TEST_OFFSET,
    )


__all__ = [
    "DEFAULT_TARGET_OFFSETS",
    "FOLD_EXCLUDED",
    "FOLD_HISTORY",
    "FOLD_TARGET",
    "RESERVED_TEST_OFFSET",
    "RollingFold",
    "RollingValidation",
    "build_fold",
    "build_rolling_validation",
    "check_fold_integrity",
    "check_no_reserved_offset_used",
]
