"""Preprocessing contracts - component 3.

Preprocessing sits between validation and splitting, and consists of steps that
are individually simple but order-sensitive: k-core filtering, event
deduplication, canonicalising text fields, and building the id mappings.

The important property encoded here is that preprocessing is a *pure function of
validated records plus configuration*. It reads no clock, no database, and no
random state beyond the configured seed - which is what makes a training run
reproducible from the recorded ``configuration_hash`` alone (ADR-006).

Phase 1 defines the protocol and its result contract. The concrete pipeline
lands in Phase 2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from omnirank.core.config import DataConfig
from omnirank.data.id_mapping import IdMapping
from omnirank.data.schemas import Interaction, Item, User


@dataclass(frozen=True, slots=True)
class PreprocessedDataset:
    """Cleaned records plus the id mappings every downstream stage shares.

    The mappings are part of this result rather than being rebuilt per model,
    because two models built on different mappings cannot share an index or an
    embedding table (ADR-006).
    """

    users: Sequence[User]
    items: Sequence[Item]
    interactions: Sequence[Interaction]
    user_mapping: IdMapping
    item_mapping: IdMapping
    # Row counts before/after each step, for the run report and drift checks.
    step_counts: dict[str, int] = field(default_factory=dict)

    @property
    def num_users(self) -> int:
        """Users surviving preprocessing."""
        return len(self.user_mapping)

    @property
    def num_items(self) -> int:
        """Items surviving preprocessing."""
        return len(self.item_mapping)


@runtime_checkable
class Preprocessor(Protocol):
    """Turns validated records into a model-ready dataset."""

    def process(
        self,
        users: Sequence[User],
        items: Sequence[Item],
        interactions: Sequence[Interaction],
        config: DataConfig,
    ) -> PreprocessedDataset:
        """Clean, filter, and index the batch.

        Implementations must apply k-core filtering iteratively (removing sparse
        users can make items sparse and vice versa) and must build the id
        mappings from the *post-filtering* survivors only, so no dense index is
        ever allocated to an entity the models will not see.
        """
        ...


__all__ = ["PreprocessedDataset", "Preprocessor"]
