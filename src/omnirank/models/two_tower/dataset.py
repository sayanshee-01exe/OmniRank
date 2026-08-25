"""Training examples for the two-tower model.

Built on the Phase 2 sequential tables rather than a second format. Those tables
already guarantee the property that matters most here -- every item in
``item_sequence`` strictly precedes ``target_item``, verified by the Phase 2
leakage checks (L05/L06) -- so reusing them means the guarantee is inherited
rather than re-implemented and re-argued.

**Padding is ``num_items``**, one past the last valid internal id, matching the
convention SASRec already uses. Reusing 0 would collide with a real item and
train the model on a content vector that belongs to something.

**Features are fetched, never held.** The dataset keeps item *ids*; vectors come
from the memory-mapped store at collate time. Materialising history features
per example would turn a 776k-row dataset into hundreds of gigabytes, and
copying the matrices into each worker would multiply the store by the worker
count for no benefit -- the mapping is already shared, read-only, and paged by
the OS.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from omnirank.core.exceptions import DataError
from omnirank.core.logging import get_logger
from omnirank.features.multimodal_store import MultimodalFeatureStore

logger = get_logger(__name__)

REQUIRED_COLUMNS: Final = ("internal_user_id", "item_sequence", "target_item")
#: Tag id used when an item has no category. Distinct from every real tag, so a
#: missing category is a state the embedding can learn rather than a silent
#: collision with whichever category happens to be id 0.
UNKNOWN_TAG: Final = 0


@dataclass(frozen=True, slots=True)
class TwoTowerBatch:
    """One collated batch. Every array is CPU; the trainer moves it."""

    user_ids: np.ndarray
    history_item_ids: np.ndarray
    history_lengths: np.ndarray
    history_padding_mask: np.ndarray

    history_text_features: np.ndarray
    history_image_features: np.ndarray
    history_text_available: np.ndarray
    history_image_available: np.ndarray
    history_tag_ids: np.ndarray

    positive_item_ids: np.ndarray
    positive_text_features: np.ndarray
    positive_image_features: np.ndarray
    positive_text_available: np.ndarray
    positive_image_available: np.ndarray
    positive_tag_ids: np.ndarray
    positive_warm_mask: np.ndarray

    def __len__(self) -> int:
        return int(self.user_ids.size)

    def validate(self, *, text_dim: int, image_dim: int, max_history: int) -> None:
        """Assert the batch has the shape the towers expect.

        Raises:
            DataError: Any shape or finiteness violation.
        """
        rows = len(self)
        expectations = {
            "history_item_ids": (rows, max_history),
            "history_padding_mask": (rows, max_history),
            "history_text_features": (rows, max_history, text_dim),
            "history_image_features": (rows, max_history, image_dim),
            "history_text_available": (rows, max_history),
            "history_image_available": (rows, max_history),
            "history_tag_ids": (rows, max_history),
            "positive_text_features": (rows, text_dim),
            "positive_image_features": (rows, image_dim),
        }
        problems = [
            f"{name}: {getattr(self, name).shape} != {shape}"
            for name, shape in expectations.items()
            if getattr(self, name).shape != shape
        ]
        if problems:
            raise DataError("Batch has unexpected shapes", problems=problems)

        for name in (
            "history_text_features",
            "history_image_features",
            "positive_text_features",
            "positive_image_features",
        ):
            array = getattr(self, name)
            if not np.isfinite(array).all():
                raise DataError(
                    "Batch contains non-finite feature values. A NaN here "
                    "propagates through the whole batch's loss and produces a "
                    "silently dead model rather than an error.",
                    field=name,
                )


class TwoTowerTrainingDataset:
    """Indexable view over the Phase 2 sequential examples.

    Deliberately not a ``torch.utils.data.Dataset``: nothing here needs torch,
    and keeping it torch-free means the dataset and collator can be tested
    without the retrieval extra installed.
    """

    def __init__(
        self,
        sequences: pd.DataFrame,
        store: MultimodalFeatureStore,
        *,
        num_items: int,
        num_users: int,
        maximum_history_length: int = 50,
        warm_items: set[int] | frozenset[int] | None = None,
        item_tags: np.ndarray | None = None,
        num_tags: int = 1,
    ) -> None:
        missing = set(REQUIRED_COLUMNS) - set(sequences.columns)
        if missing:
            raise DataError("Sequential examples missing columns", missing=sorted(missing))
        if sequences.empty:
            raise DataError("Cannot build a two-tower dataset from zero examples")
        if maximum_history_length < 1:
            raise DataError(
                "maximum_history_length must be positive",
                maximum_history_length=maximum_history_length,
            )
        if store.catalogue_size != num_items:
            raise DataError(
                "Feature store does not cover the item catalogue. Its vectors "
                "describe a different set of items than the model is being "
                "fitted on.",
                store_items=store.catalogue_size,
                num_items=num_items,
            )

        self.store = store
        self.num_items = num_items
        self.num_users = num_users
        self.maximum_history_length = maximum_history_length
        self.padding_id = num_items
        self.num_tags = max(num_tags, 1)

        self._users = sequences["internal_user_id"].to_numpy(dtype="int64")
        self._targets = sequences["target_item"].to_numpy(dtype="int64")
        self._histories: list[np.ndarray] = [
            np.asarray(row, dtype="int64") for row in sequences["item_sequence"]
        ]
        self._validate_ids()

        # Warm items are those with a fitting interaction. Anything else must be
        # encoded from content alone, so the residual is gated on this.
        self._warm = np.zeros(num_items, dtype=bool)
        if warm_items is None:
            observed = set(self._targets.tolist())
            for history in self._histories:
                observed.update(history.tolist())
            self._warm[sorted(observed)] = True
        elif warm_items:
            self._warm[sorted(warm_items)] = True

        self._tags = (
            np.asarray(item_tags, dtype="int64")
            if item_tags is not None
            else np.zeros(num_items, dtype="int64")
        )
        if self._tags.shape != (num_items,):
            raise DataError(
                "item_tags must have one entry per catalogue item",
                shape=list(self._tags.shape),
                num_items=num_items,
            )

        logger.info(
            "two_tower.dataset_built",
            examples=len(self),
            users=int(np.unique(self._users).size),
            warm_items=int(self._warm.sum()),
            cold_items=int((~self._warm).sum()),
            padding_id=self.padding_id,
            maximum_history_length=maximum_history_length,
        )

    def _validate_ids(self) -> None:
        """Reject ids the mapping cannot resolve, naming the offender."""
        if self._targets.size and (
            self._targets.min() < 0 or self._targets.max() >= self.num_items
        ):
            raise DataError(
                "Target item id outside the catalogue",
                minimum=int(self._targets.min()),
                maximum=int(self._targets.max()),
                num_items=self.num_items,
            )
        for index, history in enumerate(self._histories):
            if history.size and (history.min() < 0 or history.max() >= self.num_items):
                raise DataError(
                    "History item id outside the catalogue",
                    example=index,
                    minimum=int(history.min()),
                    maximum=int(history.max()),
                    num_items=self.num_items,
                )
            # Phase 2 guarantees this; asserting it here means a future change to
            # sequence construction cannot quietly reintroduce the target.
            if history.size and bool((history == self._targets[index]).any()):
                raise DataError(
                    "Target item appears inside its own input history. The "
                    "model would be asked to predict something it can already "
                    "see.",
                    example=index,
                    target=int(self._targets[index]),
                )

    def __len__(self) -> int:
        return int(self._users.size)

    @property
    def warm_mask(self) -> np.ndarray:
        """Per-item warm flag, indexed by internal id."""
        return self._warm

    @property
    def item_tags(self) -> np.ndarray:
        """Per-item tag id, indexed by internal id."""
        return self._tags

    def history_for(self, index: int) -> np.ndarray:
        """Truncated history for one example, oldest dropped first."""
        return self._histories[index][-self.maximum_history_length :]

    def __getitem__(self, index: int) -> dict[str, Any]:
        """One example, as ids only. Features are attached during collation."""
        history = self.history_for(index)
        return {
            "internal_user_id": int(self._users[index]),
            "history_item_ids": history,
            "history_length": int(history.size),
            "positive_item_id": int(self._targets[index]),
        }

    def batches(
        self, batch_size: int, *, rng: np.random.Generator | None = None
    ) -> Iterator[np.ndarray]:
        """Yield index blocks, shuffled when an rng is supplied.

        Deterministic given a seeded generator: the same seed produces the same
        block sequence, which is what makes a training run reproducible.
        """
        if batch_size < 1:
            raise DataError("batch_size must be positive", batch_size=batch_size)
        order = rng.permutation(len(self)) if rng is not None else np.arange(len(self))
        for start in range(0, len(order), batch_size):
            yield order[start : start + batch_size]

    def collate(self, indices: Sequence[int] | np.ndarray) -> TwoTowerBatch:
        """Build one batch, fetching features from the store.

        Padded history positions are filled with the padding id and zeroed
        features, and flagged in ``history_padding_mask``. The towers gate on
        the mask; the zeros are only there so the array has a shape.
        """
        rows = np.asarray(indices, dtype="int64")
        if rows.size == 0:
            raise DataError("Cannot collate an empty batch")
        width = self.maximum_history_length

        history_ids = np.full((rows.size, width), self.padding_id, dtype="int64")
        lengths = np.zeros(rows.size, dtype="int64")
        for position, index in enumerate(rows):
            history = self.history_for(int(index))
            lengths[position] = history.size
            if history.size:
                # Right-aligned, so the most recent item is the last column and
                # recency weighting can index from the end without knowing the
                # length.
                history_ids[position, width - history.size :] = history
        padding_mask = history_ids == self.padding_id

        # The store has no row for the padding id, so it is read as item 0 and
        # then zeroed by the mask below.
        safe_history = np.where(padding_mask, 0, history_ids)
        flat = self.store.get_batch(safe_history.ravel())
        text = flat.text.reshape(rows.size, width, -1)
        image = flat.image.reshape(rows.size, width, -1)
        text_available = flat.text_mask.reshape(rows.size, width) & ~padding_mask
        image_available = flat.image_mask.reshape(rows.size, width) & ~padding_mask
        text = text * text_available[..., None]
        image = image * image_available[..., None]

        targets = self._targets[rows]
        positive = self.store.get_batch(targets)

        history_tags = np.where(padding_mask, UNKNOWN_TAG, self._tags[safe_history])
        return TwoTowerBatch(
            user_ids=self._users[rows],
            history_item_ids=history_ids,
            history_lengths=lengths,
            history_padding_mask=padding_mask,
            history_text_features=text,
            history_image_features=image,
            history_text_available=text_available,
            history_image_available=image_available,
            history_tag_ids=history_tags,
            positive_item_ids=targets,
            positive_text_features=positive.text,
            positive_image_features=positive.image,
            positive_text_available=positive.text_mask,
            positive_image_available=positive.image_mask,
            positive_tag_ids=self._tags[targets],
            positive_warm_mask=self._warm[targets],
        )

    def positives_by_row(self, indices: Sequence[int] | np.ndarray) -> list[set[int]]:
        """Known positives per row, for false-negative masking.

        A user's history *is* a set of known positives. Without this the loss
        pushes a user away from items they demonstrably interacted with, purely
        because those items happened to be another row's target.
        """
        return [
            set(self.history_for(int(index)).tolist()) | {int(self._targets[int(index)])}
            for index in np.asarray(indices, dtype="int64")
        ]


__all__ = [
    "REQUIRED_COLUMNS",
    "UNKNOWN_TAG",
    "TwoTowerBatch",
    "TwoTowerTrainingDataset",
]
