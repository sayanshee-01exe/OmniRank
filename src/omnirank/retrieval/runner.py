"""Fit drivers for the Phase 4 retrieval models.

Deliberately thin. Every model here is evaluated by
:func:`omnirank.models.baselines.runner.run_experiment` -- the same driver the
Phase 3 baselines went through, unchanged. That is the whole point: if LightGCN
or SASRec brought its own evaluation harness, a table putting its numbers beside
BPR's would be comparing harnesses as much as models. What varies below is only
how each model is *fitted*; scoring, ground truth, slicing and bootstrap are
shared code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from omnirank.core.exceptions import DataSourceError
from omnirank.core.logging import get_logger
from omnirank.data.processed import ProcessedDataset
from omnirank.evaluation.experiment import measure

logger = get_logger(__name__)

LIGHTGCN = "lightgcn"
SASREC = "sasrec"
TWO_TOWER = "two_tower"
HYBRID = "popularity_bpr_hybrid"

#: Phase 2 writes one sequence file per split under this directory.
SEQUENCE_SUBDIR = "sequential"


def _shared_identity(dataset: ProcessedDataset) -> dict[str, Any]:
    """Mapping checksum and dataset identity, attached to every fitted model."""
    return {
        "mapping_checksum": dataset.mapping_metadata.get("item_mapping_checksum", ""),
        "dataset_identity": dataset.identity.to_dict(),
    }


def fit_lightgcn(
    dataset: ProcessedDataset,
    fit_splits: tuple[str, ...],
    config: Any,
    *,
    device: str = "auto",
    edges: pd.DataFrame | None = None,
) -> tuple[Any, Any]:
    """Fit LightGCN on the interaction graph induced by ``fit_splits``.

    ``edges`` overrides the split-derived graph, which is how a rolling
    temporal fold supplies its own history without the fold having to be
    written back to disk as a new split.
    """
    from omnirank.models.lightgcn import LightGCN, LightGCNFitData

    interactions = dataset.fit_interactions(fit_splits) if edges is None else edges
    data = LightGCNFitData(
        edges=interactions,
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        internal_to_external_item=dataset.internal_to_external_items(),
        external_to_internal_user=dataset.external_to_internal_users(),
        **_shared_identity(dataset),
    )
    model = LightGCN(config, device=device)
    with measure("fit", track_memory=True, items=len(interactions)) as timer:
        model.fit(data)
    return model, timer.result


def load_sequences(processed_root: Path | str, splits: tuple[str, ...]) -> pd.DataFrame:
    """Read and concatenate the Phase 2 sequential examples for ``splits``.

    SASRec is the only model that needs these, so they are not loaded as part
    of :class:`ProcessedDataset` -- every other model would pay for a file it
    never reads.
    """
    root = Path(processed_root) / SEQUENCE_SUBDIR
    frames = []
    for split in splits:
        path = root / f"{split}_sequences.parquet"
        if not path.is_file():
            raise DataSourceError(
                "Sequential examples are missing. Re-run the Phase 2 pipeline.",
                expected=str(path),
            )
        frames.append(pd.read_parquet(path))
    combined = pd.concat(frames, ignore_index=True)
    logger.info("sequences.loaded", splits=list(splits), examples=len(combined))
    return combined


def fit_sasrec(
    dataset: ProcessedDataset,
    fit_splits: tuple[str, ...],
    config: Any,
    *,
    processed_root: Path | str,
    device: str = "auto",
    sequences: pd.DataFrame | None = None,
) -> tuple[Any, Any]:
    """Fit SASRec on the Phase 2 sequential examples for ``fit_splits``.

    ``sequences`` overrides the split-derived examples, for rolling folds.
    """
    from omnirank.models.sasrec import SASRec, SASRecFitData

    examples = load_sequences(processed_root, fit_splits) if sequences is None else sequences
    data = SASRecFitData(
        sequences=examples,
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        internal_to_external_item=dataset.internal_to_external_items(),
        external_to_internal_user=dataset.external_to_internal_users(),
        **_shared_identity(dataset),
    )
    model = SASRec(config, device=device)
    with measure("fit", track_memory=True, items=len(examples)) as timer:
        model.fit(data)
    return model, timer.result


def load_item_tags(processed_root: Path | str, num_items: int) -> tuple[np.ndarray, int]:
    """Per-item category ids, for the item tower's tag embedding.

    Reads only ``internal_item_id`` and ``category``. The same table carries
    PixelRec's platform engagement counters in ``source_metadata``; those are
    excluded here by column selection rather than by discipline, because they
    are not point-in-time safe -- they are dataset-wide totals with no
    guarantee of reflecting what was known at any historical prediction time.
    """
    path = Path(processed_root) / "metadata" / "item_metadata.parquet"
    if not path.is_file():
        logger.warning(
            "two_tower.item_metadata_missing",
            expected=str(path),
            detail="Tag embedding disabled; every item receives the unknown tag.",
        )
        return np.zeros(num_items, dtype="int64"), 1

    frame = pd.read_parquet(path, columns=["internal_item_id", "category"])
    categories = frame["category"].fillna("__unknown__").astype("category")
    # Tag 0 is reserved for unknown, so real categories start at 1.
    codes = categories.cat.codes.to_numpy(dtype="int64") + 1
    tags = np.zeros(num_items, dtype="int64")
    ids = frame["internal_item_id"].to_numpy(dtype="int64")
    inside = (ids >= 0) & (ids < num_items)
    tags[ids[inside]] = codes[inside]
    num_tags = int(tags.max()) + 1
    logger.info("two_tower.item_tags_loaded", items=num_items, tags=num_tags)
    return tags, num_tags


def fit_two_tower(
    dataset: ProcessedDataset,
    fit_splits: tuple[str, ...],
    config: Any,
    *,
    processed_root: Path | str,
    device: str = "auto",
    subset_users: int | None = None,
    max_batches_per_epoch: int | None = None,
    validation_splits: tuple[str, ...] = (),
) -> tuple[Any, Any]:
    """Fit the multimodal two-tower model on the Phase 2 sequential examples.

    ``subset_users`` restricts training to the first N internal user ids, which
    is the development path: the full 776k-example corpus is not something to
    start by accident while checking that a pipeline works.
    """
    from omnirank.features.multimodal_store import MultimodalFeatureStore
    from omnirank.models.two_tower import (
        MultimodalTwoTower,
        TwoTowerTrainer,
        TwoTowerTrainingDataset,
    )

    root = Path(processed_root)
    store = MultimodalFeatureStore(root / "features")
    store.require_compatible(
        mapping_checksum=dataset.mapping_metadata.get("item_mapping_checksum", ""),
        feature_version=store.feature_version,
    )

    tags, num_tags = load_item_tags(root, dataset.num_items)
    sequences = load_sequences(root, fit_splits)
    if subset_users is not None:
        sequences = sequences[sequences["internal_user_id"] < subset_users]
        if sequences.empty:
            raise DataSourceError(
                "No sequential examples remain after the user subset filter.",
                subset_users=subset_users,
            )

    training = TwoTowerTrainingDataset(
        sequences,
        store,
        num_items=dataset.num_items,
        num_users=dataset.num_users,
        maximum_history_length=config.maximum_history_length,
        item_tags=tags,
        num_tags=num_tags,
    )
    validation = None
    if validation_splits:
        held_out = load_sequences(root, validation_splits)
        if subset_users is not None:
            held_out = held_out[held_out["internal_user_id"] < subset_users]
        if not held_out.empty:
            validation = TwoTowerTrainingDataset(
                held_out,
                store,
                num_items=dataset.num_items,
                num_users=dataset.num_users,
                maximum_history_length=config.maximum_history_length,
                # The warm set comes from the *training* split: an item is warm
                # because training saw it, not because validation did.
                warm_items=set(np.flatnonzero(training.warm_mask).tolist()),
                item_tags=tags,
                num_tags=num_tags,
            )

    model = MultimodalTwoTower(
        config,
        text_dim=store.dimension("text"),
        image_dim=store.dimension("image"),
        num_items=dataset.num_items,
        num_users=dataset.num_users,
        num_tags=num_tags,
    )
    trainer = TwoTowerTrainer(model, config, device=device)
    with measure("fit", track_memory=True, items=len(training)) as timer:
        history = trainer.fit(training, validation, max_batches_per_epoch=max_batches_per_epoch)
    return (model, store, history), timer.result


__all__ = [
    "HYBRID",
    "LIGHTGCN",
    "SASREC",
    "SEQUENCE_SUBDIR",
    "TWO_TOWER",
    "fit_lightgcn",
    "fit_sasrec",
    "fit_two_tower",
    "load_item_tags",
    "load_sequences",
]
