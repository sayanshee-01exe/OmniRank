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

import pandas as pd

from omnirank.core.exceptions import DataSourceError
from omnirank.core.logging import get_logger
from omnirank.data.processed import ProcessedDataset
from omnirank.evaluation.experiment import measure

logger = get_logger(__name__)

LIGHTGCN = "lightgcn"
SASREC = "sasrec"
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


__all__ = [
    "HYBRID",
    "LIGHTGCN",
    "SASREC",
    "SEQUENCE_SUBDIR",
    "fit_lightgcn",
    "fit_sasrec",
    "load_sequences",
]
