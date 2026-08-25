"""Multimodal two-tower retrieval.

PHASE: 5 - model core implemented.

Two encoders sharing one embedding space: a user tower over interaction history
and an item tower over published text and image vectors. The item tower is what
distinguishes this from every earlier retriever -- it represents an item by what
it *is* rather than by who interacted with it, so an item with no interactions
still has a usable representation.

Implemented here: dataset, towers, contrastive objective, trainer, persistence.
Not yet implemented (next milestone): full-catalogue item embedding export,
FAISS index, cold-item retrieval benchmarks, modality ablations, and the
``CandidateGenerator`` wrapper that lets fusion treat this like the other four
sources.

Requires the ``retrieval`` extra (torch). Imported lazily by the runner and the
CLIs, so a torch-free install still works.
"""

from __future__ import annotations

from omnirank.models.two_tower.config import TwoTowerConfig
from omnirank.models.two_tower.dataset import TwoTowerBatch, TwoTowerTrainingDataset
from omnirank.models.two_tower.losses import (
    ContrastiveOutput,
    build_false_negative_mask,
    in_batch_contrastive_loss,
)
from omnirank.models.two_tower.model import (
    ItemTower,
    ModalityEncoder,
    MultimodalTwoTower,
    UserTower,
)
from omnirank.models.two_tower.persistence import build_metadata, load, save
from omnirank.models.two_tower.training import TrainingHistory, TwoTowerTrainer

__all__ = [
    "ContrastiveOutput",
    "ItemTower",
    "ModalityEncoder",
    "MultimodalTwoTower",
    "TrainingHistory",
    "TwoTowerBatch",
    "TwoTowerConfig",
    "TwoTowerTrainer",
    "TwoTowerTrainingDataset",
    "UserTower",
    "build_false_negative_mask",
    "build_metadata",
    "in_batch_contrastive_loss",
    "load",
    "save",
]
