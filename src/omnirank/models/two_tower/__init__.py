"""Multimodal two-tower retrieval.

PHASE: 5 - complete.

Two encoders sharing one embedding space: a user tower over interaction history
and an item tower over published text and image vectors. The item tower is what
distinguishes this from every earlier retriever -- it represents an item by what
it *is* rather than by who interacted with it, so an item with no interactions
still has a usable representation.

The cold-start guarantee is one line of :mod:`~omnirank.models.two_tower.model`::

    embedding = content + id_residual * warm_mask

An item the fitting split never saw has ``warm_mask == 0``, so its embedding is
content only. That holds by construction rather than through a fallback path
that might not be reached, and it is why this is the only source in the system
with no unreachable cold-target users.

Provided here:

* :class:`TwoTowerConfig` -- the tracked configuration and its validation.
* :class:`TwoTowerTrainingDataset` -- batching, padding and warm/cold marking.
* :class:`MultimodalTwoTower` -- both towers; encodes, does not retrieve.
* :class:`TwoTowerTrainer` -- in-batch contrastive fitting with false-negative
  masking.
* :class:`RetrievalCatalogue` -- the warm/cold/excluded partition an index is
  built over.
* :class:`TwoTowerRetriever` -- the ``CandidateGenerator`` fusion consumes, plus
  full-catalogue embedding export.
* :func:`save` / :func:`load` -- persistence with identity enforcement.

The network and the retrieval surface are deliberately separate, matching the
``SASRecNetwork``/``SASRec`` split the codebase already uses: the ``nn.Module``
knows how to encode, the ``CandidateGenerator`` knows how to retrieve. Calling
recommendation methods on the raw network is a mistake this split makes
impossible rather than merely discouraged.

Exact FAISS indexing lives in :mod:`omnirank.retrieval.two_tower_index`, and
rolling-fold evaluation in :mod:`omnirank.retrieval.fold_evaluation`.

Requires the ``retrieval`` extra (torch). Imported lazily by the runner and the
CLIs, so a torch-free install still works.
"""

from __future__ import annotations

from omnirank.models.two_tower.catalogue import RetrievalCatalogue, build_catalogue
from omnirank.models.two_tower.config import TwoTowerConfig
from omnirank.models.two_tower.dataset import TwoTowerBatch, TwoTowerTrainingDataset
from omnirank.models.two_tower.generator import TwoTowerRetriever
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
    "RetrievalCatalogue",
    "TrainingHistory",
    "TwoTowerBatch",
    "TwoTowerConfig",
    "TwoTowerRetriever",
    "TwoTowerTrainer",
    "TwoTowerTrainingDataset",
    "UserTower",
    "build_catalogue",
    "build_false_negative_mask",
    "build_metadata",
    "in_batch_contrastive_loss",
    "load",
    "save",
]
