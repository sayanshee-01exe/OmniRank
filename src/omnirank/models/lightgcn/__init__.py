"""LightGCN collaborative retrieval.

PHASE: 4 - implemented.

Light graph convolution over the user-item bipartite graph: neighbourhood
propagation with symmetric normalisation, no feature transformation and no
non-linearity, trained with the same BPR objective as the Phase 3 matrix
factorization so the two are directly comparable.

Requires the ``retrieval`` extra (torch). Imported lazily by the runner and the
CLIs, so a torch-free install still works.
"""

from __future__ import annotations

from omnirank.models.lightgcn.model import (
    LightGCN,
    LightGCNConfig,
    LightGCNFitData,
    build_normalized_adjacency,
    propagate,
)

__all__ = [
    "LightGCN",
    "LightGCNConfig",
    "LightGCNFitData",
    "build_normalized_adjacency",
    "propagate",
]
