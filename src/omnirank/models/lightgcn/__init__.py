"""LightGCN collaborative retrieval.

PHASE: 4 - NOT IMPLEMENTED.

This package is reserved, not written. It contains no model code, and nothing
imports from it. It exists so the contract below is recorded next to where the
implementation will live rather than only in a planning document.

Planned contents
----------------
- `model.py` - light graph convolution over the user-item bipartite graph;
  no feature transformation or non-linearity, embeddings propagated and
  layer-combined.
- `training.py` - BPR-loss training loop, CPU/MPS only (no CUDA assumption).
- `export.py` - writes user/item embedding matrices for the FAISS index, and
  registers them with `required_index_version` (ADR-006).

Contract it must satisfy
------------------------
:class:`omnirank.models.base.CandidateGenerator`, unchanged. If implementing this model turns
out to require widening that interface, the interface change is reviewed on its
own - a model-specific escape hatch in the base class is how a multi-stage
pipeline degenerates into five bespoke pipelines (ADR-001).
"""

from __future__ import annotations

__all__: list[str] = []
