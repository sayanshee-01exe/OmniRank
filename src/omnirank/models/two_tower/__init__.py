"""Multimodal two-tower retrieval.

PHASE: 4 - NOT IMPLEMENTED.

This package is reserved, not written. It contains no model code, and nothing
imports from it. It exists so the contract below is recorded next to where the
implementation will live rather than only in a planning document.

Planned contents
----------------
- `towers.py` - separate user and item encoders sharing an output space.
- `fusion.py` - combines precomputed text (SentenceTransformers) and image
  (CLIP) embeddings with tabular item attributes. Embeddings are precomputed
  offline, never at request time (ADR-003), and a missing modality degrades to
  the remaining ones rather than failing.
- `training.py` - in-batch-negative contrastive training.

Contract it must satisfy
------------------------
:class:`omnirank.models.base.CandidateGenerator`, unchanged. If implementing this model turns
out to require widening that interface, the interface change is reviewed on its
own - a model-specific escape hatch in the base class is how a multi-stage
pipeline degenerates into five bespoke pipelines (ADR-001).
"""

from __future__ import annotations

__all__: list[str] = []
