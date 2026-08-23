"""Non-neural baselines: popularity and matrix factorization.

PHASE: 2 - NOT IMPLEMENTED.

This package is reserved, not written. It contains no model code, and nothing
imports from it. It exists so the contract below is recorded next to where the
implementation will live rather than only in a planning document.

Planned contents
----------------
- `popularity.py` - time-decayed global and per-category popularity. Doubles
  as the terminal stage of the serving fallback chain, so it is the one model
  that must never be unavailable.
- `matrix_factorization.py` - implicit-feedback ALS/BPR over the user-item
  matrix. The reference number every later retrieval model must beat (ADR-007).

Contract it must satisfy
------------------------
:class:`omnirank.models.base.CandidateGenerator`, unchanged. If implementing this model turns
out to require widening that interface, the interface change is reviewed on its
own - a model-specific escape hatch in the base class is how a multi-stage
pipeline degenerates into five bespoke pipelines (ADR-001).
"""

from __future__ import annotations

__all__: list[str] = []
