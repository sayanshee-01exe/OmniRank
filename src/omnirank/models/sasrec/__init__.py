"""SASRec self-attentive sequential retrieval.

PHASE: 4 - NOT IMPLEMENTED.

This package is reserved, not written. It contains no model code, and nothing
imports from it. It exists so the contract below is recorded next to where the
implementation will live rather than only in a planning document.

Planned contents
----------------
- `model.py` - causal self-attention over the user's recent item sequence,
  consuming the sequences shaped by `data.sequences` config.
- `training.py` - next-item prediction with sampled softmax.
- `export.py` - item embeddings plus the sequence encoder used at request
  time to embed the live session.

Contract it must satisfy
------------------------
:class:`omnirank.models.base.CandidateGenerator`, unchanged. If implementing this model turns
out to require widening that interface, the interface change is reviewed on its
own - a model-specific escape hatch in the base class is how a multi-stage
pipeline degenerates into five bespoke pipelines (ADR-001).
"""

from __future__ import annotations

__all__: list[str] = []
