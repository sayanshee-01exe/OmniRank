"""SASRec self-attentive sequential retrieval.

PHASE: 4 - implemented.

Causal self-attention over the user's recent item sequence, trained on
next-item prediction with a sampled binary cross-entropy objective. Unlike BPR
and LightGCN, which see a user as an unordered set of interactions, SASRec
consumes the Phase 2 sequential examples in order.

Requires the ``retrieval`` extra (torch). Imported lazily by the runner and the
CLIs, so a torch-free install still works.
"""

from __future__ import annotations

from omnirank.models.sasrec.model import (
    SASRec,
    SASRecConfig,
    SASRecFitData,
    SASRecNetwork,
    encode_sequences,
)

__all__ = [
    "SASRec",
    "SASRecConfig",
    "SASRecFitData",
    "SASRecNetwork",
    "encode_sequences",
]
