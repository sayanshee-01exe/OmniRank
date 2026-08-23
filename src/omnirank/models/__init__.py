"""Model interfaces and their (future) implementations.

``base`` holds the contracts. The subpackages hold implementations, added one
phase at a time so that every model is measured against the baselines that
preceded it (ADR-007).
"""

from __future__ import annotations

from omnirank.models.base import Candidate, CandidateGenerator, RankedItem, Ranker

__all__ = ["Candidate", "CandidateGenerator", "RankedItem", "Ranker"]
