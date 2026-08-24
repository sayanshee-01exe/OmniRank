"""Offline evaluation: contracts, metrics, evaluator, slices, and reporting.

The Phase 1 contracts live in :mod:`~omnirank.evaluation.base`; Phase 3
implements them. ``numpy`` is required (it arrives with the ``data`` extra);
nothing here imports torch.
"""

from __future__ import annotations

from omnirank.evaluation.base import Evaluator, GroundTruth, Recommendations
from omnirank.evaluation.bootstrap import (
    ConfidenceInterval,
    bootstrap_metric,
    paired_bootstrap_delta,
)
from omnirank.evaluation.evaluator import (
    PRIMARY_METRICS,
    EvaluationResult,
    OfflineEvaluator,
)
from omnirank.evaluation.ground_truth import EvaluationGroundTruth, build_ground_truth
from omnirank.evaluation.recommendations import RecommendationSet, UserRecommendations
from omnirank.evaluation.slices import SliceResult, evaluate_all_slices

__all__ = [
    "PRIMARY_METRICS",
    "ConfidenceInterval",
    "EvaluationGroundTruth",
    "EvaluationResult",
    "Evaluator",
    "GroundTruth",
    "OfflineEvaluator",
    "RecommendationSet",
    "Recommendations",
    "SliceResult",
    "UserRecommendations",
    "bootstrap_metric",
    "build_ground_truth",
    "evaluate_all_slices",
    "paired_bootstrap_delta",
]
