"""Experiment orchestration: fit, recommend, evaluate, measure.

One place that knows how to take a dataset plus a model configuration and
produce a complete, comparable result. Centralising it is what guarantees that
popularity and BPR are measured identically - a comparison where each model
brought its own evaluation harness would not be a comparison.

The fit boundary is always explicit. Nothing here defaults it, because the
difference between "fit on train" and "fit on train+validation" is the
difference between model selection and final benchmarking.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import pandas as pd

from omnirank.core.logging import get_logger
from omnirank.data.processed import ProcessedDataset
from omnirank.evaluation.bootstrap import ConfidenceInterval, bootstrap_metric
from omnirank.evaluation.evaluator import (
    PRIMARY_METRICS,
    STRICT,
    WARM,
    EvaluationResult,
    OfflineEvaluator,
)
from omnirank.evaluation.ground_truth import EvaluationGroundTruth, build_ground_truth
from omnirank.evaluation.recommendations import RecommendationSet, UserRecommendations
from omnirank.evaluation.slices import SliceResult, evaluate_all_slices

logger = get_logger(__name__)


class BatchRecommender(Protocol):
    """The narrow surface :func:`generate_recommendations` needs."""

    def recommend_batch(
        self, user_ids: list[str], k: int, *, filter_seen: bool = True
    ) -> dict[str, list[str]]:
        """Top-``k`` external item ids per user."""
        ...


@dataclass(slots=True)
class RuntimeMeasurement:
    """Wall-clock and memory costs of one stage.

    Recorded per stage rather than as one total, because "training took 40s" and
    "recommendation took 40s" have completely different implications.
    """

    stage: str
    seconds: float
    peak_memory_mb: float | None = None
    items_processed: int | None = None

    @property
    def throughput_per_second(self) -> float | None:
        """Items processed per second, when a count was supplied."""
        if self.items_processed is None or self.seconds <= 0:
            return None
        return self.items_processed / self.seconds

    def to_dict(self) -> dict[str, Any]:
        """Report-ready payload."""
        payload: dict[str, Any] = {
            "stage": self.stage,
            "seconds": round(self.seconds, 4),
        }
        if self.peak_memory_mb is not None:
            payload["peak_memory_mb"] = round(self.peak_memory_mb, 2)
        if self.items_processed is not None:
            payload["items_processed"] = self.items_processed
            throughput = self.throughput_per_second
            if throughput is not None:
                payload["per_second"] = round(throughput, 2)
                payload["mean_ms_each"] = round(1000.0 / throughput, 4)
        return payload


class measure:
    """Context manager timing a stage, optionally tracking peak memory.

    Memory tracking uses ``tracemalloc``, which sees Python allocations only -
    torch tensor memory is not counted. Reported as such rather than presented
    as total process RSS.
    """

    def __init__(self, stage: str, *, track_memory: bool = False, items: int | None = None) -> None:
        self.stage = stage
        self.track_memory = track_memory
        self.items = items
        self.result: RuntimeMeasurement | None = None
        self._start = 0.0

    def __enter__(self) -> measure:
        if self.track_memory:
            tracemalloc.start()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        seconds = time.perf_counter() - self._start
        peak_mb = None
        if self.track_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / 1e6
        self.result = RuntimeMeasurement(
            stage=self.stage, seconds=seconds, peak_memory_mb=peak_mb, items_processed=self.items
        )


@dataclass(slots=True)
class ExperimentResult:
    """Everything one model-configuration run produced."""

    model_name: str
    model_version: str
    configuration: dict[str, Any]
    fit_splits: tuple[str, ...]
    target_split: str
    strict: EvaluationResult
    warm: EvaluationResult
    slices: list[SliceResult] = field(default_factory=list)
    runtimes: list[RuntimeMeasurement] = field(default_factory=list)
    intervals: dict[str, ConfidenceInterval] = field(default_factory=dict)
    dataset_identity: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def summary_row(self) -> dict[str, Any]:
        """One row for the model-comparison table."""
        strict_flat = self.strict.flat()
        warm_flat = self.warm.flat()
        runtimes = {item.stage: item for item in self.runtimes}
        return {
            "model": self.model_name,
            "version": self.model_version,
            "fit_splits": "+".join(self.fit_splits),
            "target_split": self.target_split,
            "recall@20_strict": round(strict_flat.get("recall@20", 0.0), 6),
            "ndcg@20_strict": round(strict_flat.get("ndcg@20", 0.0), 6),
            "recall@20_warm": round(warm_flat.get("recall@20", 0.0), 6),
            "ndcg@20_warm": round(warm_flat.get("ndcg@20", 0.0), 6),
            "coverage@20": round(strict_flat.get("coverage@20", 0.0), 6),
            "novelty@20": round(strict_flat.get("novelty@20", 0.0), 6),
            "gini@20": round(strict_flat.get("gini@20", 0.0), 6),
            "category_diversity@20": round(strict_flat.get("category_diversity@20", 0.0), 6),
            "reachable_fraction": round(self.strict.provenance.get("reachable_fraction", 0.0), 6),
            "users_evaluated": self.strict.users_evaluated,
            "train_seconds": round(runtimes["fit"].seconds, 3) if "fit" in runtimes else None,
            "recommend_seconds": (
                round(runtimes["recommend"].seconds, 3) if "recommend" in runtimes else None
            ),
            "evaluate_seconds": (
                round(runtimes["evaluate"].seconds, 3) if "evaluate" in runtimes else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Full JSON payload for the metrics reports."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "configuration": self.configuration,
            "fit_splits": list(self.fit_splits),
            "target_split": self.target_split,
            "protocol": "full_catalogue",
            "dataset_identity": self.dataset_identity,
            "strict": {
                "view": STRICT,
                "metrics": {k: round(v, 6) for k, v in self.strict.flat().items()},
                "users_evaluated": self.strict.users_evaluated,
                "user_coverage": round(self.strict.user_coverage, 6),
                "provenance": self.strict.provenance,
            },
            "warm": {
                "view": WARM,
                "metrics": {k: round(v, 6) for k, v in self.warm.flat().items()},
                "users_evaluated": self.warm.users_evaluated,
                "user_coverage": round(self.warm.user_coverage, 6),
            },
            "confidence_intervals": {
                name: interval.to_dict() for name, interval in self.intervals.items()
            },
            "slices": [item.to_dict() for item in self.slices],
            "runtimes": [item.to_dict() for item in self.runtimes],
            "beyond_accuracy_notes": {
                "intra_list_diversity": (
                    self.strict.beyond_accuracy[0].intra_list_diversity_unavailable_reason
                    if self.strict.beyond_accuracy
                    else None
                ),
            },
            **self.extra,
        }


def generate_recommendations(
    model: BatchRecommender,
    user_ids: Sequence[str],
    *,
    k: int,
    model_name: str,
    model_version: str,
    filter_seen: bool = True,
) -> tuple[RecommendationSet, RuntimeMeasurement]:
    """Produce top-``k`` recommendations for every user, timed.

    The timing is **offline batch throughput**, not serving latency: it measures
    a vectorised sweep over the whole population with no request overhead,
    network, or per-request model loading. Quoting it as a latency figure would
    be wrong by an order of magnitude in the flattering direction.
    """
    with measure("recommend", track_memory=True, items=len(user_ids)) as timer:
        raw = model.recommend_batch(list(user_ids), k, filter_seen=filter_seen)
    assert timer.result is not None  # noqa: S101
    recommendations = RecommendationSet(
        (
            UserRecommendations(user_id=user_id, item_ids=tuple(raw.get(user_id, ())))
            for user_id in user_ids
        ),
        model_name=model_name,
        model_version=model_version,
    )
    logger.info(
        "experiment.recommendations_generated",
        model=model_name,
        users=len(user_ids),
        empty_lists=len(recommendations.users_with_no_recommendations),
        **timer.result.to_dict(),
    )
    return recommendations, timer.result


def build_evaluation_inputs(
    dataset: ProcessedDataset,
    *,
    fit_splits: tuple[str, ...],
    target_split: str,
    fit_item_ids: set[int],
) -> tuple[EvaluationGroundTruth, pd.DataFrame]:
    """Build ground truth for one fit/target boundary."""
    fit_interactions = dataset.fit_interactions(fit_splits)
    targets = dataset.split(target_split)
    internal_to_external_item = dataset.internal_to_external_items()
    internal_to_external_user = {
        internal: external for external, internal in dataset.external_to_internal_users().items()
    }
    ground_truth = build_ground_truth(
        targets,
        target_split=target_split,
        fit_splits=fit_splits,
        fit_item_ids=fit_item_ids,
        internal_to_external_item=internal_to_external_item,
        internal_to_external_user=internal_to_external_user,
        fit_interactions=fit_interactions,
    )
    return ground_truth, fit_interactions


def evaluate_model(
    recommendations: RecommendationSet,
    ground_truth: EvaluationGroundTruth,
    dataset: ProcessedDataset,
    *,
    evaluator: OfflineEvaluator,
    k_values: Sequence[int],
    fit_item_ids: set[int],
    fit_interactions: pd.DataFrame,
    user_slices: dict[str, Collection[str]],
    target_item_slices: dict[str, Collection[int]],
    bootstrap_samples: int = 0,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 42,
) -> tuple[EvaluationResult, EvaluationResult, list[SliceResult], dict[str, ConfidenceInterval]]:
    """Run the strict and warm evaluations, slices, and bootstrap intervals."""
    internal_to_external = dataset.internal_to_external_items()
    eligible_catalogue = {internal_to_external[item] for item in fit_item_ids}

    # Novelty uses fit-window counts recomputed from the fit interactions, so it
    # matches the boundary this run actually used rather than Phase 2's
    # train-only table - which would be wrong for the final train+validation fit.
    counts = fit_interactions.groupby("internal_item_id", observed=True).size()
    training_counts: dict[str, int] = {}
    for item, count in counts.items():
        internal_item = int(cast("int", item))
        if internal_item in internal_to_external:
            training_counts[internal_to_external[internal_item]] = int(count)

    category_by_item: dict[str, str] = {}
    for row in dataset.item_categories.itertuples():
        internal_item = int(cast("int", row.internal_item_id))
        if internal_item in internal_to_external and pd.notna(row.category):
            category_by_item[internal_to_external[internal_item]] = str(row.category)

    with measure("evaluate") as timer:
        strict = evaluator.evaluate_detailed(
            recommendations,
            ground_truth,
            k_values=k_values,
            view=STRICT,
            eligible_catalogue=eligible_catalogue,
            training_counts=training_counts,
            category_by_item=category_by_item,
        )
        warm = evaluator.evaluate_detailed(
            recommendations,
            ground_truth,
            k_values=k_values,
            view=WARM,
            eligible_catalogue=eligible_catalogue,
            training_counts=training_counts,
            category_by_item=category_by_item,
        )
        slices = evaluate_all_slices(
            evaluator,
            recommendations,
            ground_truth,
            user_slices=user_slices,
            target_item_slices=target_item_slices,
            k_values=k_values,
        )

    intervals: dict[str, ConfidenceInterval] = {}
    if bootstrap_samples > 0:
        intervals = {
            metric: bootstrap_metric(
                strict.per_user,
                metric,
                samples=bootstrap_samples,
                confidence_level=bootstrap_confidence,
                seed=bootstrap_seed,
            )
            for metric in PRIMARY_METRICS
            if any(metric in values for values in strict.per_user.values())
        }
    assert timer.result is not None  # noqa: S101
    return strict, warm, slices, intervals


def resolve_user_slices(
    dataset: ProcessedDataset, slice_names: Sequence[str]
) -> dict[str, Collection[str]]:
    """Load Phase 2 user slices and convert their ids to external form."""
    from omnirank.data.processed import load_evaluation_slice

    internal_to_external_user = {
        internal: external for external, internal in dataset.external_to_internal_users().items()
    }
    resolved: dict[str, Collection[str]] = {}
    for name in slice_names:
        internal_ids = load_evaluation_slice(dataset.root, name)
        resolved[name] = {
            internal_to_external_user[item]
            for item in internal_ids
            if item in internal_to_external_user
        }
    return resolved


def resolve_item_slices(
    dataset: ProcessedDataset, slice_names: Sequence[str]
) -> dict[str, Collection[int]]:
    """Load Phase 2 item slices as internal id sets."""
    from omnirank.data.processed import load_evaluation_slice

    return {name: load_evaluation_slice(dataset.root, name) for name in slice_names}


__all__ = [
    "BatchRecommender",
    "ExperimentResult",
    "RuntimeMeasurement",
    "build_evaluation_inputs",
    "evaluate_model",
    "generate_recommendations",
    "measure",
    "resolve_item_slices",
    "resolve_user_slices",
]
