"""Deterministic user-level bootstrap confidence intervals.

A single point estimate over 50,000 users says nothing about whether a 0.002
NDCG gap is real. Resampling users - the unit of independence in this
evaluation - gives an interval that does.

Two forms:

* :func:`bootstrap_metric` - an interval for one model's metric.
* :func:`paired_bootstrap_delta` - an interval for the *difference* between two
  models, resampling the **same** users for both. Paired resampling removes the
  between-user variance that dominates the unpaired intervals, which is why two
  overlapping individual intervals can still accompany a delta that excludes
  zero.

Both are deterministic given the seed, so a reported interval is reproducible
rather than a property of one lucky run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from omnirank.core.exceptions import DataError


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A bootstrap interval for one metric."""

    metric: str
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    samples: int
    users: int

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies entirely on one side of zero.

        The only basis on which this codebase claims a difference is real. An
        interval containing zero is reported as inconclusive, never as a win.
        """
        return (self.lower > 0.0) or (self.upper < 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Report-ready payload."""
        return {
            "metric": self.metric,
            "point_estimate": round(self.point_estimate, 6),
            "ci_lower": round(self.lower, 6),
            "ci_upper": round(self.upper, 6),
            "confidence_level": self.confidence_level,
            "bootstrap_samples": self.samples,
            "users": self.users,
            "excludes_zero": self.excludes_zero,
        }


def _percentiles(confidence_level: float) -> tuple[float, float]:
    """Two-sided percentile bounds for a confidence level."""
    if not 0.0 < confidence_level < 1.0:
        raise DataError(
            "confidence_level must be strictly between 0 and 1",
            confidence_level=confidence_level,
        )
    tail = (1.0 - confidence_level) / 2.0
    return tail * 100.0, (1.0 - tail) * 100.0


def _resample_means(values: np.ndarray, *, samples: int, seed: int) -> np.ndarray:
    """Bootstrap means of ``values``, resampling users with replacement."""
    if samples < 1:
        raise DataError("Bootstrap samples must be >= 1", samples=samples)
    generator = np.random.default_rng(seed)
    population = len(values)
    indices = generator.integers(0, population, size=(samples, population))
    means: np.ndarray = values[indices].mean(axis=1)
    return means


def bootstrap_metric(
    per_user: Mapping[str, Mapping[str, float]],
    metric: str,
    *,
    samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> ConfidenceInterval:
    """Confidence interval for one metric's user-averaged value.

    Args:
        per_user: user -> {metric name: value}, as produced by the evaluator.
        metric: Flat metric key, e.g. ``"ndcg@20"``.
        samples: Bootstrap resamples.
        confidence_level: Two-sided coverage.
        seed: Fixed for reproducibility.

    Raises:
        DataError: No user carries the requested metric.
    """
    values = np.array(
        [user_values[metric] for user_values in per_user.values() if metric in user_values],
        dtype="float64",
    )
    if values.size == 0:
        raise DataError("No per-user values for metric", metric=metric)

    means = _resample_means(values, samples=samples, seed=seed)
    low, high = _percentiles(confidence_level)
    return ConfidenceInterval(
        metric=metric,
        point_estimate=float(values.mean()),
        lower=float(np.percentile(means, low)),
        upper=float(np.percentile(means, high)),
        confidence_level=confidence_level,
        samples=samples,
        users=int(values.size),
    )


def paired_bootstrap_delta(
    per_user_a: Mapping[str, Mapping[str, float]],
    per_user_b: Mapping[str, Mapping[str, float]],
    metric: str,
    *,
    samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> ConfidenceInterval:
    """Confidence interval for ``mean(a) - mean(b)`` over shared users.

    Only users present in both sets are used, and the same resampled user
    indices apply to both models. Users are sorted before resampling so the
    result does not depend on dict insertion order.

    Raises:
        DataError: The two sets share no users carrying the metric.
    """
    shared = sorted(
        user
        for user in set(per_user_a) & set(per_user_b)
        if metric in per_user_a[user] and metric in per_user_b[user]
    )
    if not shared:
        raise DataError("Models share no evaluated users for metric", metric=metric)

    values_a = np.array([per_user_a[user][metric] for user in shared], dtype="float64")
    values_b = np.array([per_user_b[user][metric] for user in shared], dtype="float64")
    deltas = values_a - values_b

    means = _resample_means(deltas, samples=samples, seed=seed)
    low, high = _percentiles(confidence_level)
    return ConfidenceInterval(
        metric=f"delta_{metric}",
        point_estimate=float(deltas.mean()),
        lower=float(np.percentile(means, low)),
        upper=float(np.percentile(means, high)),
        confidence_level=confidence_level,
        samples=samples,
        users=len(shared),
    )


def bootstrap_primary_metrics(
    per_user: Mapping[str, Mapping[str, float]],
    metrics: Sequence[str],
    *,
    samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, ConfidenceInterval]:
    """Intervals for several metrics at once."""
    return {
        metric: bootstrap_metric(
            per_user, metric, samples=samples, confidence_level=confidence_level, seed=seed
        )
        for metric in metrics
    }


__all__ = [
    "ConfidenceInterval",
    "bootstrap_metric",
    "bootstrap_primary_metrics",
    "paired_bootstrap_delta",
]
