"""Beyond-accuracy metrics: coverage, novelty, exposure inequality, diversity.

Accuracy alone rewards a model for learning popularity. These metrics expose the
cost of that: a recommender that shows the same 200 items to everyone can post a
respectable NDCG while covering 0.3% of the catalogue.

Every definition here states its denominator, because a coverage or Gini number
quoted without one is uninterpretable.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from omnirank.core.exceptions import DataError


@dataclass(frozen=True, slots=True)
class BeyondAccuracyResult:
    """Beyond-accuracy metrics at one cut-off, with their definitions."""

    k: int
    coverage: float
    unique_items_recommended: int
    eligible_catalogue_size: int
    novelty: float
    novelty_smoothing: float
    gini: float
    gini_includes_zero_exposure: bool
    category_diversity: float | None = None
    intra_list_diversity: float | None = None
    intra_list_diversity_unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Flat, report-ready payload."""
        payload: dict[str, Any] = {
            f"coverage@{self.k}": round(self.coverage, 6),
            f"novelty@{self.k}": round(self.novelty, 6),
            f"gini@{self.k}": round(self.gini, 6),
            f"unique_items@{self.k}": self.unique_items_recommended,
        }
        if self.category_diversity is not None:
            payload[f"category_diversity@{self.k}"] = round(self.category_diversity, 6)
        return payload


def catalogue_coverage(
    exposure_counts: Mapping[str, int], eligible_catalogue: Collection[str]
) -> tuple[float, int, int]:
    """Fraction of the eligible catalogue that was recommended at least once.

    Args:
        exposure_counts: item id -> number of users shown that item.
        eligible_catalogue: Every item the model could legitimately recommend -
            its fit-item catalogue, not the full dataset catalogue. Using the
            larger denominator would penalise a model for failing to recommend
            items it has never seen, which is the cold-start metric's job.

    Returns:
        ``(coverage, unique_recommended, catalogue_size)``.
    """
    size = len(eligible_catalogue)
    if size == 0:
        raise DataError("Eligible catalogue is empty; coverage has no denominator")
    eligible = set(eligible_catalogue)
    unique = len({item for item in exposure_counts if item in eligible})
    return unique / size, unique, size


def item_novelty(training_counts: Mapping[str, int], *, smoothing: float = 1.0) -> dict[str, float]:
    """Self-information ``-log2 p(i)`` per item, from **training** popularity.

    Args:
        training_counts: item id -> training interaction count. Validation and
            test counts must never appear here: novelty computed from the full
            log would encode which items become popular after the training
            window, exactly the leak Phase 2 spends its leakage checks preventing.
        smoothing: Additive constant applied to every count. Needed because an
            item with zero training interactions has ``p = 0`` and ``-log2 0``
            is infinite. Recorded alongside the metric so the number is never
            quoted without the rule that produced it.

    Returns:
        item id -> novelty in bits. Rarer items score higher.
    """
    if smoothing < 0:
        raise DataError("Novelty smoothing must be non-negative", smoothing=smoothing)
    total = sum(training_counts.values()) + smoothing * len(training_counts)
    if total <= 0:
        raise DataError("Cannot compute novelty from an empty popularity table")
    return {
        item: -math.log2((count + smoothing) / total) for item, count in training_counts.items()
    }


def mean_list_novelty(
    recommended_lists: Sequence[Sequence[str]], novelty_by_item: Mapping[str, float]
) -> float:
    """Mean per-item novelty, averaged first within a list then across users.

    Averaging within-then-across weights every user equally. Pooling all
    recommendations instead would let a user with a longer list dominate.
    Items with no novelty entry (outside the training catalogue) are skipped
    rather than treated as maximally novel.
    """
    per_user: list[float] = []
    for items in recommended_lists:
        values = [novelty_by_item[item] for item in items if item in novelty_by_item]
        if values:
            per_user.append(sum(values) / len(values))
        elif items:
            # A non-empty list of entirely unknown items contributes nothing
            # measurable rather than a fabricated zero.
            continue
    return sum(per_user) / len(per_user) if per_user else 0.0


def exposure_gini(
    exposure_counts: Mapping[str, int],
    eligible_catalogue: Collection[str],
    *,
    include_zero_exposure: bool = True,
) -> float:
    """Gini coefficient of recommendation exposure across the catalogue.

    0.0 means every eligible item was shown equally often; 1.0 means all
    exposure went to a single item. Recommenders sit high on this scale, and
    watching it move is how a diversity intervention is judged.

    Args:
        exposure_counts: item id -> times recommended.
        eligible_catalogue: The denominator population.
        include_zero_exposure: Count never-recommended eligible items as zero
            exposure. **True** by default: excluding them measures inequality
            only among the items a model already likes, which flatters a model
            that ignores the tail.
    """
    eligible = set(eligible_catalogue)
    if not eligible:
        raise DataError("Eligible catalogue is empty; Gini has no denominator")
    if include_zero_exposure:
        values = sorted(exposure_counts.get(item, 0) for item in eligible)
    else:
        values = sorted(count for item, count in exposure_counts.items() if item in eligible)
    count = len(values)
    total = sum(values)
    if count == 0 or total == 0:
        return 0.0
    # Standard ordered formulation: sum((2i - n - 1) * x_i) / (n * sum(x)).
    weighted = sum((2 * (index + 1) - count - 1) * value for index, value in enumerate(values))
    return weighted / (count * total)


def category_diversity(
    recommended_lists: Sequence[Sequence[str]], category_by_item: Mapping[str, str]
) -> float | None:
    """Mean fraction of distinct categories within a recommendation list.

    A defensible metadata-based diversity signal for PixelRec, which assigns each
    item exactly one tag from a 108-value vocabulary with 99.99% coverage.

    Deliberately **not** called intra-list diversity: that term means embedding
    similarity in the literature, and PixelRec's multimodal vectors are not
    available in this phase. Naming this one honestly keeps the two from being
    conflated when the real metric arrives.

    Returns:
        Mean over users of ``distinct_categories / list_length``, or ``None``
        when no list had a usable category.
    """
    per_user: list[float] = []
    for items in recommended_lists:
        categories = [category_by_item[item] for item in items if item in category_by_item]
        if categories:
            per_user.append(len(set(categories)) / len(categories))
    return sum(per_user) / len(per_user) if per_user else None


#: Why embedding-based intra-list diversity is not reported in this phase.
INTRA_LIST_DIVERSITY_UNAVAILABLE: Final = (
    "Embedding-based intra-list diversity requires item vectors. PixelRec "
    "publishes 1024-d text and image features as two ~8.6 GiB JSON files "
    "covering all 408,374 full-PixelRec items; they are not downloaded in this "
    "phase, so measured feature coverage is 0.0. Reporting 0.0 here would be "
    "indistinguishable from a genuinely undiverse recommender, so the metric is "
    "marked unavailable instead. `category_diversity@k` is reported in its "
    "place, from the 108-value item tag vocabulary. The embedding metric lands "
    "with the multimodal work in Phase 5."
)


def compute_beyond_accuracy(
    recommended_lists: Sequence[Sequence[str]],
    exposure_counts: Mapping[str, int],
    *,
    k: int,
    eligible_catalogue: Collection[str],
    training_counts: Mapping[str, int],
    category_by_item: Mapping[str, str] | None = None,
    novelty_smoothing: float = 1.0,
    gini_includes_zero_exposure: bool = True,
) -> BeyondAccuracyResult:
    """Compute every beyond-accuracy metric at one cut-off."""
    coverage, unique, size = catalogue_coverage(exposure_counts, eligible_catalogue)
    novelty_by_item = item_novelty(training_counts, smoothing=novelty_smoothing)
    return BeyondAccuracyResult(
        k=k,
        coverage=coverage,
        unique_items_recommended=unique,
        eligible_catalogue_size=size,
        novelty=mean_list_novelty(recommended_lists, novelty_by_item),
        novelty_smoothing=novelty_smoothing,
        gini=exposure_gini(
            exposure_counts,
            eligible_catalogue,
            include_zero_exposure=gini_includes_zero_exposure,
        ),
        gini_includes_zero_exposure=gini_includes_zero_exposure,
        category_diversity=(
            category_diversity(recommended_lists, category_by_item) if category_by_item else None
        ),
        intra_list_diversity=None,
        intra_list_diversity_unavailable_reason=INTRA_LIST_DIVERSITY_UNAVAILABLE,
    )


__all__ = [
    "INTRA_LIST_DIVERSITY_UNAVAILABLE",
    "BeyondAccuracyResult",
    "catalogue_coverage",
    "category_diversity",
    "compute_beyond_accuracy",
    "exposure_gini",
    "item_novelty",
    "mean_list_novelty",
]
