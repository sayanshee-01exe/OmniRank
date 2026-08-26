"""Shared fit/evaluate driver for the Phase 3 baselines.

Both CLIs and the comparison script go through here, so popularity and BPR are
always fitted and measured the same way. A comparison in which each model
brought its own harness would not be a comparison.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from omnirank.core.config import AppConfig
from omnirank.core.exceptions import DataError
from omnirank.core.logging import get_logger
from omnirank.data.processed import ProcessedDataset
from omnirank.evaluation.evaluator import OfflineEvaluator
from omnirank.evaluation.experiment import (
    ExperimentResult,
    build_evaluation_inputs,
    evaluate_model,
    generate_recommendations,
    measure,
    resolve_item_slices,
    resolve_user_slices,
)
from omnirank.evaluation.slices import TARGET_ITEM_SLICES, USER_ACTIVITY_SLICES
from omnirank.models.baselines.popularity import (
    PopularityConfig,
    PopularityFitData,
    PopularityRecommender,
    build_seen_by_user,
)

logger = get_logger(__name__)

#: Selection fits on training only and scores the validation targets.
SELECTION_BOUNDARY: tuple[tuple[str, ...], str] = (("train",), "validation")
#: The final benchmark fits on train+validation and scores test, exactly once.
FINAL_BOUNDARY: tuple[tuple[str, ...], str] = (("train", "validation"), "test")

POPULARITY = "popularity"
MATRIX_FACTORIZATION = "matrix_factorization"


def boundary_for_stage(stage: str) -> tuple[tuple[str, ...], str]:
    """Return ``(fit_splits, target_split)`` for a stage.

    Raises:
        DataError: Unknown stage. There are exactly two, and conflating them is
            the mistake this whole phase is arranged to prevent.
    """
    if stage == "selection":
        return SELECTION_BOUNDARY
    if stage == "final":
        return FINAL_BOUNDARY
    raise DataError("Unknown stage", stage=stage, available=["selection", "final"])


def fit_popularity(
    dataset: ProcessedDataset,
    fit_splits: tuple[str, ...],
    config: PopularityConfig,
    *,
    interactions: pd.DataFrame | None = None,
) -> tuple[PopularityRecommender, Any]:
    """Fit a popularity model on the given splits.

    ``interactions`` overrides the split-derived log, which is how a
    point-in-time snapshot fits popularity on one rolling fold's pre-cutoff
    history. Popularity is the most leak-prone source in the system precisely
    because it is the cheapest: counting the whole log takes no longer than
    counting a prefix of it, so the wrong version is never slow enough to
    notice.
    """
    fit_interactions = (
        dataset.fit_interactions(fit_splits) if interactions is None else interactions
    )
    data = PopularityFitData(
        interactions=fit_interactions,
        internal_to_external_item=dataset.internal_to_external_items(),
        external_to_internal_user=dataset.external_to_internal_users(),
        seen_by_user=build_seen_by_user(fit_interactions),
        mapping_checksum=dataset.mapping_metadata.get("item_mapping_checksum", ""),
        dataset_identity=dataset.identity.to_dict(),
    )
    model = PopularityRecommender(config)
    with measure("fit", track_memory=True, items=len(fit_interactions)) as timer:
        model.fit(data)
    return model, timer.result


def fit_bpr(
    dataset: ProcessedDataset,
    fit_splits: tuple[str, ...],
    config: Any,
    *,
    device: str = "auto",
    interactions: pd.DataFrame | None = None,
) -> tuple[Any, Any]:
    """Fit a BPR model on the given splits.

    ``interactions`` overrides the split-derived log, for point-in-time fits on
    a rolling fold's pre-cutoff history.

    Imported lazily so this module - and everything that only needs popularity -
    stays importable without the ``baseline`` extra installed.
    """
    from omnirank.models.baselines.bpr import BPRFitData, BPRMatrixFactorization

    fit_interactions = (
        dataset.fit_interactions(fit_splits) if interactions is None else interactions
    )
    data = BPRFitData(
        interactions=fit_interactions,
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        internal_to_external_item=dataset.internal_to_external_items(),
        external_to_internal_user=dataset.external_to_internal_users(),
        mapping_checksum=dataset.mapping_metadata.get("item_mapping_checksum", ""),
        dataset_identity=dataset.identity.to_dict(),
    )
    model = BPRMatrixFactorization(config, device=device)
    with measure("fit", track_memory=True, items=len(fit_interactions)) as timer:
        model.fit(data)
    return model, timer.result


def run_experiment(
    model: Any,
    dataset: ProcessedDataset,
    app_config: AppConfig,
    *,
    model_name: str,
    model_version: str,
    fit_splits: tuple[str, ...],
    target_split: str,
    fit_measurement: Any,
    configuration: dict[str, Any],
    k_values: Sequence[int] | None = None,
    bootstrap: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> ExperimentResult:
    """Generate recommendations, evaluate them, and package the result."""
    cuts = tuple(k_values or app_config.evaluation.k_values)
    max_k = max(cuts)
    fit_item_ids = model.fit_item_catalogue

    ground_truth, fit_interactions = build_evaluation_inputs(
        dataset, fit_splits=fit_splits, target_split=target_split, fit_item_ids=fit_item_ids
    )
    users = sorted(ground_truth.users)
    recommendations, recommend_measurement = generate_recommendations(
        model,
        users,
        k=max_k,
        model_name=model_name,
        model_version=model_version,
        filter_seen=app_config.evaluation.filter_seen,
    )

    bootstrap_config = app_config.evaluation.bootstrap
    use_bootstrap = bootstrap_config.enabled if bootstrap is None else bootstrap
    strict, warm, slices, intervals = evaluate_model(
        recommendations,
        ground_truth,
        dataset,
        evaluator=OfflineEvaluator(
            metrics=app_config.evaluation.metrics,
            novelty_smoothing=app_config.evaluation.beyond_accuracy.novelty_smoothing,
            gini_includes_zero_exposure=(
                app_config.evaluation.beyond_accuracy.gini_includes_zero_exposure
            ),
        ),
        k_values=cuts,
        fit_item_ids=fit_item_ids,
        fit_interactions=fit_interactions,
        user_slices=resolve_user_slices(dataset, USER_ACTIVITY_SLICES),
        target_item_slices=resolve_item_slices(dataset, TARGET_ITEM_SLICES),
        bootstrap_samples=bootstrap_config.samples if use_bootstrap else 0,
        bootstrap_confidence=bootstrap_config.confidence_level,
        bootstrap_seed=bootstrap_config.seed,
    )

    coverage = strict.user_coverage
    if coverage < app_config.evaluation.min_user_coverage:
        logger.warning(
            "experiment.low_user_coverage",
            model=model_name,
            user_coverage=round(coverage, 6),
            required=app_config.evaluation.min_user_coverage,
            detail=(
                "Fewer users received recommendations than the configured floor. "
                "The metrics below are still correct - users with no list score "
                "zero - but the model is not serving part of its population."
            ),
        )

    runtimes = [
        measurement
        for measurement in (fit_measurement, recommend_measurement)
        if measurement is not None
    ]
    return ExperimentResult(
        model_name=model_name,
        model_version=model_version,
        configuration=configuration,
        fit_splits=fit_splits,
        target_split=target_split,
        strict=strict,
        warm=warm,
        slices=slices,
        runtimes=runtimes,
        intervals=intervals,
        dataset_identity=dataset.identity.to_dict(),
        extra={
            # Deterministic, anonymised, and paired with explicit failures.
            "examples": recommendation_examples(recommendations, ground_truth),
            **(extra or {}),
        },
    )


def recommendation_examples(
    recommendations: Any, ground_truth: Any, *, count: int = 5, seed: int = 0
) -> dict[str, Any]:
    """A deterministic sample of users, plus deterministic failures and successes.

    Selection is a seeded draw over the **sorted** user list, so it depends only
    on the seed and not on which users happened to score well - the sample
    cannot be cherry-picked to flatter a model, and it reproduces exactly.

    Users are reported under stable pseudonyms. A public report does not need
    real identifiers to be useful, and the dataset licence gives no grounds for
    republishing them.

    Returns:
        Three groups, each a deterministic draw:

        * ``sampled`` - a neutral draw over all evaluated users. At a hit rate
          of about 1% this is almost entirely misses, which is the honest
          picture of the task.
        * ``failures`` - drawn only from users whose target was missed.
        * ``successes`` - drawn only from users whose target was found, so the
          report can show what a hit looks like. **Not representative**, and
          labelled as such; ``hit_rate_in_top_10`` gives the base rate it was
          drawn from, so nobody mistakes the sample for the population.

        Reporting all three is what stops a "here are some examples" section
        from quietly being a highlight reel.
    """
    users = sorted(ground_truth.users)
    if not users:
        return {"sampled": [], "failures": []}

    def pseudonym(user: str) -> str:
        """A stable, anonymous label for one user.

        Derived from the id, so the *same* user carries the *same* label across
        models - which is what makes "popularity found this user's target,
        BPR did not" a comparison rather than two unrelated lists. Positional
        labels would not survive that, and would also render two different
        samples identically.
        """
        return "user_" + hashlib.blake2b(user.encode(), digest_size=4).hexdigest()

    def describe(user: str) -> dict[str, Any]:
        target = set(ground_truth.truth.items_for(user))
        recommended = list(recommendations.items_for(user))
        top = recommended[:10]
        rank = next((index + 1 for index, item in enumerate(top) if item in target), None)
        return {
            "user": pseudonym(user),
            "target_rank_within_top_10": rank,
            "target_in_top_10": rank is not None,
            "recommended_count": len(recommended),
            "target_was_cold": user in ground_truth.cold_target_users,
        }

    generator = np.random.default_rng(seed)
    sampled_index = generator.choice(len(users), size=min(count, len(users)), replace=False)
    sampled = [describe(users[int(index)]) for index in sorted(sampled_index.tolist())]

    def hit(user: str) -> bool:
        return bool(
            set(ground_truth.truth.items_for(user))
            & set(list(recommendations.items_for(user))[:10])
        )

    missed = [user for user in users if not hit(user)]
    found = [user for user in users if hit(user)]

    def draw(population: list[str], offset: int) -> list[dict[str, Any]]:
        if not population:
            return []
        generator = np.random.default_rng(seed + offset)
        index = generator.choice(len(population), size=min(count, len(population)), replace=False)
        return [describe(population[int(item)]) for item in sorted(index.tolist())]

    return {
        "sampled": sampled,
        "failures": draw(missed, 1),
        "successes": draw(found, 2),
        "hit_rate_in_top_10": round(len(found) / len(users), 6) if users else 0.0,
    }


__all__ = [
    "FINAL_BOUNDARY",
    "MATRIX_FACTORIZATION",
    "POPULARITY",
    "SELECTION_BOUNDARY",
    "boundary_for_stage",
    "fit_bpr",
    "fit_popularity",
    "recommendation_examples",
    "run_experiment",
]
