"""End-to-end baseline workflow over a small deterministic fixture.

    fixture data -> train popularity -> evaluate -> train BPR -> evaluate
                 -> save -> load -> re-evaluate -> compare identical

Offline, CPU-only, seconds. No PixelRec download, no GPU, no database, no
MLflow server, no multimodal vectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omnirank.evaluation.evaluator import STRICT, WARM, OfflineEvaluator
from omnirank.evaluation.ground_truth import build_ground_truth
from omnirank.evaluation.recommendations import RecommendationSet, UserRecommendations
from omnirank.models.baselines.popularity import (
    PopularityConfig,
    PopularityFitData,
    PopularityRecommender,
    build_seen_by_user,
)

pytestmark = pytest.mark.integration

USERS = 40
ITEMS = 30
DAY = 86_400


@pytest.fixture
def fixture_dataset() -> dict[str, pd.DataFrame]:
    """A tiny leave-last-one split with the shape Phase 2 produces.

    Each user has eight interactions from a cluster-preferring distribution;
    the last is the test target, the second-to-last is validation.
    """
    rng = np.random.default_rng(11)
    rows = []
    for user in range(USERS):
        block = user % 3
        for position in range(8):
            item = block * 10 + int(rng.integers(0, 10))
            rows.append(
                {
                    "internal_user_id": user,
                    "internal_item_id": item,
                    "interaction_order": position,
                    "timestamp": (position + 1) * 10 * DAY,
                    "event_type": "interaction",
                    "interaction_weight": 1.0,
                }
            )
    frame = pd.DataFrame(rows)
    frame["split"] = "train"
    frame.loc[frame.interaction_order == 6, "split"] = "validation"
    frame.loc[frame.interaction_order == 7, "split"] = "test"
    return {str(name): group.reset_index(drop=True) for name, group in frame.groupby("split")}


@pytest.fixture
def mappings() -> tuple[dict[int, str], dict[str, int]]:
    return (
        {index: f"i{index}" for index in range(ITEMS)},
        {f"u{user}": user for user in range(USERS)},
    )


def fit_popularity_model(dataset, mappings, splits=("train",)):
    item_map, user_map = mappings
    fit = pd.concat([dataset[name] for name in splits], ignore_index=True)
    model = PopularityRecommender(PopularityConfig("time_decay", half_life_days=30.0))
    model.fit(
        PopularityFitData(
            interactions=fit,
            internal_to_external_item=item_map,
            external_to_internal_user=user_map,
            seen_by_user=build_seen_by_user(fit),
            mapping_checksum="fixture-checksum",
        )
    )
    return model, fit


def make_ground_truth(dataset, mappings, fit, fit_items, target_split, fit_splits):
    item_map, user_map = mappings
    return build_ground_truth(
        dataset[target_split],
        target_split=target_split,
        fit_splits=fit_splits,
        fit_item_ids=fit_items,
        internal_to_external_item=item_map,
        internal_to_external_user={v: k for k, v in user_map.items()},
        fit_interactions=fit,
    )


def evaluate(model, ground_truth, *, k=10):
    users = sorted(ground_truth.users)
    raw = model.recommend_batch(users, k)
    recommendations = RecommendationSet(
        (UserRecommendations(user, tuple(raw.get(user, ()))) for user in users),
        model_name=model.name,
        model_version="fixture",
    )
    evaluator = OfflineEvaluator()
    strict = evaluator.evaluate_detailed(
        recommendations, ground_truth, k_values=[5, 10], view=STRICT
    )
    warm = evaluator.evaluate_detailed(recommendations, ground_truth, k_values=[5, 10], view=WARM)
    return recommendations, strict, warm


class TestPopularityWorkflow:
    def test_full_selection_workflow(self, fixture_dataset, mappings):
        model, fit = fit_popularity_model(fixture_dataset, mappings)
        truth = make_ground_truth(
            fixture_dataset, mappings, fit, model.fit_item_catalogue, "validation", ("train",)
        )
        _, strict, warm = evaluate(model, truth)
        assert strict.users_evaluated == USERS
        assert 0.0 <= strict.metrics["ndcg@10"] <= 1.0
        assert warm.metrics["recall@10"] >= strict.metrics["recall@10"]

    def test_save_load_reevaluate_is_identical(self, fixture_dataset, mappings, tmp_path):
        model, fit = fit_popularity_model(fixture_dataset, mappings)
        truth = make_ground_truth(
            fixture_dataset, mappings, fit, model.fit_item_catalogue, "validation", ("train",)
        )
        _, before, _ = evaluate(model, truth)

        model.save(tmp_path / "popularity")
        loaded = PopularityRecommender.load(tmp_path / "popularity")
        _, after, _ = evaluate(loaded, truth)
        assert after.metrics == before.metrics

    def test_final_stage_uses_a_wider_fit_boundary(self, fixture_dataset, mappings):
        selection, _ = fit_popularity_model(fixture_dataset, mappings, ("train",))
        final, _ = fit_popularity_model(fixture_dataset, mappings, ("train", "validation"))
        assert len(final.fit_item_catalogue) >= len(selection.fit_item_catalogue)
        assert final.reference_timestamp >= selection.reference_timestamp

    def test_seen_items_never_recommended(self, fixture_dataset, mappings):
        model, fit = fit_popularity_model(fixture_dataset, mappings)
        seen = build_seen_by_user(fit)
        for user, items in model.recommend_batch([f"u{u}" for u in range(USERS)], 10).items():
            observed = seen[int(user[1:])]
            assert not ({int(item[1:]) for item in items} & observed)


class TestBPRWorkflow:
    def test_full_workflow_and_persistence(self, fixture_dataset, mappings, tmp_path):
        pytest.importorskip("torch", reason="BPR requires the 'baseline' extra")
        from omnirank.models.baselines.bpr import BPRConfig, BPRFitData, BPRMatrixFactorization

        item_map, user_map = mappings
        fit = fixture_dataset["train"]
        model = BPRMatrixFactorization(
            BPRConfig(embedding_dim=8, epochs=6, batch_size=64, learning_rate=0.05, seed=5),
            device="cpu",
        )
        model.fit(
            BPRFitData(
                interactions=fit,
                num_users=USERS,
                num_items=ITEMS,
                internal_to_external_item=item_map,
                external_to_internal_user=user_map,
                mapping_checksum="fixture-checksum",
            )
        )
        truth = make_ground_truth(
            fixture_dataset, mappings, fit, model.fit_item_catalogue, "validation", ("train",)
        )
        _, before, _ = evaluate(model, truth)
        assert model.loss_history[-1] < model.loss_history[0]

        model.save(tmp_path / "bpr")
        loaded = BPRMatrixFactorization.load(tmp_path / "bpr", device="cpu")
        _, after, _ = evaluate(loaded, truth)
        assert after.metrics == before.metrics

    def test_both_models_are_evaluated_identically(self, fixture_dataset, mappings):
        """The comparison is only valid because one harness measures both."""
        pytest.importorskip("torch", reason="BPR requires the 'baseline' extra")
        from omnirank.models.baselines.bpr import BPRConfig, BPRFitData, BPRMatrixFactorization

        item_map, user_map = mappings
        popularity, fit = fit_popularity_model(fixture_dataset, mappings)
        bpr = BPRMatrixFactorization(
            BPRConfig(embedding_dim=8, epochs=4, batch_size=64, seed=2), device="cpu"
        )
        bpr.fit(
            BPRFitData(fit, USERS, ITEMS, item_map, user_map, mapping_checksum="fixture-checksum")
        )
        truth = make_ground_truth(
            fixture_dataset, mappings, fit, popularity.fit_item_catalogue, "validation", ("train",)
        )
        _, popularity_result, _ = evaluate(popularity, truth)
        _, bpr_result, _ = evaluate(bpr, truth)
        assert popularity_result.users_evaluated == bpr_result.users_evaluated
        assert set(popularity_result.metrics) == set(bpr_result.metrics)


class TestSlicesAndBootstrap:
    def test_slices_partition_the_population(self, fixture_dataset, mappings):
        from omnirank.evaluation.slices import evaluate_all_slices

        model, fit = fit_popularity_model(fixture_dataset, mappings)
        truth = make_ground_truth(
            fixture_dataset, mappings, fit, model.fit_item_catalogue, "validation", ("train",)
        )
        recommendations, _, _ = evaluate(model, truth)
        results = evaluate_all_slices(
            OfflineEvaluator(),
            recommendations,
            truth,
            user_slices={"all_users": truth.users, "empty_slice": set()},
            target_item_slices={"every_item": set(range(ITEMS))},
            k_values=[10],
        )
        by_name = {item.slice_name: item for item in results}
        assert by_name["all_users"].users == USERS
        assert by_name["empty_slice"].empty is True
        assert by_name["targets_reachable_warm"].users == len(truth.warm_users)

    def test_bootstrap_interval_is_reproducible(self, fixture_dataset, mappings):
        from omnirank.evaluation.bootstrap import bootstrap_metric

        model, fit = fit_popularity_model(fixture_dataset, mappings)
        truth = make_ground_truth(
            fixture_dataset, mappings, fit, model.fit_item_catalogue, "validation", ("train",)
        )
        _, strict, _ = evaluate(model, truth)
        first = bootstrap_metric(strict.per_user, "recall@10", samples=200, seed=1)
        second = bootstrap_metric(strict.per_user, "recall@10", samples=200, seed=1)
        assert (first.lower, first.upper) == (second.lower, second.upper)
