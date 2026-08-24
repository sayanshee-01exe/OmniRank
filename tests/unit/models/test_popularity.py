"""Popularity baseline: scoring, decay, filtering, and persistence."""

from __future__ import annotations

import pandas as pd
import pytest

from omnirank.core.exceptions import ArtifactValidationError, DataError, ModelNotFittedError
from omnirank.models.baselines.popularity import (
    GLOBAL_COUNT,
    TIME_DECAY,
    PopularityConfig,
    PopularityFitData,
    PopularityRecommender,
    build_seen_by_user,
)

DAY = 86_400


@pytest.fixture
def interactions() -> pd.DataFrame:
    """Item 10 is the most popular but its interactions are old; item 12 is new."""
    return pd.DataFrame(
        {
            "internal_user_id": [0, 0, 1, 1, 2, 2],
            "internal_item_id": [10, 11, 10, 12, 10, 12],
            "timestamp": [0, 0, 0, 100 * DAY, 0, 100 * DAY],
        }
    )


@pytest.fixture
def fit_data(interactions) -> PopularityFitData:
    return PopularityFitData(
        interactions=interactions,
        internal_to_external_item={10: "iA", 11: "iB", 12: "iC"},
        external_to_internal_user={"uA": 0, "uB": 1, "uC": 2},
        seen_by_user=build_seen_by_user(interactions),
        mapping_checksum="checksum-abc",
    )


class TestConfig:
    def test_rejects_unknown_variant(self):
        with pytest.raises(DataError):
            PopularityConfig(variant="mystery")

    @pytest.mark.parametrize("half_life", [0.0, -1.0])
    def test_rejects_invalid_half_life(self, half_life):
        with pytest.raises(DataError):
            PopularityConfig(TIME_DECAY, half_life_days=half_life)

    def test_global_variant_ignores_half_life(self):
        PopularityConfig(GLOBAL_COUNT, half_life_days=0.0)


class TestGlobalCount:
    def test_ranks_by_raw_count(self, fit_data):
        model = PopularityRecommender(PopularityConfig(GLOBAL_COUNT))
        model.fit(fit_data)
        assert [c.item_id for c in model.recommend("unknown", 3)] == ["iA", "iC", "iB"]

    def test_scores_are_counts(self, fit_data):
        model = PopularityRecommender(PopularityConfig(GLOBAL_COUNT))
        model.fit(fit_data)
        assert model.score("uA", ["iA", "iB", "iC"]) == [3.0, 1.0, 2.0]


class TestTimeDecay:
    def test_recent_items_overtake_older_more_popular_ones(self, fit_data):
        """Item 10 has 3 old interactions; item 12 has 2 recent ones."""
        model = PopularityRecommender(PopularityConfig(TIME_DECAY, half_life_days=10.0))
        model.fit(fit_data)
        assert next(c.item_id for c in model.recommend("unknown", 3)) == "iC"

    def test_decay_arithmetic_is_exact(self, fit_data):
        # half-life 100 days, reference = 100 days: the three age-100 events
        # each weigh 0.5, the two age-0 events weigh 1.0.
        model = PopularityRecommender(PopularityConfig(TIME_DECAY, half_life_days=100.0))
        model.fit(fit_data)
        assert model.score("uA", ["iA", "iC"]) == pytest.approx([1.5, 2.0])

    def test_reference_is_the_fit_maximum_not_the_wall_clock(self, fit_data):
        """Otherwise scores would change every time the model was loaded."""
        model = PopularityRecommender(PopularityConfig(TIME_DECAY))
        model.fit(fit_data)
        assert model.reference_timestamp == 100 * DAY

    def test_longer_half_life_approaches_global_counts(self, fit_data):
        model = PopularityRecommender(PopularityConfig(TIME_DECAY, half_life_days=1e6))
        model.fit(fit_data)
        assert model.score("uA", ["iA"])[0] == pytest.approx(3.0, abs=1e-3)


class TestRecommendation:
    def test_seen_items_excluded(self, fit_data):
        model = PopularityRecommender(PopularityConfig(GLOBAL_COUNT))
        model.fit(fit_data)
        # uA has seen items 10 and 11 -> only iC remains.
        assert [c.item_id for c in model.recommend("uA", 5)] == ["iC"]

    def test_seen_filtering_can_be_disabled(self, fit_data):
        model = PopularityRecommender(PopularityConfig(GLOBAL_COUNT))
        model.fit(fit_data)
        recs = model.recommend("uA", 5, {"filter_seen": False})
        assert len(recs) == 3

    def test_unknown_user_gets_the_global_list(self, fit_data):
        """The fallback must answer for anyone; that is its whole purpose."""
        model = PopularityRecommender(PopularityConfig(GLOBAL_COUNT))
        model.fit(fit_data)
        assert len(model.recommend("nobody", 3)) == 3

    def test_ties_break_on_item_id_deterministically(self):
        frame = pd.DataFrame(
            {"internal_user_id": [0, 0], "internal_item_id": [7, 3], "timestamp": [0, 0]}
        )
        data = PopularityFitData(frame, {3: "i3", 7: "i7"}, {"u": 0}, build_seen_by_user(frame))
        model = PopularityRecommender(PopularityConfig(GLOBAL_COUNT))
        model.fit(data)
        assert [c.item_id for c in model.recommend("nobody", 2)] == ["i3", "i7"]

    def test_batch_matches_single_user_calls(self, fit_data):
        model = PopularityRecommender(PopularityConfig(TIME_DECAY, half_life_days=30.0))
        model.fit(fit_data)
        batch = model.recommend_batch(["uA", "uB", "uC"], 3)
        for user in ("uA", "uB", "uC"):
            assert batch[user] == [c.item_id for c in model.recommend(user, 3)]

    def test_unknown_item_scores_zero(self, fit_data):
        model = PopularityRecommender(PopularityConfig(GLOBAL_COUNT))
        model.fit(fit_data)
        assert model.score("uA", ["not-an-item"]) == [0.0]

    def test_invalid_k_rejected(self, fit_data):
        model = PopularityRecommender()
        model.fit(fit_data)
        with pytest.raises(DataError):
            model.recommend("uA", 0)


class TestFittedState:
    def test_recommend_before_fit_raises(self):
        with pytest.raises(ModelNotFittedError):
            PopularityRecommender().recommend("uA", 5)

    def test_score_before_fit_raises(self):
        with pytest.raises(ModelNotFittedError):
            PopularityRecommender().score("uA", ["iA"])

    def test_empty_fit_data_rejected(self):
        empty = pd.DataFrame(columns=["internal_user_id", "internal_item_id", "timestamp"])
        with pytest.raises(DataError):
            PopularityRecommender().fit(PopularityFitData(empty, {}, {}, {}))

    def test_wrong_bundle_type_rejected(self):
        with pytest.raises(DataError):
            PopularityRecommender().fit({"not": "a bundle"})


class TestPersistence:
    def test_round_trip_preserves_recommendations(self, fit_data, tmp_path):
        model = PopularityRecommender(PopularityConfig(TIME_DECAY, half_life_days=30.0))
        model.fit(fit_data)
        before = model.recommend_batch(["uA", "uB", "uC"], 3)
        model.save(tmp_path / "model")
        loaded = PopularityRecommender.load(tmp_path / "model")
        assert loaded.recommend_batch(["uA", "uB", "uC"], 3) == before

    def test_round_trip_preserves_scores_exactly(self, fit_data, tmp_path):
        model = PopularityRecommender(PopularityConfig(TIME_DECAY, half_life_days=30.0))
        model.fit(fit_data)
        before = model.score("uA", ["iA", "iB", "iC"])
        model.save(tmp_path / "model")
        assert (
            PopularityRecommender.load(tmp_path / "model").score("uA", ["iA", "iB", "iC"]) == before
        )

    def test_round_trip_preserves_configuration(self, fit_data, tmp_path):
        model = PopularityRecommender(PopularityConfig(TIME_DECAY, half_life_days=42.0))
        model.fit(fit_data)
        model.save(tmp_path / "model")
        loaded = PopularityRecommender.load(tmp_path / "model")
        assert loaded.config.half_life_days == 42.0
        assert loaded.reference_timestamp == model.reference_timestamp
        assert loaded.is_fitted

    def test_missing_files_fail_clearly(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ArtifactValidationError):
            PopularityRecommender.load(tmp_path / "empty")

    def test_corrupted_config_fails_clearly(self, fit_data, tmp_path):
        model = PopularityRecommender()
        model.fit(fit_data)
        model.save(tmp_path / "model")
        (tmp_path / "model" / "config.json").write_text("{not json")
        with pytest.raises(ArtifactValidationError):
            PopularityRecommender.load(tmp_path / "model")

    def test_wrong_model_type_fails_clearly(self, fit_data, tmp_path):
        import json

        model = PopularityRecommender()
        model.fit(fit_data)
        model.save(tmp_path / "model")
        path = tmp_path / "model" / "config.json"
        payload = json.loads(path.read_text())
        payload["model"] = "something_else"
        path.write_text(json.dumps(payload))
        with pytest.raises(ArtifactValidationError) as exc:
            PopularityRecommender.load(tmp_path / "model")
        assert "different model type" in str(exc.value)

    def test_mapping_mismatch_fails_clearly(self, fit_data, tmp_path):
        model = PopularityRecommender()
        model.fit(fit_data)
        model.require_mapping("checksum-abc")
        with pytest.raises(ArtifactValidationError) as exc:
            model.require_mapping("a-different-checksum")
        assert "mapping checksum" in str(exc.value).lower()


class TestTrainOnlyComputation:
    def test_scores_use_only_the_supplied_fit_interactions(self, interactions):
        """A model fitted on a subset must not know about the rest."""
        subset = interactions[interactions.internal_item_id != 12]
        data = PopularityFitData(
            subset, {10: "iA", 11: "iB", 12: "iC"}, {"uA": 0}, build_seen_by_user(subset)
        )
        model = PopularityRecommender(PopularityConfig(GLOBAL_COUNT))
        model.fit(data)
        assert model.score("uA", ["iC"]) == [0.0]
        assert 12 not in model.fit_item_catalogue

    def test_never_recommends_outside_the_fit_catalogue(self, interactions):
        subset = interactions[interactions.internal_item_id != 12]
        data = PopularityFitData(
            subset, {10: "iA", 11: "iB", 12: "iC"}, {"uZ": 9}, build_seen_by_user(subset)
        )
        model = PopularityRecommender(PopularityConfig(GLOBAL_COUNT))
        model.fit(data)
        assert "iC" not in [c.item_id for c in model.recommend("uZ", 10)]


class TestDeterminism:
    def test_two_fits_agree(self, fit_data):
        first, second = PopularityRecommender(), PopularityRecommender()
        first.fit(fit_data)
        second.fit(fit_data)
        assert first.recommend_batch(["uA"], 3) == second.recommend_batch(["uA"], 3)

    def test_row_order_does_not_matter(self, interactions):
        def fit(frame):
            model = PopularityRecommender(PopularityConfig(TIME_DECAY, half_life_days=30.0))
            model.fit(
                PopularityFitData(
                    frame,
                    {10: "iA", 11: "iB", 12: "iC"},
                    {"uA": 0, "uB": 1, "uC": 2},
                    build_seen_by_user(frame),
                )
            )
            return model

        assert fit(interactions).recommend_batch(["uA"], 3) == (
            fit(interactions.iloc[::-1]).recommend_batch(["uA"], 3)
        )
