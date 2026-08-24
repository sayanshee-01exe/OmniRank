"""Recommendation storage invariants."""

from __future__ import annotations

import pytest

from omnirank.core.exceptions import DataError
from omnirank.evaluation.recommendations import RecommendationSet, UserRecommendations


class TestUserRecommendations:
    def test_preserves_order(self):
        entry = UserRecommendations("u1", ("c", "a", "b"))
        assert entry.item_ids == ("c", "a", "b")

    def test_rejects_duplicates_rather_than_deduplicating(self):
        """Silent deduplication would hide a model bug and inflate metrics."""
        with pytest.raises(DataError) as exc:
            UserRecommendations("u1", ("a", "b", "a"))
        assert "duplicate" in str(exc.value).lower()

    def test_names_the_duplicate(self):
        with pytest.raises(DataError) as exc:
            UserRecommendations("u1", ("a", "b", "a"))
        assert "a" in str(exc.value)

    def test_empty_list_is_allowed(self):
        assert len(UserRecommendations("u1", ())) == 0

    def test_scores_must_align_with_items(self):
        with pytest.raises(DataError):
            UserRecommendations("u1", ("a", "b"), scores=(1.0,))

    def test_aligned_scores_accepted(self):
        entry = UserRecommendations("u1", ("a", "b"), scores=(2.0, 1.0))
        assert entry.scores == (2.0, 1.0)


class TestRecommendationSet:
    def test_lookup_and_users(self):
        rec = RecommendationSet.from_mapping({"u1": ["a", "b"], "u2": ["c"]})
        assert set(rec.users()) == {"u1", "u2"}
        assert rec.items_for("u1") == ("a", "b")

    def test_unknown_user_returns_empty(self):
        rec = RecommendationSet.from_mapping({"u1": ["a"]})
        assert rec.items_for("nobody") == ()

    def test_duplicate_user_is_rejected(self):
        rec = RecommendationSet()
        rec.add(UserRecommendations("u1", ("a",)))
        with pytest.raises(DataError):
            rec.add(UserRecommendations("u1", ("b",)))

    def test_users_with_no_recommendations_are_reported(self):
        rec = RecommendationSet.from_mapping({"u1": ["a"], "u2": []})
        assert rec.users_with_no_recommendations == ("u2",)

    def test_exposure_counts(self):
        rec = RecommendationSet.from_mapping({"u1": ["a", "b"], "u2": ["a"]})
        assert rec.exposure_counts() == {"a": 2, "b": 1}

    def test_truncation(self):
        rec = RecommendationSet.from_mapping({"u1": ["a", "b", "c"]})
        assert rec.truncated(2).items_for("u1") == ("a", "b")

    def test_truncation_rejects_invalid_length(self):
        with pytest.raises(DataError):
            RecommendationSet.from_mapping({"u1": ["a"]}).truncated(0)

    def test_total_recommendations(self):
        rec = RecommendationSet.from_mapping({"u1": ["a", "b"], "u2": ["c"]})
        assert rec.total_recommendations == 3


class TestSerialisation:
    def test_round_trip(self, tmp_path):
        rec = RecommendationSet.from_mapping(
            {"u2": ["c", "d"], "u1": ["a", "b"]},
            scores={"u1": [2.0, 1.0]},
            model_name="popularity",
            model_version="v1",
        )
        path = rec.save(tmp_path / "recs.json")
        loaded = RecommendationSet.load(path)
        assert loaded.items_for("u1") == ("a", "b")
        assert loaded.scores_for("u1") == (2.0, 1.0)
        assert loaded.model_name == "popularity"

    def test_serialisation_is_deterministic(self, tmp_path):
        """Byte-identical output is what makes save/load equality testable."""
        forward = RecommendationSet.from_mapping({"u1": ["a"], "u2": ["b"]})
        reverse = RecommendationSet.from_mapping({"u2": ["b"], "u1": ["a"]})
        assert forward.save(tmp_path / "a.json").read_text() == (
            reverse.save(tmp_path / "b.json").read_text()
        )

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DataError):
            RecommendationSet.load(tmp_path / "absent.json")

    def test_malformed_file_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(DataError):
            RecommendationSet.load(path)
