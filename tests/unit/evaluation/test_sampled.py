"""Sampled-negative evaluation.

This protocol exists to be *fast*, not to be reported. Ranking a target against
a hundred sampled items instead of a 69,347-item catalogue produces numbers
roughly an order of magnitude higher, and those numbers are not comparable to a
full-catalogue result or to anything published. The tests that matter most here
are therefore the refusals: that a sampled number cannot reach a final stage,
and that a pool never contains an item the user has already seen.

The determinism tests carry similar weight. Two models compared under this
protocol are only comparable if they ranked within the *identical* pool, so a
pool that depended on dict iteration order would silently invalidate every
comparison drawn from it.
"""

from __future__ import annotations

import pytest

from omnirank.core.exceptions import DataError
from omnirank.evaluation.sampled import (
    FULL_CATALOGUE,
    SAMPLED,
    SampledCandidateSet,
    assert_not_final,
    build_sampled_candidates,
    restrict_to_pool,
    warn_if_incomparable,
)

CATALOGUE = [f"i{index}" for index in range(200)]


@pytest.fixture
def targets() -> dict[str, str]:
    return {"u1": "i10", "u2": "i20", "u3": "i30"}


@pytest.fixture
def seen() -> dict[str, set[str]]:
    return {"u1": {"i1", "i2", "i3"}, "u2": {"i5"}, "u3": set()}


@pytest.fixture
def pool(targets: dict[str, str], seen: dict[str, set[str]]) -> SampledCandidateSet:
    return build_sampled_candidates(
        targets=targets, seen_by_user=seen, catalogue=CATALOGUE, num_negatives=20, seed=7
    )


class TestPoolConstruction:
    def test_pool_is_the_target_plus_the_requested_negatives(
        self, pool: SampledCandidateSet
    ) -> None:
        assert len(pool.pool_for("u1")) == 21

    def test_target_is_present_and_first(
        self, pool: SampledCandidateSet, targets: dict[str, str]
    ) -> None:
        """Target first is what makes construction order-independent."""
        for user, target in targets.items():
            assert pool.pool_for(user)[0] == target

    def test_pool_never_contains_a_seen_item(
        self, pool: SampledCandidateSet, seen: dict[str, set[str]]
    ) -> None:
        """A 'negative' the user already interacted with is not a negative."""
        for user, items in seen.items():
            assert not set(pool.pool_for(user)) & items

    def test_negatives_are_distinct(self, pool: SampledCandidateSet) -> None:
        """A duplicated negative shrinks the pool without saying so."""
        for user in ("u1", "u2", "u3"):
            candidates = pool.pool_for(user)
            assert len(set(candidates)) == len(candidates)

    def test_target_appears_exactly_once(
        self, pool: SampledCandidateSet, targets: dict[str, str]
    ) -> None:
        for user, target in targets.items():
            assert pool.pool_for(user).count(target) == 1

    def test_unknown_user_gets_an_empty_pool(self, pool: SampledCandidateSet) -> None:
        assert pool.pool_for("stranger") == ()


class TestDeterminism:
    def test_same_seed_produces_identical_pools(
        self, targets: dict[str, str], seen: dict[str, set[str]]
    ) -> None:
        """Without this, two sampled numbers are not comparable to each other."""

        def build() -> SampledCandidateSet:
            return build_sampled_candidates(
                targets=targets,
                seen_by_user=seen,
                catalogue=CATALOGUE,
                num_negatives=20,
                seed=7,
            )

        assert dict(build().candidates) == dict(build().candidates)

    def test_different_seed_produces_different_pools(
        self, targets: dict[str, str], seen: dict[str, set[str]]
    ) -> None:
        other = build_sampled_candidates(
            targets=targets, seen_by_user=seen, catalogue=CATALOGUE, num_negatives=20, seed=8
        )
        first = build_sampled_candidates(
            targets=targets, seen_by_user=seen, catalogue=CATALOGUE, num_negatives=20, seed=7
        )
        assert other.pool_for("u1") != first.pool_for("u1")

    def test_pools_do_not_depend_on_target_insertion_order(
        self, targets: dict[str, str], seen: dict[str, set[str]]
    ) -> None:
        """Users are sorted internally, so dict order must not reach the draw."""
        forward = build_sampled_candidates(
            targets=targets, seen_by_user=seen, catalogue=CATALOGUE, num_negatives=20, seed=7
        )
        backward = build_sampled_candidates(
            targets=dict(reversed(list(targets.items()))),
            seen_by_user=seen,
            catalogue=CATALOGUE,
            num_negatives=20,
            seed=7,
        )
        assert dict(forward.candidates) == dict(backward.candidates)


class TestValidation:
    def test_rejects_a_non_positive_negative_count(
        self, targets: dict[str, str], seen: dict[str, set[str]]
    ) -> None:
        with pytest.raises(DataError):
            build_sampled_candidates(
                targets=targets, seen_by_user=seen, catalogue=CATALOGUE, num_negatives=0
            )

    def test_rejects_a_catalogue_too_small_to_sample_from(
        self, targets: dict[str, str], seen: dict[str, set[str]]
    ) -> None:
        with pytest.raises(DataError):
            build_sampled_candidates(
                targets=targets,
                seen_by_user=seen,
                catalogue=["i1", "i2"],
                num_negatives=50,
            )

    def test_a_user_who_has_seen_almost_everything_is_reported(self) -> None:
        """Better a named failure than a quietly short pool."""
        small = [f"i{index}" for index in range(30)]
        with pytest.raises(DataError):
            build_sampled_candidates(
                targets={"u1": "i0"},
                seen_by_user={"u1": set(small[1:])},
                catalogue=small,
                num_negatives=20,
            )


class TestRestrictToPool:
    def test_keeps_only_pool_items_in_original_order(self, pool: SampledCandidateSet) -> None:
        candidates = pool.pool_for("u1")
        recommendations = {"u1": [candidates[3], "i999", candidates[1], "i998"]}
        assert restrict_to_pool(recommendations, pool)["u1"] == [
            candidates[3],
            candidates[1],
        ]

    def test_a_user_outside_the_pool_keeps_nothing(self, pool: SampledCandidateSet) -> None:
        assert restrict_to_pool({"stranger": ["i1", "i2"]}, pool)["stranger"] == []

    def test_ranking_within_the_pool_is_preserved(self, pool: SampledCandidateSet) -> None:
        """The model still ranks the catalogue; only the scoring set narrows."""
        candidates = list(pool.pool_for("u2"))
        assert restrict_to_pool({"u2": candidates}, pool)["u2"] == candidates


class TestFinalStageRefusal:
    """The refusal that keeps a fast protocol from becoming a reported number."""

    @pytest.mark.parametrize("stage", ["final", "test", "report"])
    def test_sampled_is_refused_at_reporting_stages(self, stage: str) -> None:
        with pytest.raises(DataError):
            assert_not_final(SAMPLED, stage)

    @pytest.mark.parametrize("stage", ["selection", "validation", "exploration"])
    def test_sampled_is_allowed_at_selection_stages(self, stage: str) -> None:
        assert_not_final(SAMPLED, stage)

    @pytest.mark.parametrize("stage", ["final", "test", "report", "selection"])
    def test_full_catalogue_is_allowed_everywhere(self, stage: str) -> None:
        assert_not_final(FULL_CATALOGUE, stage)


class TestIncomparabilityWarning:
    def test_warns_when_protocols_are_mixed(self, caplog) -> None:
        warn_if_incomparable([SAMPLED, FULL_CATALOGUE])
        assert "incomparable_protocols" in caplog.text

    def test_silent_when_every_result_is_sampled(self, caplog) -> None:
        """All-sampled is internally comparable; only the mixture is not."""
        warn_if_incomparable([SAMPLED, SAMPLED])
        assert "incomparable_protocols" not in caplog.text

    def test_silent_when_every_result_is_full_catalogue(self, caplog) -> None:
        warn_if_incomparable([FULL_CATALOGUE, FULL_CATALOGUE])
        assert "incomparable_protocols" not in caplog.text


class TestDescription:
    def test_description_travels_with_the_metrics(self, pool: SampledCandidateSet) -> None:
        """A sampled number without its protocol attached will be misread."""
        described = pool.describe()
        assert described["protocol"] == SAMPLED
        assert described["num_negatives"] == 20
        assert described["seed"] == 7
        assert described["comparable_to_full_catalogue"] is False
