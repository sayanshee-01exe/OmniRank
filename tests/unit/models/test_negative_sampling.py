"""Negative sampling correctness.

The central property - never sampling a known positive - is a silent failure if
broken: training against false negatives teaches the model to rank genuine
preferences downward, and nothing in the loss curve reveals it.
"""

from __future__ import annotations

import numpy as np
import pytest

from omnirank.core.exceptions import DataError
from omnirank.models.baselines.negative_sampling import (
    UniformNegativeSampler,
    build_positives_by_user,
)


@pytest.fixture
def positives() -> dict[int, np.ndarray]:
    users = np.array([0, 0, 1, 1, 2, 2, 2])
    items = np.array([1, 3, 0, 2, 0, 1, 2])
    return build_positives_by_user(users, items)


@pytest.fixture
def sampler(positives) -> UniformNegativeSampler:
    return UniformNegativeSampler(positives, catalogue_size=8, seed=42)


class TestGrouping:
    def test_groups_by_user(self, positives):
        assert positives[0].tolist() == [1, 3]
        assert positives[2].tolist() == [0, 1, 2]

    def test_items_are_sorted_and_unique(self):
        grouped = build_positives_by_user(np.array([0, 0, 0]), np.array([5, 1, 5]))
        assert grouped[0].tolist() == [1, 5]


class TestCorrectness:
    def test_never_samples_a_known_positive(self, sampler, positives):
        users = np.array([0, 1, 2] * 40)
        drawn = sampler.sample(users, 3)
        for row, user in enumerate(users.tolist()):
            for item in drawn[row]:
                assert item not in positives[user], (user, item)

    def test_output_shape(self, sampler):
        assert sampler.sample(np.array([0, 1]), 5).shape == (2, 5)

    def test_all_indices_in_range(self, sampler):
        drawn = sampler.sample(np.array([0, 1, 2] * 20), 4)
        assert drawn.min() >= 0
        assert drawn.max() < 8

    def test_empty_batch(self, sampler):
        assert sampler.sample(np.array([], dtype="int64"), 3).shape == (0, 3)

    def test_user_with_no_positives_can_sample_anything(self):
        sampler = UniformNegativeSampler({}, catalogue_size=5, seed=1)
        assert sampler.sample(np.array([99]), 3).shape == (1, 3)


class TestDeterminism:
    def test_same_seed_same_samples(self, positives):
        first = UniformNegativeSampler(positives, catalogue_size=8, seed=7)
        second = UniformNegativeSampler(positives, catalogue_size=8, seed=7)
        users = np.array([0, 1, 2])
        assert np.array_equal(first.sample(users, 3), second.sample(users, 3))

    def test_different_seeds_differ(self, positives):
        first = UniformNegativeSampler(positives, catalogue_size=50, seed=1)
        second = UniformNegativeSampler(positives, catalogue_size=50, seed=2)
        users = np.array([0, 1, 2] * 10)
        assert not np.array_equal(first.sample(users, 3), second.sample(users, 3))

    def test_reset_replays_the_stream(self, sampler):
        users = np.array([0, 1, 2])
        first = sampler.sample(users, 3)
        sampler.reset()
        assert np.array_equal(sampler.sample(users, 3), first)


class TestDenseUsers:
    def test_user_seeing_all_but_one_item_terminates_with_the_right_answer(self):
        """Rejection sampling alone would spin here; the complement path resolves it."""
        sampler = UniformNegativeSampler({0: np.arange(9)}, catalogue_size=10, seed=1)
        drawn = sampler.sample(np.array([0] * 50), 3)
        assert set(drawn.flatten().tolist()) == {9}

    def test_user_seeing_all_but_two_items(self):
        sampler = UniformNegativeSampler({0: np.arange(8)}, catalogue_size=10, seed=1)
        drawn = sampler.sample(np.array([0] * 50), 2)
        assert set(drawn.flatten().tolist()) <= {8, 9}

    def test_full_catalogue_user_is_rejected_at_construction(self):
        with pytest.raises(DataError) as exc:
            UniformNegativeSampler({0: np.arange(5)}, catalogue_size=5)
        assert "entire catalogue" in str(exc.value)


class TestValidation:
    def test_tiny_catalogue_rejected(self):
        with pytest.raises(DataError):
            UniformNegativeSampler({}, catalogue_size=1)

    @pytest.mark.parametrize("count", [0, -1])
    def test_invalid_negative_count_rejected(self, sampler, count):
        with pytest.raises(DataError):
            sampler.sample(np.array([0]), count)

    def test_configuration_is_recorded_for_the_manifest(self, sampler):
        config = sampler.configuration
        assert config["strategy"] == "uniform"
        assert config["seed"] == 42
        assert config["catalogue_size"] == 8


class TestScale:
    def test_many_users_and_negatives_stay_correct(self):
        """Vectorised collision detection must not lose the core guarantee."""
        rng = np.random.default_rng(0)
        users = np.repeat(np.arange(200), 5)
        items = rng.integers(0, 500, size=users.size)
        positives = build_positives_by_user(users, items)
        sampler = UniformNegativeSampler(positives, catalogue_size=500, seed=3)
        batch = rng.integers(0, 200, size=2000)
        drawn = sampler.sample(batch, 4)
        for row, user in enumerate(batch.tolist()):
            known = set(positives[user].tolist())
            assert not (set(drawn[row].tolist()) & known)
