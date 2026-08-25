"""Fitting the same configuration twice must produce the same model.

This is not a general "is training deterministic" test. It targets one specific
defect: parameter initialisation draws from the global torch RNG, so a model
constructed before the seed is set inherits whatever state the process is in.
The symptom is that a run's result depends on how many runs preceded it in the
same process -- which is invisible from any single run, and makes multi-seed
verification measure process history rather than seed sensitivity.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

pytest.importorskip("torch", reason="Two-tower requires the 'retrieval' extra")

import torch

from omnirank.models.two_tower import (
    MultimodalTwoTower,
    TwoTowerConfig,
    TwoTowerTrainer,
)


def build(seed: int) -> MultimodalTwoTower:
    """Construct a model the way ``fit_two_tower`` does: seed, then build."""
    config = TwoTowerConfig(
        embedding_dim=16,
        text_projection_dim=8,
        image_projection_dim=8,
        seed=seed,
        max_epochs=1,
        batch_size=4,
    )
    TwoTowerTrainer.set_seeds(config.seed)
    return MultimodalTwoTower(
        config, text_dim=6, image_dim=6, num_items=20, num_users=10, num_tags=3
    )


def flat_weights(model: MultimodalTwoTower) -> np.ndarray:
    """Every parameter, concatenated, for exact comparison."""
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])


class TestInitialisationDeterminism:
    def test_same_seed_gives_identical_weights(self) -> None:
        assert np.array_equal(flat_weights(build(42)), flat_weights(build(42)))

    def test_different_seeds_give_different_weights(self) -> None:
        """Otherwise the seed is not reaching initialisation at all."""
        assert not np.array_equal(flat_weights(build(42)), flat_weights(build(43)))

    def test_preceding_rng_consumption_does_not_change_the_result(self) -> None:
        """The defect this file exists for.

        A model built after other work in the same process must be identical to
        one built in a fresh process. Seeding *after* construction -- which is
        what the trainer alone would do -- fails this.
        """
        baseline = flat_weights(build(42))

        # Simulate an earlier run in the same process consuming randomness.
        torch.manual_seed(999)
        torch.randn(1000)
        np.random.default_rng(7).normal(size=1000)

        assert np.array_equal(flat_weights(build(42)), baseline)


class TestFitSeedsBeforeBuilding:
    """``fit_two_tower`` must seed before it constructs, not after.

    Tested by call order rather than by output, because the wrong order
    produces a *valid* model -- just one whose weights depend on process
    history. There is no assertion on a single fitted model that reveals it.
    """

    def test_set_seeds_precedes_model_construction(self, monkeypatch, tmp_path) -> None:
        import pandas as pd

        from omnirank.models import two_tower as tt
        from omnirank.retrieval import runner

        calls: list[str] = []

        class Stop(RuntimeError):
            """Ends the fit once the ordering has been observed."""

        def record_seeds(seed: int) -> None:
            calls.append("set_seeds")

        def record_build(*args: object, **kwargs: object) -> None:
            calls.append("build")
            raise Stop

        monkeypatch.setattr(tt.TwoTowerTrainer, "set_seeds", staticmethod(record_seeds))
        monkeypatch.setattr(tt, "MultimodalTwoTower", record_build)

        class Store:
            feature_version = "1"

            def require_compatible(self, **_: object) -> None:
                return None

            def dimension(self, _: str) -> int:
                return 4

        monkeypatch.setattr(runner, "load_item_tags", lambda *a, **k: (np.zeros(6, "int64"), 1))
        monkeypatch.setattr(
            "omnirank.features.multimodal_store.MultimodalFeatureStore",
            lambda *a, **k: Store(),
        )
        monkeypatch.setattr(tt, "TwoTowerTrainingDataset", lambda *a, **k: [0, 1])

        class Dataset:
            num_items = 6
            num_users = 3
            mapping_metadata: ClassVar[dict[str, str]] = {"item_mapping_checksum": "abc"}

        sequences = pd.DataFrame(
            {"internal_user_id": [0], "item_sequence": [[1, 2]], "target_item": [3]}
        )
        config = TwoTowerConfig(embedding_dim=8, seed=42)

        with pytest.raises(Stop):
            runner.fit_two_tower(
                Dataset(),  # type: ignore[arg-type]
                ("train",),
                config,
                processed_root=tmp_path,
                sequences=sequences,
            )

        assert calls == ["set_seeds", "build"]
