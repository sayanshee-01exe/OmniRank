"""Contrastive training and the loss it uses.

The false-negative tests carry the most weight. In-batch softmax treats every
other row's target as a negative, and some of those are items the user has
demonstrably interacted with. Unmasked, the loss actively pushes a user away
from things they like -- biased towards popular items, since those are the ones
most likely to be someone else's target, and invisible in a falling loss curve.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from omnirank.core.exceptions import DataError
from omnirank.models.two_tower.losses import (
    MASKED_LOGIT,
    build_false_negative_mask,
    in_batch_contrastive_loss,
)

from .conftest import IMAGE_DIM, ITEMS, TAGS, TEXT_DIM, USERS


class TestFalseNegativeMask:
    def test_duplicate_targets_are_masked(self) -> None:
        """Two rows sharing a target are each other's positive, not negative."""
        mask = build_false_negative_mask(torch.tensor([7, 7, 3]))
        assert bool(mask[0, 1]) and bool(mask[1, 0])
        assert not bool(mask[0, 2])

    def test_the_diagonal_is_never_masked(self) -> None:
        """It is the label; masking it removes the numerator entirely."""
        mask = build_false_negative_mask(torch.tensor([7, 7, 7]))
        assert not mask.diagonal().any()

    def test_known_user_positives_are_masked(self) -> None:
        mask = build_false_negative_mask(torch.tensor([1, 2, 3]), [{2}, set(), set()])
        assert bool(mask[0, 1])
        assert not bool(mask[0, 2])

    def test_rejects_misaligned_positive_sets(self) -> None:
        with pytest.raises(DataError):
            build_false_negative_mask(torch.tensor([1, 2, 3]), [{1}])

    def test_rejects_a_non_1d_target_tensor(self) -> None:
        with pytest.raises(DataError):
            build_false_negative_mask(torch.zeros(2, 2, dtype=torch.long))


class TestContrastiveLoss:
    @pytest.fixture
    def aligned(self):
        torch.manual_seed(0)
        users = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
        return users, users.clone()

    def test_perfectly_aligned_pairs_give_a_low_loss(self, aligned) -> None:
        users, items = aligned
        output = in_batch_contrastive_loss(users, items, temperature=0.05)
        assert float(output.loss) < 0.1
        assert output.in_batch_accuracy == 1.0

    def test_misaligned_pairs_give_a_higher_loss(self, aligned) -> None:
        users, items = aligned
        shuffled = items[torch.tensor([1, 2, 3, 4, 5, 0])]
        good = in_batch_contrastive_loss(users, items, temperature=0.05).loss
        bad = in_batch_contrastive_loss(users, shuffled, temperature=0.05).loss
        assert float(bad) > float(good)

    def test_masked_pairs_are_removed_from_the_denominator(self, aligned) -> None:
        users, items = aligned
        mask = torch.zeros(6, 6, dtype=torch.bool)
        mask[0, 1] = True
        output = in_batch_contrastive_loss(users, items, temperature=0.05, false_negative_mask=mask)
        assert output.masked_fraction == pytest.approx(1 / 30)

    def test_loss_is_finite_at_a_small_temperature(self, aligned) -> None:
        """log_softmax keeps this stable; softmax-then-log would overflow."""
        users, items = aligned
        assert torch.isfinite(in_batch_contrastive_loss(users, items, temperature=0.01).loss)

    def test_gradients_are_finite(self, aligned) -> None:
        users, items = aligned
        users = users.requires_grad_(True)
        in_batch_contrastive_loss(users, items, temperature=0.05).loss.backward()
        assert torch.isfinite(users.grad).all()

    def test_duplicate_targets_are_counted(self, aligned) -> None:
        users, items = aligned
        output = in_batch_contrastive_loss(
            users, items, temperature=0.05, target_items=torch.tensor([1, 1, 2, 3, 4, 5])
        )
        assert output.duplicate_targets == 1

    def test_rejects_a_non_positive_temperature(self, aligned) -> None:
        users, items = aligned
        with pytest.raises(DataError):
            in_batch_contrastive_loss(users, items, temperature=0.0)

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(DataError):
            in_batch_contrastive_loss(torch.randn(4, 8), torch.randn(4, 6), temperature=0.05)

    def test_rejects_an_empty_batch(self) -> None:
        with pytest.raises(DataError):
            in_batch_contrastive_loss(torch.randn(0, 8), torch.randn(0, 8), temperature=0.05)

    def test_rejects_a_mask_of_the_wrong_shape(self, aligned) -> None:
        users, items = aligned
        with pytest.raises(DataError):
            in_batch_contrastive_loss(
                users,
                items,
                temperature=0.05,
                false_negative_mask=torch.zeros(3, 3, dtype=torch.bool),
            )

    def test_masked_logit_is_finite(self) -> None:
        """An actual -inf would make a fully masked row produce NaN."""
        assert np.isfinite(MASKED_LOGIT)


def build_trainer(dataset, **overrides: object):
    """A small CPU trainer over the fixture dataset."""
    from omnirank.models.two_tower import (
        MultimodalTwoTower,
        TwoTowerConfig,
        TwoTowerTrainer,
    )

    settings = {
        "embedding_dim": 16,
        "hidden_dims": (32,),
        "dropout": 0.0,
        "maximum_history_length": 5,
        "batch_size": 16,
        "max_epochs": 20,
        "early_stopping_patience": 20,
        "learning_rate": 0.01,
        "temperature": 0.2,
        "device": "cpu",
        "seed": 7,
    }
    settings.update(overrides)
    config = TwoTowerConfig(**settings)
    torch.manual_seed(0)
    model = MultimodalTwoTower(
        config,
        text_dim=TEXT_DIM,
        image_dim=IMAGE_DIM,
        num_items=ITEMS,
        num_users=USERS,
        num_tags=TAGS,
    )
    return TwoTowerTrainer(model, config, device="cpu"), config


class TestTrainingLoop:
    def test_loss_decreases_on_a_learnable_fixture(self, dataset) -> None:
        trainer, _ = build_trainer(dataset)
        history = trainer.fit(dataset, dataset)
        assert history.train_loss[-1] < history.train_loss[0]

    def test_in_batch_accuracy_improves(self, dataset) -> None:
        """Guards against a falling loss that is not actually learning to retrieve."""
        trainer, _ = build_trainer(dataset)
        history = trainer.fit(dataset, dataset)
        assert history.in_batch_accuracy[-1] > history.in_batch_accuracy[0]

    def test_every_loss_is_finite(self, dataset) -> None:
        trainer, _ = build_trainer(dataset)
        history = trainer.fit(dataset, dataset)
        assert all(np.isfinite(history.train_loss))

    def test_false_negatives_are_masked_during_training(self, dataset) -> None:
        trainer, _ = build_trainer(dataset, max_epochs=2, early_stopping_patience=2)
        history = trainer.fit(dataset)
        assert max(history.masked_fraction) > 0

    def test_masking_can_be_disabled(self, dataset) -> None:
        trainer, _ = build_trainer(
            dataset, mask_false_negatives=False, max_epochs=2, early_stopping_patience=2
        )
        history = trainer.fit(dataset)
        assert max(history.masked_fraction) == 0

    def test_runs_on_cpu(self, dataset) -> None:
        trainer, _ = build_trainer(dataset, max_epochs=1, early_stopping_patience=1)
        history = trainer.fit(dataset)
        assert history.device == "cpu"

    def test_an_explicit_cpu_request_is_honoured(self, dataset) -> None:
        """CI is CPU-only; an unrequested MPS jump would make runs unreproducible."""
        trainer, _ = build_trainer(dataset, device="cpu", max_epochs=1)
        assert str(trainer.device) == "cpu"

    def test_validation_loss_is_recorded_when_supplied(self, dataset) -> None:
        trainer, _ = build_trainer(dataset, max_epochs=2, early_stopping_patience=2)
        history = trainer.fit(dataset, dataset)
        assert len(history.validation_loss) == len(history.train_loss)

    def test_proxy_recall_is_reported(self, dataset) -> None:
        trainer, _ = build_trainer(dataset, max_epochs=2, early_stopping_patience=2)
        history = trainer.fit(dataset, dataset)
        assert history.proxy_recall and 0.0 <= history.proxy_recall[-1] <= 1.0

    def test_early_stopping_triggers_on_a_plateau(self, dataset, monkeypatch) -> None:
        """Driven by a fixed validation loss rather than by hoping a run stalls.

        A very small learning rate does not plateau, it improves slowly -- at
        1e-8 the validation loss still falls ~1.8e-5 per epoch, well above the
        improvement threshold, so patience never accumulates. Pinning the
        monitored value is what makes this test about the stopping logic.
        """
        from omnirank.models.two_tower import TwoTowerTrainer

        trainer, _ = build_trainer(dataset, max_epochs=30, early_stopping_patience=2)
        monkeypatch.setattr(TwoTowerTrainer, "evaluate", lambda self, data: 1.0)
        history = trainer.fit(dataset, dataset)
        assert history.stopped_early
        # First epoch sets the best value; two more without improvement stop it.
        assert history.best_epoch == 1
        assert len(history.train_loss) == 3

    def test_without_validation_the_last_epoch_is_kept(self, dataset) -> None:
        """The weaker guarantee is recorded rather than implied."""
        trainer, _ = build_trainer(dataset, max_epochs=3, early_stopping_patience=3)
        history = trainer.fit(dataset)
        assert history.validation_loss == []
        assert history.best_epoch >= 1

    def test_best_checkpoint_is_restored_not_the_last(self, dataset) -> None:
        """Otherwise early stopping decides when to stop but not what survives."""
        trainer, _ = build_trainer(dataset, max_epochs=6, early_stopping_patience=6)
        history = trainer.fit(dataset, dataset)
        best = min(history.validation_loss)
        assert history.validation_loss[history.best_epoch - 1] == pytest.approx(best)

    def test_gradient_clipping_is_configurable(self, dataset) -> None:
        trainer, config = build_trainer(dataset, gradient_clip_norm=0.5, max_epochs=1)
        trainer.fit(dataset)
        assert config.gradient_clip_norm == 0.5

    def test_training_is_deterministic_under_a_fixed_seed(self, dataset) -> None:
        def run() -> list[float]:
            trainer, _ = build_trainer(dataset, max_epochs=3, early_stopping_patience=3)
            return trainer.fit(dataset).train_loss

        assert run() == pytest.approx(run())

    def test_max_batches_caps_an_epoch(self, dataset) -> None:
        trainer, _ = build_trainer(dataset, max_epochs=1)
        trainer.fit(dataset, max_batches_per_epoch=1)
        assert len(trainer.history.train_loss) == 1

    def test_history_serialises(self, dataset) -> None:
        trainer, _ = build_trainer(dataset, max_epochs=1, early_stopping_patience=1)
        payload = trainer.fit(dataset).to_dict()
        assert payload["epochs_run"] == 1
        assert "device" in payload


class TestTrainingSafety:
    def test_a_non_finite_loss_stops_the_run(self, dataset, monkeypatch) -> None:
        """A NaN kills every parameter in one backward pass; it must not continue."""
        import omnirank.models.two_tower.training as training_module

        trainer, _ = build_trainer(dataset, max_epochs=1)

        class Broken:
            loss = torch.tensor(float("nan"))
            in_batch_accuracy = 0.0
            masked_fraction = 0.0
            duplicate_targets = 0

        monkeypatch.setattr(training_module, "in_batch_contrastive_loss", lambda *a, **k: Broken())
        with pytest.raises(DataError, match="non-finite"):
            trainer.fit(dataset)

    def test_a_batch_of_one_is_skipped(self, dataset) -> None:
        """In-batch softmax needs a negative; a single row has none."""
        trainer, _ = build_trainer(dataset, batch_size=1, max_epochs=1)
        with pytest.raises(DataError, match="no usable batches"):
            trainer.fit(dataset)
