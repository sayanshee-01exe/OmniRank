"""Contrastive training loop for the two-tower model.

Deliberately ordinary in shape -- shuffle, forward, loss, clip, step -- because
the interesting decisions are elsewhere. What this module is careful about is
the three ways a contrastive run fails without saying so:

**A non-finite loss trains a dead model.** One NaN propagates through every
parameter in a single backward pass, and from then on the loss prints as ``nan``
while the run continues to completion. Checked per batch and raised, not warned.

**A device that silently degrades.** MPS is used when available and falls back
to CPU explicitly, with the resolved device logged and recorded in the artifact.
An unlogged fallback shows up only as an unexplained tenfold slowdown.

**Early stopping on the wrong quantity.** Stopping on *training* loss selects
the most overfit checkpoint. Validation loss is the stopping signal; a subset
retrieval proxy is reported alongside it because a falling contrastive loss and
a rising retrieval quality are not the same claim.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from omnirank.core.exceptions import DataError
from omnirank.core.logging import get_logger
from omnirank.models.baselines.bpr import resolve_torch_device
from omnirank.models.two_tower.config import TwoTowerConfig
from omnirank.models.two_tower.dataset import TwoTowerBatch, TwoTowerTrainingDataset
from omnirank.models.two_tower.losses import (
    build_false_negative_mask,
    in_batch_contrastive_loss,
)
from omnirank.models.two_tower.model import MultimodalTwoTower

logger = get_logger(__name__)

#: Validation rows used for the retrieval proxy. Small on purpose: this is a
#: per-epoch sanity signal, not an evaluation. Full-catalogue retrieval is the
#: next milestone's job and costs orders of magnitude more.
PROXY_ROWS = 512
PROXY_CUTOFF = 20


@dataclass
class TrainingHistory:
    """Per-epoch record of what happened."""

    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    proxy_recall: list[float] = field(default_factory=list)
    in_batch_accuracy: list[float] = field(default_factory=list)
    masked_fraction: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False
    device: str = "cpu"
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Report-ready payload."""
        return {
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "proxy_recall@20": self.proxy_recall,
            "in_batch_accuracy": self.in_batch_accuracy,
            "masked_fraction": self.masked_fraction,
            "best_epoch": self.best_epoch,
            "epochs_run": len(self.train_loss),
            "stopped_early": self.stopped_early,
            "device": self.device,
            "seconds": round(self.seconds, 2),
        }


def _to_tensors(batch: TwoTowerBatch, device: torch.device) -> dict[str, torch.Tensor]:
    """Move one collated batch onto the compute device.

    The dataset stays device-agnostic so it can be tested and shared; every
    transfer happens here, in one place, which is also what keeps a partially
    moved model from being possible.
    """
    return {
        "user_ids": torch.from_numpy(batch.user_ids).to(device),
        "history_text_features": torch.from_numpy(batch.history_text_features).to(device),
        "history_image_features": torch.from_numpy(batch.history_image_features).to(device),
        "history_text_available": torch.from_numpy(batch.history_text_available).to(device),
        "history_image_available": torch.from_numpy(batch.history_image_available).to(device),
        "history_tag_ids": torch.from_numpy(batch.history_tag_ids).to(device),
        "history_padding_mask": torch.from_numpy(batch.history_padding_mask).to(device),
        "history_lengths": torch.from_numpy(batch.history_lengths).to(device),
        "positive_item_ids": torch.from_numpy(batch.positive_item_ids).to(device),
        "positive_text_features": torch.from_numpy(batch.positive_text_features).to(device),
        "positive_image_features": torch.from_numpy(batch.positive_image_features).to(device),
        "positive_text_available": torch.from_numpy(batch.positive_text_available).to(device),
        "positive_image_available": torch.from_numpy(batch.positive_image_available).to(device),
        "positive_tag_ids": torch.from_numpy(batch.positive_tag_ids).to(device),
        "positive_warm_mask": torch.from_numpy(batch.positive_warm_mask).to(device),
    }


class TwoTowerTrainer:
    """Trains a :class:`MultimodalTwoTower` with in-batch contrastive loss."""

    def __init__(
        self,
        model: MultimodalTwoTower,
        config: TwoTowerConfig,
        *,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.device = resolve_torch_device(device or config.device)
        # Moved whole, once. Moving submodules piecemeal is how half a model
        # ends up on one device and produces a cryptic dtype error later.
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.history = TrainingHistory(device=str(self.device))
        logger.info(
            "two_tower.trainer_ready",
            device=str(self.device),
            requested=device or config.device,
            parameters=sum(p.numel() for p in self.model.parameters()),
        )

    @staticmethod
    def set_seeds(seed: int) -> None:
        """Seed every RNG that can affect a run."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    def _step(
        self, dataset: TwoTowerTrainingDataset, rows: np.ndarray, *, train: bool
    ) -> tuple[torch.Tensor, float, float]:
        """One forward pass, returning loss and diagnostics."""
        batch = dataset.collate(rows)
        batch.validate(
            text_dim=self.model.text_dim,
            image_dim=self.model.image_dim,
            max_history=dataset.maximum_history_length,
        )
        tensors = _to_tensors(batch, self.device)
        users, items = self.model(tensors)

        mask = None
        if self.config.mask_false_negatives:
            mask = build_false_negative_mask(
                tensors["positive_item_ids"],
                dataset.positives_by_row(rows) if train else None,
            )
        output = in_batch_contrastive_loss(
            users,
            items,
            temperature=self.config.temperature,
            false_negative_mask=mask,
            target_items=tensors["positive_item_ids"],
        )
        if not torch.isfinite(output.loss):
            raise DataError(
                "Two-tower loss became non-finite. One NaN propagates to every "
                "parameter in a single backward pass, so this stops the run "
                "rather than training a dead model to completion.",
                temperature=self.config.temperature,
                learning_rate=self.config.learning_rate,
            )
        return output.loss, output.in_batch_accuracy, output.masked_fraction

    def _proxy_recall(self, dataset: TwoTowerTrainingDataset, rows: np.ndarray) -> float:
        """Recall@20 of the positive within a fixed validation subset.

        Ranks each user's true item against the other items in the same subset,
        not against the catalogue. That makes it far easier than real retrieval
        and not comparable to a reported Recall@20 -- it is a direction signal
        for early stopping, and it is named `proxy` everywhere for that reason.
        """
        if rows.size < 2:
            return 0.0
        self.model.eval()
        with torch.no_grad():
            batch = dataset.collate(rows)
            tensors = _to_tensors(batch, self.device)
            users, items = self.model(tensors)
            scores = self.model.similarity(users, items)
            cutoff = min(PROXY_CUTOFF, scores.shape[1])
            top = scores.topk(cutoff, dim=1).indices
            labels = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
            return float((top == labels).any(dim=1).float().mean())

    def fit(
        self,
        train_dataset: TwoTowerTrainingDataset,
        validation_dataset: TwoTowerTrainingDataset | None = None,
        *,
        max_batches_per_epoch: int | None = None,
    ) -> TrainingHistory:
        """Train with early stopping on validation loss.

        Args:
            train_dataset: Examples to fit on.
            validation_dataset: Held-out examples for early stopping. Without
                one, training loss is used and the checkpoint is the last epoch
                -- recorded in the history so the weaker guarantee is visible.
            max_batches_per_epoch: Development cap, for smoke runs.
        """
        import time

        self.set_seeds(self.config.seed)
        shuffle = np.random.default_rng(self.config.seed)
        proxy_rows = (
            np.arange(min(PROXY_ROWS, len(validation_dataset)))
            if validation_dataset is not None
            else np.empty(0, dtype="int64")
        )

        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        patience = 0
        started = time.perf_counter()

        for epoch in range(self.config.max_epochs):
            self.model.train()
            losses: list[float] = []
            accuracies: list[float] = []
            masked: list[float] = []

            for index, rows in enumerate(
                train_dataset.batches(self.config.batch_size, rng=shuffle)
            ):
                if max_batches_per_epoch is not None and index >= max_batches_per_epoch:
                    break
                if rows.size < 2:
                    # In-batch softmax needs at least one negative; a batch of
                    # one has only its own positive and no signal.
                    continue
                loss, accuracy, masked_fraction = self._step(train_dataset, rows, train=True)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                for name, parameter in self.model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise DataError(
                            "Non-finite gradient during two-tower training",
                            epoch=epoch,
                            parameter=name,
                        )
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
                self.optimizer.step()
                losses.append(float(loss.detach()))
                accuracies.append(accuracy)
                masked.append(masked_fraction)

            if not losses:
                raise DataError(
                    "An epoch produced no usable batches. Every batch had fewer "
                    "than two examples, so the contrastive objective had no "
                    "negatives to work with.",
                    batch_size=self.config.batch_size,
                    examples=len(train_dataset),
                )

            train_loss = float(np.mean(losses))
            self.history.train_loss.append(train_loss)
            self.history.in_batch_accuracy.append(float(np.mean(accuracies)))
            self.history.masked_fraction.append(float(np.mean(masked)))

            monitored = train_loss
            if validation_dataset is not None:
                validation_loss = self.evaluate(validation_dataset)
                self.history.validation_loss.append(validation_loss)
                self.history.proxy_recall.append(self._proxy_recall(validation_dataset, proxy_rows))
                monitored = validation_loss

            logger.info(
                "two_tower.epoch",
                epoch=epoch + 1,
                train_loss=round(train_loss, 6),
                validation_loss=(
                    round(self.history.validation_loss[-1], 6)
                    if self.history.validation_loss
                    else None
                ),
                proxy_recall=(
                    round(self.history.proxy_recall[-1], 6) if self.history.proxy_recall else None
                ),
                in_batch_accuracy=round(self.history.in_batch_accuracy[-1], 6),
            )

            if monitored < best_loss - 1e-6:
                best_loss = monitored
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                self.history.best_epoch = epoch + 1
                patience = 0
            else:
                patience += 1
                if patience >= self.config.early_stopping_patience:
                    self.history.stopped_early = True
                    logger.info(
                        "two_tower.early_stopped",
                        epoch=epoch + 1,
                        best_epoch=self.history.best_epoch,
                    )
                    break

        if best_state is not None:
            # Restore the best checkpoint, not the last. Without this, early
            # stopping only decides when to stop, not which weights survive.
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        self.history.seconds = time.perf_counter() - started
        logger.info("two_tower.training_completed", **self.history.to_dict())
        return self.history

    def evaluate(self, dataset: TwoTowerTrainingDataset) -> float:
        """Mean contrastive loss over a dataset, without gradients."""
        self.model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for rows in dataset.batches(self.config.batch_size):
                if rows.size < 2:
                    continue
                loss, _, _ = self._step(dataset, rows, train=False)
                losses.append(float(loss))
        return float(np.mean(losses)) if losses else float("inf")


__all__ = ["PROXY_CUTOFF", "PROXY_ROWS", "TrainingHistory", "TwoTowerTrainer"]
