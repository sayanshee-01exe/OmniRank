"""SASRec - self-attentive sequential recommendation.

Collaborative models ask "what do users like this one enjoy?". SASRec asks a
different question: "given the last N things *this* user watched, in order, what
comes next?". On a short-video corpus that ordering carries real signal, and it
is signal LightGCN and BPR discard entirely.

The architecture is a causal transformer over the user's item sequence:

    item embedding + positional embedding
        -> N x [causal self-attention -> feed-forward], residual + layer-norm
        -> final non-padding hidden state
        -> dot product against item embeddings

**Causality is the correctness property.** Position *t* may attend to positions
<= *t* and never beyond. If it could see ahead, the model would learn to read the
answer, training loss would collapse, and offline metrics would look excellent
while the model was useless — the failure is invisible in the loss curve, so it
is asserted by a test that checks a future token cannot change an earlier
representation.

**Padding is ``num_items``**, one past the last valid internal id. Reusing 0
would collide with a real item, and Phase 2's global mappings are not altered to
make room. Padding is masked in attention, excluded from the loss, and can never
be recommended.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Self, cast

import numpy as np
import pandas as pd
import torch

from omnirank.core.exceptions import (
    ArtifactValidationError,
    DataError,
)
from omnirank.core.logging import get_logger
from omnirank.models.base import Candidate, CandidateGenerator
from omnirank.models.baselines.bpr import resolve_torch_device

logger = get_logger(__name__)

FORMAT_VERSION: Final = 1
#: Bumped when the sequence encoding changes shape or meaning.
SEQUENCE_SCHEMA_VERSION: Final = 1

_STATE_FILENAME: Final = "state.pt"
_CONFIG_FILENAME: Final = "config.json"

UNKNOWN_ITEM_SCORE: Final = 0.0


@dataclass(frozen=True, slots=True)
class SASRecConfig:
    """SASRec hyperparameters. Validated, including the head/dim divisibility."""

    maximum_sequence_length: int = 50
    embedding_dim: int = 128
    num_blocks: int = 2
    num_heads: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 256
    negatives_per_positive: int = 1
    max_epochs: int = 100
    early_stopping_patience: int = 10
    evaluation_user_batch_size: int = 256
    seed: int = 42

    def __post_init__(self) -> None:
        checks = (
            ("maximum_sequence_length", self.maximum_sequence_length > 0),
            ("embedding_dim", self.embedding_dim > 0),
            ("num_blocks", self.num_blocks > 0),
            ("num_heads", self.num_heads > 0),
            ("dropout", 0.0 <= self.dropout < 1.0),
            ("learning_rate", self.learning_rate > 0),
            ("batch_size", self.batch_size > 0),
            ("negatives_per_positive", self.negatives_per_positive > 0),
            ("max_epochs", self.max_epochs > 0),
            ("early_stopping_patience", self.early_stopping_patience > 0),
            ("evaluation_user_batch_size", self.evaluation_user_batch_size > 0),
            ("seed", self.seed >= 0),
        )
        problems = [name for name, ok in checks if not ok]
        if problems:
            raise DataError("Invalid SASRec configuration", invalid=problems)
        if self.embedding_dim % self.num_heads != 0:
            raise DataError(
                "embedding_dim must be divisible by num_heads: multi-head "
                "attention splits the embedding across heads.",
                embedding_dim=self.embedding_dim,
                num_heads=self.num_heads,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable payload."""
        return {
            "maximum_sequence_length": self.maximum_sequence_length,
            "embedding_dim": self.embedding_dim,
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "negatives_per_positive": self.negatives_per_positive,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "evaluation_user_batch_size": self.evaluation_user_batch_size,
            "seed": self.seed,
        }

    @property
    def label(self) -> str:
        """Compact identifier for experiment tables."""
        return (
            f"L{self.maximum_sequence_length}_d{self.embedding_dim}"
            f"_b{self.num_blocks}_h{self.num_heads}_do{self.dropout}"
        )


@dataclass(frozen=True, slots=True)
class SASRecFitData:
    """Phase 2 sequential examples plus the mappings they use."""

    sequences: pd.DataFrame
    num_users: int
    num_items: int
    internal_to_external_item: dict[int, str]
    external_to_internal_user: dict[str, int]
    mapping_checksum: str = ""
    dataset_identity: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = {"internal_user_id", "item_sequence", "target_item"}
        missing = required - set(self.sequences.columns)
        if missing:
            raise DataError("Sequential data missing columns", missing=sorted(missing))


class SASRecNetwork(torch.nn.Module):
    """The causal transformer encoder.

    Separated from the :class:`SASRec` generator so the architecture can be unit
    tested — particularly the causal mask — without any fitting machinery.
    """

    def __init__(self, config: SASRecConfig, num_items: int) -> None:
        super().__init__()
        self.config = config
        self.num_items = num_items
        # One extra row for padding, which is why padding_id == num_items.
        self.padding_id = num_items
        self.item_embedding = torch.nn.Embedding(
            num_items + 1, config.embedding_dim, padding_idx=self.padding_id
        )
        self.position_embedding = torch.nn.Embedding(
            config.maximum_sequence_length, config.embedding_dim
        )
        self.dropout = torch.nn.Dropout(config.dropout)
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.TransformerEncoderLayer(
                    d_model=config.embedding_dim,
                    nhead=config.num_heads,
                    dim_feedforward=config.embedding_dim * 4,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.final_norm = torch.nn.LayerNorm(config.embedding_dim)

    def causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask: position t may not attend beyond t.

        Boolean rather than additive ``-inf``: torch deprecates mixing a float
        attention mask with the boolean padding mask, and both are needed here.
        ``True`` marks a position that may not be attended to.
        """
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Encode padded sequences into per-position hidden states.

        Args:
            sequences: ``(batch, length)`` of item ids, right-aligned and
                left-padded with ``padding_id``.

        Returns:
            ``(batch, length, dim)`` hidden states.
        """
        batch, length = sequences.shape
        positions = torch.arange(length, device=sequences.device).unsqueeze(0).expand(batch, -1)
        hidden = self.item_embedding(sequences) * math.sqrt(self.config.embedding_dim)
        hidden = self.dropout(hidden + self.position_embedding(positions))

        # Padding is masked in attention so it contributes nothing; combined with
        # the causal mask, a position sees only real, earlier items.
        padding_mask = sequences == self.padding_id
        attention_mask = self.causal_mask(length, sequences.device)
        for block in self.blocks:
            hidden = cast(
                "torch.Tensor",
                block(hidden, src_mask=attention_mask, src_key_padding_mask=padding_mask),
            )
        return cast("torch.Tensor", self.final_norm(hidden))

    def score_items(self, hidden: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Dot-product scores between hidden states and item embeddings."""
        embedded = cast("torch.Tensor", self.item_embedding(item_ids))
        return (hidden.unsqueeze(-2) * embedded).sum(dim=-1)


def encode_sequences(
    sequences: list[list[int]], *, maximum_length: int, padding_id: int
) -> np.ndarray:
    """Left-pad and right-align sequences to a fixed width.

    Right-alignment puts the most recent item in the final column, so the last
    position is always the one inference reads. Truncation keeps the newest
    ``maximum_length`` items — the ones a self-attentive model actually attends
    to — and drops the oldest.
    """
    encoded = np.full((len(sequences), maximum_length), padding_id, dtype="int64")
    for row, sequence in enumerate(sequences):
        recent = list(sequence)[-maximum_length:]
        if recent:
            encoded[row, maximum_length - len(recent) :] = recent
    return encoded


class SASRec(CandidateGenerator):
    """Self-attentive sequential retrieval."""

    name = "sasrec"

    def __init__(self, config: SASRecConfig | None = None, *, device: str = "auto") -> None:
        super().__init__()
        self.config = config or SASRecConfig()
        self.device_preference = device
        self._device = torch.device("cpu")
        self._network: SASRecNetwork | None = None
        self._internal_to_external: dict[int, str] = {}
        self._external_to_internal: dict[str, int] = {}
        self._external_to_internal_user: dict[str, int] = {}
        self._user_histories: dict[int, list[int]] = {}
        self._seen_by_user: dict[int, set[int]] = {}
        self._fit_item_ids: np.ndarray = np.empty(0, dtype="int64")
        self._num_items = 0
        self._mapping_checksum = ""
        self._dataset_identity: dict[str, Any] = {}
        self._loss_history: list[float] = []
        self._best_epoch = 0

    @property
    def padding_id(self) -> int:
        """The padding token: one past the last valid internal item id."""
        return self._num_items

    # -- fitting ------------------------------------------------------------ #
    def fit(self, data: Any) -> None:
        """Train with a sampled binary cross-entropy next-item objective.

        Sampled rather than a full softmax: a 69,347-way softmax at every one of
        up to 50 positions per sequence, over 776k sequences, is not practical
        and profiling was not needed to establish that. One positive and
        ``negatives_per_positive`` negatives per valid position is the standard
        SASRec formulation.
        """
        if not isinstance(data, SASRecFitData):
            raise DataError(
                "SASRec.fit expects a SASRecFitData bundle", received=type(data).__name__
            )
        frame = data.sequences
        if frame.empty:
            raise DataError("Cannot fit SASRec on an empty sequence set")

        self._device = resolve_torch_device(self.device_preference)
        self._num_items = data.num_items
        self._set_seeds(self.config.seed)

        sequences = [list(row) for row in frame["item_sequence"]]
        targets = frame["target_item"].to_numpy(dtype="int64")
        users = frame["internal_user_id"].to_numpy(dtype="int64")

        # The model predicts the next item at every position, so the label
        # sequence is the input shifted one step, with the target appended.
        inputs = encode_sequences(
            sequences,
            maximum_length=self.config.maximum_sequence_length,
            padding_id=self.padding_id,
        )
        labels = np.full_like(inputs, self.padding_id)
        labels[:, :-1] = inputs[:, 1:]
        labels[:, -1] = targets

        network = SASRecNetwork(self.config, data.num_items).to(self._device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.config.learning_rate)

        rows = len(sequences)
        shuffle = np.random.default_rng(self.config.seed)
        sampler_rng = np.random.default_rng(self.config.seed + 1)

        logger.info(
            "sasrec.training_started",
            device=str(self._device),
            sequences=rows,
            items=data.num_items,
            padding_id=self.padding_id,
            **self.config.to_dict(),
        )

        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        patience = 0
        self._loss_history = []

        for epoch in range(self.config.max_epochs):
            network.train()
            order = shuffle.permutation(rows)
            epoch_loss, batches = 0.0, 0
            for start in range(0, rows, self.config.batch_size):
                index = order[start : start + self.config.batch_size]
                batch_inputs = torch.from_numpy(inputs[index]).to(self._device)
                batch_labels = torch.from_numpy(labels[index]).to(self._device)

                hidden = network(batch_inputs)
                # Only positions with a real next item contribute.
                valid = batch_labels != self.padding_id
                if not valid.any():
                    continue

                negatives = torch.from_numpy(
                    sampler_rng.integers(
                        0,
                        data.num_items,
                        size=(*batch_labels.shape, self.config.negatives_per_positive),
                    )
                ).to(self._device)

                positive_scores = network.score_items(hidden, batch_labels.unsqueeze(-1)).squeeze(
                    -1
                )
                negative_scores = network.score_items(hidden, negatives)

                positive_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    positive_scores, torch.ones_like(positive_scores), reduction="none"
                )
                negative_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    negative_scores, torch.zeros_like(negative_scores), reduction="none"
                ).mean(dim=-1)
                loss = ((positive_loss + negative_loss) * valid).sum() / valid.sum()

                if not torch.isfinite(loss):
                    raise DataError(
                        "SASRec loss became non-finite; the learning rate is probably too high.",
                        epoch=epoch,
                        learning_rate=self.config.learning_rate,
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                for name, parameter in network.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise DataError(
                            "Non-finite gradient during SASRec training",
                            epoch=epoch,
                            parameter=name,
                        )
                optimizer.step()
                epoch_loss += float(loss.detach())
                batches += 1

            mean_loss = epoch_loss / max(batches, 1)
            self._loss_history.append(mean_loss)
            logger.info("sasrec.epoch", epoch=epoch + 1, mean_loss=round(mean_loss, 6))

            if mean_loss < best_loss - 1e-6:
                best_loss = mean_loss
                best_state = {
                    key: value.detach().clone() for key, value in network.state_dict().items()
                }
                self._best_epoch = epoch + 1
                patience = 0
            else:
                patience += 1
                if patience >= self.config.early_stopping_patience:
                    logger.info(
                        "sasrec.early_stopped", epoch=epoch + 1, best_epoch=self._best_epoch
                    )
                    break

        if best_state is not None:
            network.load_state_dict(best_state)
        network.eval()
        self._network = network

        # The inference history is the full sequence plus its target: at serving
        # time everything the user has done is available.
        histories: dict[int, list[int]] = {}
        seen: dict[int, set[int]] = {}
        for user, sequence, target in zip(users, sequences, targets, strict=True):
            key = int(user)
            history = [*sequence, int(target)]
            if key not in histories or len(history) > len(histories[key]):
                histories[key] = history
            seen.setdefault(key, set()).update(history)
        self._user_histories = histories
        self._seen_by_user = seen

        observed = {item for sequence in sequences for item in sequence}
        observed.update(int(target) for target in targets)
        self._fit_item_ids = np.array(sorted(observed), dtype="int64")

        self._internal_to_external = data.internal_to_external_item
        self._external_to_internal = {v: k for k, v in data.internal_to_external_item.items()}
        self._external_to_internal_user = data.external_to_internal_user
        self._mapping_checksum = data.mapping_checksum
        self._dataset_identity = data.dataset_identity
        self._fitted = True
        logger.info(
            "sasrec.training_completed",
            epochs_run=len(self._loss_history),
            best_epoch=self._best_epoch,
            first_loss=round(self._loss_history[0], 6),
            final_loss=round(self._loss_history[-1], 6),
        )

    @staticmethod
    def _set_seeds(seed: int) -> None:
        """Seed every RNG that can affect training."""
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    # -- inference ---------------------------------------------------------- #
    def _query_vectors(self, internal_users: list[int]) -> torch.Tensor:
        """Encode each user's recent history into a query vector."""
        assert self._network is not None  # noqa: S101
        histories = [self._user_histories.get(user, []) for user in internal_users]
        encoded = encode_sequences(
            histories,
            maximum_length=self.config.maximum_sequence_length,
            padding_id=self.padding_id,
        )
        with torch.no_grad():
            hidden = cast("torch.Tensor", self._network(torch.from_numpy(encoded).to(self._device)))
        # Right-alignment means the last column is always the most recent item.
        return hidden[:, -1, :]

    def recommend(
        self, user_id: str, k: int, context: dict[str, Any] | None = None
    ) -> list[Candidate]:
        """Top-``k`` items for one user.

        Unknown users and users with no history return an empty list: a
        sequential model has nothing to encode, and inventing a query vector
        would produce confident noise.
        """
        self.ensure_fitted()
        internal = self._external_to_internal_user.get(user_id)
        if internal is None or not self._user_histories.get(internal):
            return []
        filter_seen = True if context is None else bool(context.get("filter_seen", True))
        items, scores = self._top_k([internal], k, filter_seen=filter_seen)
        return [
            Candidate(
                item_id=self._internal_to_external[int(item)],
                score=float(score),
                sources=(self.name,),
                source_scores={self.name: float(score)},
            )
            for item, score in zip(items[0], scores[0], strict=True)
            if item >= 0
        ]

    def recommend_batch(
        self, user_ids: list[str], k: int, *, filter_seen: bool = True
    ) -> dict[str, list[str]]:
        """Top-``k`` external item ids for many users, memory-bounded."""
        self.ensure_fitted()
        results: dict[str, list[str]] = {}
        known: list[tuple[str, int]] = []
        for user_id in user_ids:
            internal = self._external_to_internal_user.get(user_id)
            if internal is None or not self._user_histories.get(internal):
                results[user_id] = []
            else:
                known.append((user_id, internal))

        batch = self.config.evaluation_user_batch_size
        for start in range(0, len(known), batch):
            chunk = known[start : start + batch]
            items, _ = self._top_k([internal for _, internal in chunk], k, filter_seen=filter_seen)
            for (user_id, _), row in zip(chunk, items, strict=True):
                results[user_id] = [
                    self._internal_to_external[int(item)] for item in row if item >= 0
                ]
        return results

    def _top_k(
        self, internal_users: list[int], k: int, *, filter_seen: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score a user batch against the catalogue and reduce to top-k."""
        if k < 1:
            raise DataError("k must be >= 1", k=k)
        assert self._network is not None  # noqa: S101
        queries = self._query_vectors(internal_users)
        # Drop the padding row: it must never be scored, let alone recommended.
        item_matrix = self._network.item_embedding.weight[: self._num_items]
        scores = queries @ item_matrix.T

        outside = torch.ones(scores.shape[1], dtype=torch.bool, device=self._device)
        outside[torch.from_numpy(self._fit_item_ids).to(self._device)] = False
        scores[:, outside] = float("-inf")

        if filter_seen:
            for row, user in enumerate(internal_users):
                seen = self._seen_by_user.get(user)
                if seen:
                    scores[row, torch.tensor(sorted(seen), device=self._device)] = float("-inf")

        take = min(k, scores.shape[1])
        top_scores, top_items = torch.topk(scores, take, dim=1)
        items = top_items.cpu().numpy().astype("int64")
        values = top_scores.detach().cpu().numpy().astype("float64")
        items = np.where(np.isneginf(values), -1, items)
        if take < k:
            pad = k - take
            items = np.concatenate(
                [items, np.full((len(internal_users), pad), -1, dtype="int64")], axis=1
            )
            values = np.concatenate(
                [values, np.full((len(internal_users), pad), float("-inf"))], axis=1
            )
        return items, values

    def score(self, user_id: str, item_ids: list[str]) -> list[float]:
        """Score specific items. Unknown users and items score 0.0."""
        self.ensure_fitted()
        assert self._network is not None  # noqa: S101
        internal = self._external_to_internal_user.get(user_id)
        if internal is None or not self._user_histories.get(internal):
            return [UNKNOWN_ITEM_SCORE] * len(item_ids)
        query = self._query_vectors([internal])[0]
        with torch.no_grad():
            weights = self._network.item_embedding.weight
            scores: list[float] = []
            for item in item_ids:
                internal_item = self._external_to_internal.get(item)
                scores.append(
                    UNKNOWN_ITEM_SCORE
                    if internal_item is None
                    else float(query @ weights[internal_item])
                )
        return scores

    # -- introspection ------------------------------------------------------ #
    @property
    def fit_item_catalogue(self) -> set[int]:
        """Internal ids of every item this model can recommend."""
        return set(self._fit_item_ids.tolist())

    @property
    def loss_history(self) -> list[float]:
        """Mean training loss per epoch."""
        return list(self._loss_history)

    @property
    def device(self) -> str:
        """Device the network lives on."""
        return str(self._device)

    def item_embeddings(self) -> np.ndarray:
        """Item embeddings, excluding padding, for building a vector index."""
        self.ensure_fitted()
        assert self._network is not None  # noqa: S101
        weights = self._network.item_embedding.weight[: self._num_items]
        return weights.detach().cpu().numpy().astype("float32")

    def query_embeddings(self, user_ids: list[str]) -> np.ndarray:
        """Query vectors for the given users, for searching a vector index."""
        self.ensure_fitted()
        internal = [
            self._external_to_internal_user[user]
            for user in user_ids
            if user in self._external_to_internal_user
        ]
        if not internal:
            return np.empty((0, self.config.embedding_dim), dtype="float32")
        return self._query_vectors(internal).detach().cpu().numpy().astype("float32")

    def history_length(self, user_id: str) -> int:
        """How many interactions this user's inference history holds."""
        internal = self._external_to_internal_user.get(user_id)
        return len(self._user_histories.get(internal, [])) if internal is not None else 0

    def metadata(self) -> dict[str, Any]:
        """Configuration and fit provenance, for the artifact manifest."""
        return {
            "model": self.name,
            "format_version": FORMAT_VERSION,
            "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
            "config": self.config.to_dict(),
            "device": str(self._device),
            "loss_history": self._loss_history,
            "best_epoch": self._best_epoch,
            "epochs_run": len(self._loss_history),
            "num_items": self._num_items,
            "padding_id": self.padding_id,
            "catalogue_size": int(self._fit_item_ids.size),
            "mapping_checksum": self._mapping_checksum,
            "dataset_identity": self._dataset_identity,
            "objective": "sampled binary cross-entropy over next-item positions",
            "attention": "causal (upper-triangular mask), padding masked",
        }

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        """Persist to a directory, device-neutrally."""
        self.ensure_fitted()
        assert self._network is not None  # noqa: S101
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        users = sorted(self._user_histories)
        torch.save(
            {
                "network": {key: value.cpu() for key, value in self._network.state_dict().items()},
                "fit_item_ids": torch.from_numpy(self._fit_item_ids),
                "history_users": torch.tensor(users, dtype=torch.int64),
                "history_lengths": torch.tensor(
                    [len(self._user_histories[u]) for u in users], dtype=torch.int64
                ),
                "history_flat": torch.tensor(
                    [item for u in users for item in self._user_histories[u]], dtype=torch.int64
                ),
            },
            target / _STATE_FILENAME,
        )
        payload = self.metadata()
        payload["item_mapping"] = {
            str(key): value for key, value in sorted(self._internal_to_external.items())
        }
        payload["user_mapping"] = dict(sorted(self._external_to_internal_user.items()))
        (target / _CONFIG_FILENAME).write_text(json.dumps(payload, indent=2, sort_keys=True))
        logger.info("sasrec.saved", path=str(target))

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> Self:
        """Restore a saved model."""
        source = Path(path)
        state_path, config_path = source / _STATE_FILENAME, source / _CONFIG_FILENAME
        for candidate in (state_path, config_path):
            if not candidate.is_file():
                raise ArtifactValidationError(
                    "SASRec artifact is incomplete", missing=str(candidate)
                )
        try:
            metadata = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError(
                "SASRec config is not valid JSON", path=str(config_path)
            ) from exc
        if metadata.get("model") != cls.name:
            raise ArtifactValidationError(
                "Artifact was written by a different model type",
                expected=cls.name,
                found=metadata.get("model"),
            )
        if metadata.get("format_version") != FORMAT_VERSION:
            raise ArtifactValidationError(
                "Unsupported SASRec artifact format version",
                expected=FORMAT_VERSION,
                found=metadata.get("format_version"),
            )
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ArtifactValidationError(
                "SASRec state file could not be read; it may be corrupted",
                path=str(state_path),
                reason=str(exc)[:200],
            ) from exc

        model = cls(SASRecConfig(**metadata["config"]), device=device)
        model._device = resolve_torch_device(device)
        model._num_items = int(metadata["num_items"])
        network = SASRecNetwork(model.config, model._num_items)
        network.load_state_dict(state["network"])
        network.eval()
        model._network = network.to(model._device)
        model._fit_item_ids = state["fit_item_ids"].numpy().astype("int64")

        flat = state["history_flat"].tolist()
        offset = 0
        histories: dict[int, list[int]] = {}
        for user, length in zip(
            state["history_users"].tolist(), state["history_lengths"].tolist(), strict=True
        ):
            histories[int(user)] = flat[offset : offset + int(length)]
            offset += int(length)
        model._user_histories = histories
        model._seen_by_user = {user: set(items) for user, items in histories.items()}

        model._internal_to_external = {
            int(key): str(value) for key, value in metadata["item_mapping"].items()
        }
        model._external_to_internal = {v: k for k, v in model._internal_to_external.items()}
        model._external_to_internal_user = {
            str(key): int(value) for key, value in metadata["user_mapping"].items()
        }
        model._mapping_checksum = metadata.get("mapping_checksum", "")
        model._dataset_identity = metadata.get("dataset_identity", {})
        model._loss_history = list(metadata.get("loss_history", []))
        model._best_epoch = int(metadata.get("best_epoch", 0))
        model._fitted = True
        logger.info("sasrec.loaded", path=str(source), device=str(model._device))
        return model

    def require_mapping(self, mapping_checksum: str) -> None:
        """Assert this model was fitted against the given item mapping."""
        if self._mapping_checksum and mapping_checksum != self._mapping_checksum:
            raise ArtifactValidationError(
                "Item mapping checksum does not match the one this model was fitted against.",
                expected=self._mapping_checksum,
                found=mapping_checksum,
            )


__all__ = [
    "FORMAT_VERSION",
    "SEQUENCE_SCHEMA_VERSION",
    "UNKNOWN_ITEM_SCORE",
    "SASRec",
    "SASRecConfig",
    "SASRecFitData",
    "SASRecNetwork",
    "encode_sequences",
]
