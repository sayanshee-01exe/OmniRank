"""BPR matrix factorization - the first personalised baseline.

Bayesian Personalized Ranking optimises a *ranking* objective rather than a
reconstruction one, which is what implicit feedback calls for: PixelRec records
that a user engaged with an item and nothing about how much, so there is no
rating to reconstruct. BPR only asks that an observed item outrank an unobserved
one, which is exactly the claim the data supports.

For a user ``u``, an observed item ``i``, and a sampled unobserved ``j``::

    L = -log sigmoid(y_ui - y_uj) + lambda * ||Theta||^2      with y_ui = p_u . q_i

**Device policy.** CPU and Apple MPS; never CUDA implicitly. MPS is used when
available and requested, with a logged fallback to CPU on failure - and no claim
is made that MPS and CPU produce bitwise-identical results, because they do not.

Torch is imported at module import, so this module is only importable with the
``baseline`` extra installed. Nothing else in the package imports it, which is
why popularity and the evaluator still work without torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Self

import numpy as np
import pandas as pd
import torch

from omnirank.core.exceptions import (
    ArtifactValidationError,
    DataError,
)
from omnirank.core.logging import get_logger
from omnirank.models.base import Candidate, CandidateGenerator
from omnirank.models.baselines.negative_sampling import (
    UniformNegativeSampler,
    build_positives_by_user,
)

logger = get_logger(__name__)

FORMAT_VERSION: Final = 1
_STATE_FILENAME: Final = "state.pt"
_CONFIG_FILENAME: Final = "config.json"

#: Score returned for an item the model never saw. Zero rather than -inf so the
#: value composes with a ranking stage that sums generator scores.
UNKNOWN_ITEM_SCORE: Final = 0.0


@dataclass(frozen=True, slots=True)
class BPRConfig:
    """BPR hyperparameters. Every value is validated, not merely stored."""

    embedding_dim: int = 64
    learning_rate: float = 0.005
    regularization: float = 1e-4
    batch_size: int = 4096
    epochs: int = 20
    negatives_per_positive: int = 1
    evaluation_user_batch_size: int = 512
    seed: int = 42

    def __post_init__(self) -> None:
        checks = (
            ("embedding_dim", self.embedding_dim > 0, "must be positive"),
            ("learning_rate", self.learning_rate > 0, "must be positive"),
            ("regularization", self.regularization >= 0, "must be non-negative"),
            ("batch_size", self.batch_size > 0, "must be positive"),
            ("epochs", self.epochs > 0, "must be positive"),
            ("negatives_per_positive", self.negatives_per_positive > 0, "must be positive"),
            ("evaluation_user_batch_size", self.evaluation_user_batch_size > 0, "must be positive"),
            ("seed", self.seed >= 0, "must be non-negative"),
        )
        problems = [f"{name} {message}" for name, ok, message in checks if not ok]
        if problems:
            raise DataError("Invalid BPR configuration", problems=problems)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable payload."""
        return {
            "embedding_dim": self.embedding_dim,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "negatives_per_positive": self.negatives_per_positive,
            "evaluation_user_batch_size": self.evaluation_user_batch_size,
            "seed": self.seed,
        }

    @property
    def label(self) -> str:
        """Compact identifier for experiment tables."""
        return (
            f"d{self.embedding_dim}_lr{self.learning_rate}_reg{self.regularization}"
            f"_neg{self.negatives_per_positive}_e{self.epochs}"
        )


@dataclass(frozen=True, slots=True)
class BPRFitData:
    """Fit interactions plus the mappings they are expressed in."""

    interactions: pd.DataFrame
    num_users: int
    num_items: int
    internal_to_external_item: dict[int, str]
    external_to_internal_user: dict[str, int]
    mapping_checksum: str = ""
    dataset_identity: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = {"internal_user_id", "internal_item_id"} - set(self.interactions.columns)
        if missing:
            raise DataError("Fit interactions missing columns", missing=sorted(missing))


def resolve_torch_device(preferred: str = "auto", *, allow_cuda: bool = False) -> torch.device:
    """Pick a torch device, never assuming CUDA.

    ``auto`` selects MPS when it is available and CPU otherwise. An explicit MPS
    request that cannot be satisfied logs and falls back to CPU rather than
    failing: a slower run is better than no run, and the fallback is recorded in
    the artifact so a result is never silently attributed to the wrong device.
    """
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if allow_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("bpr.cuda_unavailable", falling_back_to="cpu", allow_cuda=allow_cuda)
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "mps":
        logger.warning(
            "bpr.mps_unavailable",
            falling_back_to="cpu",
            reason="torch reports MPS is not available on this host",
        )
    return torch.device("cpu")


class BPRMatrixFactorization(CandidateGenerator):
    """Implicit-feedback matrix factorization trained with the BPR objective."""

    name = "matrix_factorization"

    def __init__(self, config: BPRConfig | None = None, *, device: str = "auto") -> None:
        super().__init__()
        self.config = config or BPRConfig()
        self.device_preference = device
        self._device = torch.device("cpu")
        self._user_factors: torch.Tensor | None = None
        self._item_factors: torch.Tensor | None = None
        self._internal_to_external: dict[int, str] = {}
        self._external_to_internal: dict[str, int] = {}
        self._external_to_internal_user: dict[str, int] = {}
        self._seen_by_user: dict[int, np.ndarray] = {}
        self._fit_item_ids: np.ndarray = np.empty(0, dtype="int64")
        self._mapping_checksum: str = ""
        self._dataset_identity: dict[str, Any] = {}
        self._loss_history: list[float] = []
        self._sampler_configuration: dict[str, Any] = {}

    # -- fitting ------------------------------------------------------------ #
    def fit(self, data: Any) -> None:
        """Train on implicit positives.

        Repeated ``(user, item)`` pairs are collapsed to **unique binary
        positives**. Measured on PixelRec50K, the training split contains zero
        repeats, so this changes nothing here - but making it explicit means a
        dataset that does repeat cannot let a handful of heavily-repeated pairs
        dominate the sampler. Recorded in the artifact metadata.
        """
        if not isinstance(data, BPRFitData):
            raise DataError(
                "BPRMatrixFactorization.fit expects a BPRFitData bundle",
                received=type(data).__name__,
            )
        frame = data.interactions
        if frame.empty:
            raise DataError("Cannot fit BPR on an empty interaction set")

        pairs = frame.loc[:, ["internal_user_id", "internal_item_id"]].drop_duplicates()
        collapsed = len(frame) - len(pairs)
        users = pairs["internal_user_id"].to_numpy(dtype="int64")
        items = pairs["internal_item_id"].to_numpy(dtype="int64")

        self._device = resolve_torch_device(self.device_preference)
        self._set_seeds(self.config.seed)

        positives_by_user = build_positives_by_user(users, items)
        sampler = UniformNegativeSampler(
            positives_by_user, catalogue_size=data.num_items, seed=self.config.seed
        )
        self._sampler_configuration = sampler.configuration

        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        # 0.1 scale keeps initial dot products small, so early sigmoids sit in
        # the responsive part of the curve rather than saturating.
        user_embedding = torch.nn.Embedding(data.num_users, self.config.embedding_dim, sparse=True)
        item_embedding = torch.nn.Embedding(data.num_items, self.config.embedding_dim, sparse=True)
        with torch.no_grad():
            user_embedding.weight.copy_(
                torch.randn(data.num_users, self.config.embedding_dim, generator=generator) * 0.1
            )
            item_embedding.weight.copy_(
                torch.randn(data.num_items, self.config.embedding_dim, generator=generator) * 0.1
            )
        user_embedding = user_embedding.to(self._device)
        item_embedding = item_embedding.to(self._device)

        # SparseAdam over sparse-gradient embeddings updates only the rows a
        # batch actually touched. Dense Adam updates all 119k rows every step -
        # for BPR, where a batch touches a few thousand, that is roughly an
        # order of magnitude of wasted work and dominates the epoch time.
        optimizer = torch.optim.SparseAdam(
            list(user_embedding.parameters()) + list(item_embedding.parameters()),
            lr=self.config.learning_rate,
        )
        user_tensor = torch.from_numpy(users).to(self._device)
        item_tensor = torch.from_numpy(items).to(self._device)
        rows = len(users)
        shuffle_generator = np.random.default_rng(self.config.seed)

        logger.info(
            "bpr.training_started",
            device=str(self._device),
            interactions=rows,
            collapsed_duplicate_pairs=collapsed,
            users=data.num_users,
            items=data.num_items,
            **self.config.to_dict(),
        )

        self._loss_history = []
        for epoch in range(self.config.epochs):
            order = shuffle_generator.permutation(rows)
            epoch_loss, batches = 0.0, 0
            for start in range(0, rows, self.config.batch_size):
                index = order[start : start + self.config.batch_size]
                batch_users = user_tensor[index]
                batch_items = item_tensor[index]
                negatives = sampler.sample(users[index], self.config.negatives_per_positive)
                negative_tensor = torch.from_numpy(negatives).to(self._device)

                loss = self._bpr_loss(
                    user_embedding, item_embedding, batch_users, batch_items, negative_tensor
                )
                if not torch.isfinite(loss):
                    raise DataError(
                        "BPR loss became non-finite. The learning rate is probably "
                        "too high for this configuration.",
                        epoch=epoch,
                        batch=batches,
                        learning_rate=self.config.learning_rate,
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                for module, label in ((user_embedding, "user"), (item_embedding, "item")):
                    gradient = module.weight.grad
                    if gradient is not None:
                        # A sparse gradient stores only the touched rows; check
                        # those values rather than materialising the dense form.
                        values = gradient._values() if gradient.is_sparse else gradient
                        if not torch.isfinite(values).all():
                            raise DataError(
                                "Non-finite gradient encountered during BPR training",
                                epoch=epoch,
                                tensor=label,
                            )
                optimizer.step()  # type: ignore[no-untyped-call]
                epoch_loss += float(loss.detach())
                batches += 1

            mean_loss = epoch_loss / max(batches, 1)
            self._loss_history.append(mean_loss)
            logger.info("bpr.epoch", epoch=epoch + 1, mean_loss=round(mean_loss, 6))

        self._user_factors = user_embedding.weight.detach()
        self._item_factors = item_embedding.weight.detach()
        self._internal_to_external = data.internal_to_external_item
        self._external_to_internal = {v: k for k, v in data.internal_to_external_item.items()}
        self._external_to_internal_user = data.external_to_internal_user
        self._seen_by_user = positives_by_user
        self._fit_item_ids = np.unique(items)
        self._mapping_checksum = data.mapping_checksum
        self._dataset_identity = data.dataset_identity
        self._fitted = True
        logger.info(
            "bpr.training_completed",
            epochs=self.config.epochs,
            first_loss=round(self._loss_history[0], 6),
            final_loss=round(self._loss_history[-1], 6),
        )

    def _bpr_loss(
        self,
        user_embedding: torch.nn.Embedding,
        item_embedding: torch.nn.Embedding,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """``-log sigmoid(y_ui - y_uj)`` plus L2, averaged over the batch.

        The L2 term covers only the embeddings this batch touched, which is what
        keeps the gradient sparse. Penalising the whole matrix would densify it
        and undo the SparseAdam saving.
        """
        user_embeddings = user_embedding(users)
        positive_embeddings = item_embedding(positives)
        # (batch, negatives, dim) so several negatives share one positive.
        negative_embeddings = item_embedding(negatives)

        positive_scores = (user_embeddings * positive_embeddings).sum(dim=1, keepdim=True)
        negative_scores = torch.einsum("bd,bnd->bn", user_embeddings, negative_embeddings)
        # softplus(-x) is -log(sigmoid(x)) computed without overflowing.
        ranking_loss = torch.nn.functional.softplus(-(positive_scores - negative_scores)).mean()

        penalty = (
            user_embeddings.pow(2).sum()
            + positive_embeddings.pow(2).sum()
            + negative_embeddings.pow(2).sum()
        ) / users.shape[0]
        total: torch.Tensor = ranking_loss + self.config.regularization * penalty
        return total

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
    def recommend(
        self, user_id: str, k: int, context: dict[str, Any] | None = None
    ) -> list[Candidate]:
        """Top-``k`` items for one user.

        An **unknown user returns an empty list**. A collaborative model has no
        representation for them, and inventing a random embedding would produce
        confident-looking nonsense. The serving fallback chain - not this model -
        is what answers for cold users.
        """
        self.ensure_fitted()
        internal_user = self._external_to_internal_user.get(user_id)
        if internal_user is None:
            logger.debug("bpr.unknown_user", user_id_known=False)
            return []
        filter_seen = True if context is None else bool(context.get("filter_seen", True))
        items, scores = self._top_k_for_users(
            np.array([internal_user], dtype="int64"), k, filter_seen=filter_seen
        )
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
        """Top-``k`` external item ids for many users, in memory-bounded batches.

        Never materialises a users x items matrix. Each batch scores
        ``batch_size x catalogue_size`` and reduces immediately with ``topk``, so
        peak memory is set by the configured batch size rather than by the user
        population. Unknown users receive an empty list.
        """
        self.ensure_fitted()
        results: dict[str, list[str]] = {}
        known: list[tuple[str, int]] = []
        for user_id in user_ids:
            internal = self._external_to_internal_user.get(user_id)
            if internal is None:
                results[user_id] = []
            else:
                known.append((user_id, internal))

        batch = self.config.evaluation_user_batch_size
        for start in range(0, len(known), batch):
            chunk = known[start : start + batch]
            internal_ids = np.array([internal for _, internal in chunk], dtype="int64")
            items, _ = self._top_k_for_users(internal_ids, k, filter_seen=filter_seen)
            for (user_id, _), row in zip(chunk, items, strict=True):
                results[user_id] = [
                    self._internal_to_external[int(item)] for item in row if item >= 0
                ]
        return results

    def _top_k_for_users(
        self, internal_users: np.ndarray, k: int, *, filter_seen: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score one user batch against the full catalogue and reduce to top-k.

        Returns ``(items, scores)`` padded with ``-1`` / ``-inf`` when fewer than
        ``k`` items survive masking, so callers get a rectangular result and can
        filter on the sentinel rather than on length.
        """
        if k < 1:
            raise DataError("k must be >= 1", k=k)
        assert self._user_factors is not None and self._item_factors is not None  # noqa: S101
        user_tensor = torch.from_numpy(internal_users).to(self._device)
        scores = self._user_factors[user_tensor] @ self._item_factors.T

        # Items outside the fit catalogue must never be recommended: their
        # factors are still at initialisation and mean nothing.
        outside = torch.ones(scores.shape[1], dtype=torch.bool, device=self._device)
        outside[torch.from_numpy(self._fit_item_ids).to(self._device)] = False
        scores[:, outside] = float("-inf")

        if filter_seen:
            for row, user in enumerate(internal_users.tolist()):
                seen = self._seen_by_user.get(int(user))
                if seen is not None and seen.size:
                    scores[row, torch.from_numpy(seen).to(self._device)] = float("-inf")

        take = min(k, scores.shape[1])
        top_scores, top_items = torch.topk(scores, take, dim=1)
        items = top_items.cpu().numpy().astype("int64")
        values = top_scores.cpu().numpy().astype("float64")
        # -inf means the position was masked out, not a real recommendation.
        items = np.where(np.isneginf(values), -1, items)
        if take < k:
            padding = np.full((len(internal_users), k - take), -1, dtype="int64")
            items = np.concatenate([items, padding], axis=1)
            values = np.concatenate(
                [values, np.full((len(internal_users), k - take), float("-inf"))], axis=1
            )
        return items, values

    def score(self, user_id: str, item_ids: list[str]) -> list[float]:
        """Score specific items for a user, in the order given.

        Unknown users and unknown items both score
        :data:`UNKNOWN_ITEM_SCORE` (0.0) rather than raising - both are
        legitimate inputs to a ranking stage.
        """
        self.ensure_fitted()
        assert self._user_factors is not None and self._item_factors is not None  # noqa: S101
        internal_user = self._external_to_internal_user.get(user_id)
        if internal_user is None:
            return [UNKNOWN_ITEM_SCORE] * len(item_ids)

        user_vector = self._user_factors[internal_user]
        scores: list[float] = []
        for item in item_ids:
            internal_item = self._external_to_internal.get(item)
            if internal_item is None:
                scores.append(UNKNOWN_ITEM_SCORE)
                continue
            scores.append(float(user_vector @ self._item_factors[internal_item]))
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
        """Device the factors currently live on."""
        return str(self._device)

    def metadata(self) -> dict[str, Any]:
        """Configuration and fit provenance, for the artifact manifest."""
        return {
            "model": self.name,
            "format_version": FORMAT_VERSION,
            "implementation": "bpr",
            "config": self.config.to_dict(),
            "device": str(self._device),
            # Full precision: the loss curve is a diagnostic record, and
            # rounding it here would mean a saved model no longer reports
            # exactly what it measured.
            "loss_history": self._loss_history,
            "catalogue_size": int(self._fit_item_ids.size),
            "mapping_checksum": self._mapping_checksum,
            "dataset_identity": self._dataset_identity,
            "negative_sampler": self._sampler_configuration,
            "duplicate_positive_policy": "unique_binary",
        }

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        """Persist to a directory: JSON metadata plus a device-neutral tensor file."""
        self.ensure_fitted()
        assert self._user_factors is not None and self._item_factors is not None  # noqa: S101
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)

        seen_users = sorted(self._seen_by_user)
        torch.save(
            {
                # Saved on CPU so an artifact trained on MPS loads on any host.
                "user_factors": self._user_factors.cpu(),
                "item_factors": self._item_factors.cpu(),
                "fit_item_ids": torch.from_numpy(self._fit_item_ids),
                "seen_users": torch.tensor(seen_users, dtype=torch.int64),
                "seen_lengths": torch.tensor(
                    [len(self._seen_by_user[user]) for user in seen_users], dtype=torch.int64
                ),
                "seen_flat": torch.from_numpy(
                    np.concatenate([self._seen_by_user[user] for user in seen_users])
                    if seen_users
                    else np.empty(0, dtype="int64")
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
        logger.info("bpr.saved", path=str(target))

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> Self:
        """Restore a saved model.

        Loads tensors with ``weights_only=True``: an artifact read from disk must
        not be able to execute code, and this state dict contains only tensors.

        Raises:
            ArtifactValidationError: Files missing, malformed, wrong model type,
                or an unsupported format version.
        """
        source = Path(path)
        state_path, config_path = source / _STATE_FILENAME, source / _CONFIG_FILENAME
        for candidate in (state_path, config_path):
            if not candidate.is_file():
                raise ArtifactValidationError("BPR artifact is incomplete", missing=str(candidate))
        try:
            metadata = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError(
                "BPR config is not valid JSON", path=str(config_path), reason=str(exc)[:200]
            ) from exc
        if metadata.get("model") != cls.name:
            raise ArtifactValidationError(
                "Artifact was written by a different model type",
                expected=cls.name,
                found=metadata.get("model"),
            )
        if metadata.get("format_version") != FORMAT_VERSION:
            raise ArtifactValidationError(
                "Unsupported BPR artifact format version",
                expected=FORMAT_VERSION,
                found=metadata.get("format_version"),
            )
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ArtifactValidationError(
                "BPR state file could not be read; it may be corrupted",
                path=str(state_path),
                reason=str(exc)[:200],
            ) from exc

        model = cls(BPRConfig(**metadata["config"]), device=device)
        model._device = resolve_torch_device(device)
        model._user_factors = state["user_factors"].to(model._device)
        model._item_factors = state["item_factors"].to(model._device)
        model._fit_item_ids = state["fit_item_ids"].numpy().astype("int64")
        model._internal_to_external = {
            int(key): str(value) for key, value in metadata["item_mapping"].items()
        }
        model._external_to_internal = {v: k for k, v in model._internal_to_external.items()}
        model._external_to_internal_user = {
            str(key): int(value) for key, value in metadata["user_mapping"].items()
        }
        flat = state["seen_flat"].numpy().astype("int64")
        offset = 0
        seen: dict[int, np.ndarray] = {}
        for user, length in zip(
            state["seen_users"].tolist(), state["seen_lengths"].tolist(), strict=True
        ):
            seen[int(user)] = flat[offset : offset + int(length)]
            offset += int(length)
        model._seen_by_user = seen
        model._mapping_checksum = metadata.get("mapping_checksum", "")
        model._dataset_identity = metadata.get("dataset_identity", {})
        model._loss_history = list(metadata.get("loss_history", []))
        model._sampler_configuration = metadata.get("negative_sampler", {})
        model._fitted = True
        logger.info("bpr.loaded", path=str(source), device=str(model._device))
        return model

    def require_mapping(self, mapping_checksum: str) -> None:
        """Assert this model was fitted against the given item mapping.

        Raises:
            ArtifactValidationError: Checksums differ.
        """
        if self._mapping_checksum and mapping_checksum != self._mapping_checksum:
            raise ArtifactValidationError(
                "Item mapping checksum does not match the one this model was "
                "fitted against. Every recommended id would resolve to the wrong item.",
                expected=self._mapping_checksum,
                found=mapping_checksum,
            )


__all__ = [
    "FORMAT_VERSION",
    "UNKNOWN_ITEM_SCORE",
    "BPRConfig",
    "BPRFitData",
    "BPRMatrixFactorization",
    "resolve_torch_device",
]
