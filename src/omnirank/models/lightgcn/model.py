"""LightGCN - light graph convolution for collaborative retrieval.

LightGCN's claim is subtractive: the feature transformation and non-linearity
that ordinary GCNs inherit from node-classification work do not help
collaborative filtering, and removing them leaves a model that is simpler,
faster, and better. What remains is neighbourhood averaging over the user-item
bipartite graph::

    e_u^(k+1) = sum_{i in N(u)} 1/sqrt(|N(u)||N(i)|) * e_i^(k)
    e_i^(k+1) = sum_{u in N(i)} 1/sqrt(|N(i)||N(u)|) * e_u^(k)

with the final representation a weighted sum of every layer::

    e_final = sum_k alpha_k * e^(k)

Equal layer weights (``alpha_k = 1/(K+1)``) are used, as in the paper.

**There is no weight matrix and no activation inside propagation.** A model with
either is not LightGCN, whatever it is called, and the tests assert their absence
by checking propagation against a hand-computed adjacency.

Training reuses the Phase 3 BPR objective and the tested uniform negative
sampler, so the only thing that differs from the matrix-factorization baseline is
the propagation - which is what makes the comparison between them meaningful.
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
from omnirank.models.base import Candidate, CandidateGenerator, ScoredCandidate
from omnirank.models.baselines.bpr import resolve_torch_device
from omnirank.models.baselines.negative_sampling import (
    UniformNegativeSampler,
    build_positives_by_user,
)

logger = get_logger(__name__)

FORMAT_VERSION: Final = 1
_STATE_FILENAME: Final = "state.pt"
_CONFIG_FILENAME: Final = "config.json"

#: Returned for items the model never saw.
UNKNOWN_ITEM_SCORE: Final = 0.0


@dataclass(frozen=True, slots=True)
class LightGCNConfig:
    """LightGCN hyperparameters. Every value validated, not merely stored."""

    embedding_dim: int = 128
    num_layers: int = 2
    learning_rate: float = 0.005
    regularization: float = 1e-4
    batch_size: int = 8192
    negatives_per_positive: int = 3
    max_epochs: int = 100
    early_stopping_patience: int = 10
    evaluation_user_batch_size: int = 256
    seed: int = 42

    def __post_init__(self) -> None:
        checks = (
            ("embedding_dim", self.embedding_dim > 0),
            # 0 layers is legal and meaningful: it degenerates to matrix
            # factorization, which is a useful ablation rather than an error.
            ("num_layers", self.num_layers >= 0),
            ("learning_rate", self.learning_rate > 0),
            ("regularization", self.regularization >= 0),
            ("batch_size", self.batch_size > 0),
            ("negatives_per_positive", self.negatives_per_positive > 0),
            ("max_epochs", self.max_epochs > 0),
            ("early_stopping_patience", self.early_stopping_patience > 0),
            ("evaluation_user_batch_size", self.evaluation_user_batch_size > 0),
            ("seed", self.seed >= 0),
        )
        problems = [name for name, ok in checks if not ok]
        if problems:
            raise DataError("Invalid LightGCN configuration", invalid=problems)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable payload."""
        return {
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
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
            f"d{self.embedding_dim}_L{self.num_layers}_lr{self.learning_rate}"
            f"_reg{self.regularization}_neg{self.negatives_per_positive}"
        )


@dataclass(frozen=True, slots=True)
class LightGCNFitData:
    """Graph edges plus the mappings they are expressed in."""

    edges: pd.DataFrame
    num_users: int
    num_items: int
    internal_to_external_item: dict[int, str]
    external_to_internal_user: dict[str, int]
    mapping_checksum: str = ""
    dataset_identity: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = {"internal_user_id", "internal_item_id"} - set(self.edges.columns)
        if missing:
            raise DataError("Graph edges missing columns", missing=sorted(missing))


def build_normalized_adjacency(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    *,
    num_users: int,
    num_items: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, str]:
    """Build the symmetric normalised bipartite adjacency as a sparse tensor.

    Nodes are laid out as ``[users | items]``, so the item at internal index
    ``i`` is node ``num_users + i``. The matrix is symmetric with entries::

        A_norm[u, i] = A_norm[i, u] = 1 / sqrt(deg(u) * deg(i))

    Degree-zero nodes get a zero row rather than a division by zero: an item
    nobody interacted with propagates nothing, which is correct, and it will
    never be recommended because it is outside the fit catalogue.

    Returns:
        ``(sparse_adjacency, checksum)``. The checksum covers the edge set and
        node counts, so a model can refuse a graph it was not trained on.
    """
    if num_users < 1 or num_items < 1:
        raise DataError(
            "Graph needs at least one user and one item", num_users=num_users, num_items=num_items
        )
    if user_ids.size != item_ids.size:
        raise DataError("Edge arrays must be the same length")
    if user_ids.size == 0:
        raise DataError("Cannot build a graph with no edges")
    if user_ids.max() >= num_users or item_ids.max() >= num_items:
        raise DataError(
            "Edge references a node outside the declared node counts",
            max_user=int(user_ids.max()),
            num_users=num_users,
            max_item=int(item_ids.max()),
            num_items=num_items,
        )

    total_nodes = num_users + num_items
    item_nodes = item_ids + num_users

    degrees = np.zeros(total_nodes, dtype="float64")
    np.add.at(degrees, user_ids, 1.0)
    np.add.at(degrees, item_nodes, 1.0)
    # 1/sqrt(0) would be inf; a zero-degree node simply propagates nothing.
    inverse_sqrt = np.zeros_like(degrees)
    nonzero = degrees > 0
    inverse_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])

    weights = inverse_sqrt[user_ids] * inverse_sqrt[item_nodes]
    # Both directions: the bipartite adjacency is symmetric.
    rows = np.concatenate([user_ids, item_nodes])
    cols = np.concatenate([item_nodes, user_ids])
    values = np.concatenate([weights, weights])

    indices = torch.from_numpy(np.vstack([rows, cols]).astype("int64"))
    adjacency = torch.sparse_coo_tensor(
        indices,
        torch.from_numpy(values.astype("float32")),
        size=(total_nodes, total_nodes),
    ).coalesce()
    if device is not None:
        adjacency = adjacency.to(device)

    checksum = _graph_checksum(user_ids, item_ids, num_users, num_items)
    logger.info(
        "lightgcn.graph_built",
        nodes=total_nodes,
        users=num_users,
        items=num_items,
        edges=int(user_ids.size),
        isolated_nodes=int((~nonzero).sum()),
        checksum=checksum[:16],
    )
    return adjacency, checksum


def _graph_checksum(
    user_ids: np.ndarray, item_ids: np.ndarray, num_users: int, num_items: int
) -> str:
    """Content hash of the edge set, order-independent."""
    import hashlib

    keys = np.sort(user_ids.astype("int64") * (num_items + 1) + item_ids.astype("int64"))
    digest = hashlib.sha256()
    digest.update(f"{num_users}:{num_items}:".encode())
    digest.update(keys.tobytes())
    return digest.hexdigest()


def propagate(adjacency: torch.Tensor, embeddings: torch.Tensor, num_layers: int) -> torch.Tensor:
    """Run light graph convolution and return the layer-averaged embeddings.

    No weight matrix, no activation - just repeated sparse multiplication and a
    mean over layers, which is the whole of LightGCN's propagation.
    """
    layers = [embeddings]
    current = embeddings
    for _ in range(num_layers):
        current = torch.sparse.mm(adjacency, current)
        layers.append(current)
    # Equal alpha_k, as in the paper.
    return torch.stack(layers, dim=0).mean(dim=0)


class LightGCN(CandidateGenerator):
    """LightGCN collaborative retrieval over the user-item bipartite graph."""

    name = "lightgcn"

    def __init__(self, config: LightGCNConfig | None = None, *, device: str = "auto") -> None:
        super().__init__()
        self.config = config or LightGCNConfig()
        self.device_preference = device
        self._device = torch.device("cpu")
        self._user_final: torch.Tensor | None = None
        self._item_final: torch.Tensor | None = None
        self._internal_to_external: dict[int, str] = {}
        self._external_to_internal: dict[str, int] = {}
        self._external_to_internal_user: dict[str, int] = {}
        self._seen_by_user: dict[int, np.ndarray] = {}
        self._fit_item_ids: np.ndarray = np.empty(0, dtype="int64")
        self._graph_checksum: str = ""
        self._mapping_checksum: str = ""
        self._dataset_identity: dict[str, Any] = {}
        self._loss_history: list[float] = []
        self._validation_history: list[float] = []
        self._best_epoch: int = 0
        self._sampler_configuration: dict[str, Any] = {}

    # -- fitting ------------------------------------------------------------ #
    def fit(self, data: Any) -> None:
        """Train with the BPR objective over propagated embeddings."""
        if not isinstance(data, LightGCNFitData):
            raise DataError(
                "LightGCN.fit expects a LightGCNFitData bundle",
                received=type(data).__name__,
            )
        pairs = data.edges.loc[:, ["internal_user_id", "internal_item_id"]].drop_duplicates()
        if pairs.empty:
            raise DataError("Cannot fit LightGCN on an empty graph")

        users = pairs["internal_user_id"].to_numpy(dtype="int64")
        items = pairs["internal_item_id"].to_numpy(dtype="int64")

        self._device = self._resolve_device()
        self._set_seeds(self.config.seed)

        adjacency, graph_checksum = build_normalized_adjacency(
            users,
            items,
            num_users=data.num_users,
            num_items=data.num_items,
            device=self._device,
        )

        positives_by_user = build_positives_by_user(users, items)
        sampler = UniformNegativeSampler(
            positives_by_user, catalogue_size=data.num_items, seed=self.config.seed
        )
        self._sampler_configuration = sampler.configuration

        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        embedding = torch.nn.Embedding(data.num_users + data.num_items, self.config.embedding_dim)
        with torch.no_grad():
            embedding.weight.copy_(
                torch.randn(
                    data.num_users + data.num_items,
                    self.config.embedding_dim,
                    generator=generator,
                )
                * 0.1
            )
        embedding = embedding.to(self._device)
        optimizer = torch.optim.Adam(embedding.parameters(), lr=self.config.learning_rate)

        user_tensor = torch.from_numpy(users).to(self._device)
        item_tensor = torch.from_numpy(items).to(self._device)
        rows = len(users)
        shuffle = np.random.default_rng(self.config.seed)

        logger.info(
            "lightgcn.training_started",
            device=str(self._device),
            edges=rows,
            users=data.num_users,
            items=data.num_items,
            **self.config.to_dict(),
        )

        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        patience = 0
        self._loss_history = []

        for epoch in range(self.config.max_epochs):
            order = shuffle.permutation(rows)
            epoch_loss, batches = 0.0, 0
            for start in range(0, rows, self.config.batch_size):
                index = order[start : start + self.config.batch_size]
                # Propagation runs once per batch because the embeddings change
                # every step; this is the dominant cost and why LightGCN is
                # slower per epoch than plain matrix factorization.
                propagated = propagate(adjacency, embedding.weight, self.config.num_layers)
                user_vectors = propagated[user_tensor[index]]
                positive_vectors = propagated[item_tensor[index] + data.num_users]

                negatives = sampler.sample(users[index], self.config.negatives_per_positive)
                negative_vectors = propagated[
                    torch.from_numpy(negatives).to(self._device) + data.num_users
                ]

                positive_scores = (user_vectors * positive_vectors).sum(dim=1, keepdim=True)
                negative_scores = torch.einsum("bd,bnd->bn", user_vectors, negative_vectors)
                ranking = torch.nn.functional.softplus(-(positive_scores - negative_scores)).mean()
                # Regularise the *base* embeddings, not the propagated ones:
                # propagated vectors are functions of the base, so penalising
                # them double-counts.
                penalty = (
                    embedding.weight[user_tensor[index]].pow(2).sum()
                    + embedding.weight[item_tensor[index] + data.num_users].pow(2).sum()
                ) / len(index)
                loss = ranking + self.config.regularization * penalty

                if not torch.isfinite(loss):
                    raise DataError(
                        "LightGCN loss became non-finite; the learning rate is probably too high.",
                        epoch=epoch,
                        learning_rate=self.config.learning_rate,
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                gradient = embedding.weight.grad
                if gradient is not None and not torch.isfinite(gradient).all():
                    raise DataError("Non-finite gradient during LightGCN training", epoch=epoch)
                optimizer.step()
                epoch_loss += float(loss.detach())
                batches += 1

            mean_loss = epoch_loss / max(batches, 1)
            self._loss_history.append(mean_loss)
            logger.info("lightgcn.epoch", epoch=epoch + 1, mean_loss=round(mean_loss, 6))

            # Early stopping on training loss. Using a validation metric would be
            # better, but it would need a retrieval pass per epoch; the loss is a
            # documented proxy and the patience is configurable.
            if mean_loss < best_loss - 1e-6:
                best_loss = mean_loss
                best_state = {"weight": embedding.weight.detach().clone()}
                self._best_epoch = epoch + 1
                patience = 0
            else:
                patience += 1
                if patience >= self.config.early_stopping_patience:
                    logger.info(
                        "lightgcn.early_stopped",
                        epoch=epoch + 1,
                        best_epoch=self._best_epoch,
                    )
                    break

        if best_state is not None:
            with torch.no_grad():
                embedding.weight.copy_(best_state["weight"])

        with torch.no_grad():
            propagated = propagate(adjacency, embedding.weight, self.config.num_layers)
            self._user_final = propagated[: data.num_users].detach()
            self._item_final = propagated[data.num_users :].detach()

        self._internal_to_external = data.internal_to_external_item
        self._external_to_internal = {v: k for k, v in data.internal_to_external_item.items()}
        self._external_to_internal_user = data.external_to_internal_user
        self._seen_by_user = positives_by_user
        self._fit_item_ids = np.unique(items)
        self._graph_checksum = graph_checksum
        self._mapping_checksum = data.mapping_checksum
        self._dataset_identity = data.dataset_identity
        self._fitted = True
        logger.info(
            "lightgcn.training_completed",
            epochs_run=len(self._loss_history),
            best_epoch=self._best_epoch,
            first_loss=round(self._loss_history[0], 6),
            final_loss=round(self._loss_history[-1], 6),
        )

    def _resolve_device(self) -> torch.device:
        """Pick a device, falling back to CPU when sparse ops are unsupported.

        MPS sparse support has historically been incomplete. Rather than
        discovering that mid-training, a tiny sparse matmul is attempted first
        and the fallback is logged - never a silent mix of devices.
        """
        device = resolve_torch_device(self.device_preference)
        if device.type != "mps":
            return device
        try:
            probe_indices = torch.tensor([[0, 1], [1, 0]], device=device)
            probe = torch.sparse_coo_tensor(
                probe_indices, torch.ones(2, device=device), size=(2, 2)
            ).coalesce()
            torch.sparse.mm(probe, torch.ones(2, 2, device=device))
        except Exception as exc:  # any failure means fall back
            logger.warning(
                "lightgcn.mps_sparse_unsupported",
                falling_back_to="cpu",
                reason=str(exc)[:200],
                detail=(
                    "Sparse matmul failed on MPS, which LightGCN propagation "
                    "requires. Running entirely on CPU rather than splitting "
                    "tensors across devices."
                ),
            )
            return torch.device("cpu")
        return device

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
        """Top-``k`` items for one user. Unknown users get an empty list."""
        self.ensure_fitted()
        internal = self._external_to_internal_user.get(user_id)
        if internal is None:
            return []
        filter_seen = True if context is None else bool(context.get("filter_seen", True))
        items, scores = self._top_k(np.array([internal], dtype="int64"), k, filter_seen=filter_seen)
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
            if internal is None:
                results[user_id] = []
            else:
                known.append((user_id, internal))

        batch = self.config.evaluation_user_batch_size
        for start in range(0, len(known), batch):
            chunk = known[start : start + batch]
            internal_ids = np.array([internal for _, internal in chunk], dtype="int64")
            items, _ = self._top_k(internal_ids, k, filter_seen=filter_seen)
            for (user_id, _), row in zip(chunk, items, strict=True):
                results[user_id] = [
                    self._internal_to_external[int(item)] for item in row if item >= 0
                ]
        return results

    def recommend_batch_scored(
        self, user_ids: list[str], k: int, *, filter_seen: bool = True
    ) -> dict[str, list[ScoredCandidate]]:
        """Batch retrieval that keeps the scores instead of discarding them.

        :meth:`recommend_batch` computes exactly these values and then drops
        them with ``items, _ = ...``. A ranking snapshot needs the score as well
        as the position, so this variant returns both rather than leaving a
        caller to reconstruct a stand-in from the rank.
        """
        self.ensure_fitted()
        results: dict[str, list[ScoredCandidate]] = {}
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
            items, scores = self._top_k(internal_ids, k, filter_seen=filter_seen)
            for (user_id, _), row, row_scores in zip(chunk, items, scores, strict=True):
                results[user_id] = [
                    ScoredCandidate(
                        item_id=self._internal_to_external[int(item)],
                        rank=position,
                        score=float(value),
                        source=self.name,
                    )
                    for position, (item, value) in enumerate(
                        zip(row, row_scores, strict=True), start=1
                    )
                    if item >= 0
                ]
        return results

    def _top_k(
        self, internal_users: np.ndarray, k: int, *, filter_seen: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score one user batch against the catalogue and reduce to top-k."""
        if k < 1:
            raise DataError("k must be >= 1", k=k)
        assert self._user_final is not None and self._item_final is not None  # noqa: S101
        user_tensor = torch.from_numpy(internal_users).to(self._device)
        scores = self._user_final[user_tensor] @ self._item_final.T

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
        assert self._user_final is not None and self._item_final is not None  # noqa: S101
        internal_user = self._external_to_internal_user.get(user_id)
        if internal_user is None:
            return [UNKNOWN_ITEM_SCORE] * len(item_ids)
        user_vector = self._user_final[internal_user]
        scores: list[float] = []
        for item in item_ids:
            internal_item = self._external_to_internal.get(item)
            scores.append(
                UNKNOWN_ITEM_SCORE
                if internal_item is None
                else float(user_vector @ self._item_final[internal_item])
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
        """Device the embeddings live on."""
        return str(self._device)

    @property
    def graph_checksum(self) -> str:
        """Checksum of the graph this model was trained on."""
        return self._graph_checksum

    def item_embeddings(self) -> np.ndarray:
        """Final propagated item embeddings, for building a vector index."""
        self.ensure_fitted()
        assert self._item_final is not None  # noqa: S101
        return self._item_final.cpu().numpy().astype("float32")

    def user_embeddings(self) -> np.ndarray:
        """Final propagated user embeddings, for querying a vector index."""
        self.ensure_fitted()
        assert self._user_final is not None  # noqa: S101
        return self._user_final.cpu().numpy().astype("float32")

    def metadata(self) -> dict[str, Any]:
        """Configuration and fit provenance, for the artifact manifest."""
        return {
            "model": self.name,
            "format_version": FORMAT_VERSION,
            "config": self.config.to_dict(),
            "device": str(self._device),
            "loss_history": self._loss_history,
            "validation_history": self._validation_history,
            "best_epoch": self._best_epoch,
            "epochs_run": len(self._loss_history),
            "catalogue_size": int(self._fit_item_ids.size),
            "graph_checksum": self._graph_checksum,
            "mapping_checksum": self._mapping_checksum,
            "dataset_identity": self._dataset_identity,
            "negative_sampler": self._sampler_configuration,
            "propagation": "symmetric normalised, no transform, no activation",
            "layer_combination": "equal alpha_k (mean over layers 0..K)",
        }

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        """Persist to a directory, device-neutrally."""
        self.ensure_fitted()
        assert self._user_final is not None and self._item_final is not None  # noqa: S101
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        seen_users = sorted(self._seen_by_user)
        torch.save(
            {
                "user_final": self._user_final.cpu(),
                "item_final": self._item_final.cpu(),
                "fit_item_ids": torch.from_numpy(self._fit_item_ids),
                "seen_users": torch.tensor(seen_users, dtype=torch.int64),
                "seen_lengths": torch.tensor(
                    [len(self._seen_by_user[u]) for u in seen_users], dtype=torch.int64
                ),
                "seen_flat": torch.from_numpy(
                    np.concatenate([self._seen_by_user[u] for u in seen_users])
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
        logger.info("lightgcn.saved", path=str(target))

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> Self:
        """Restore a saved model.

        Raises:
            ArtifactValidationError: Files missing, malformed, wrong model type,
                or an unsupported format version.
        """
        source = Path(path)
        state_path, config_path = source / _STATE_FILENAME, source / _CONFIG_FILENAME
        for candidate in (state_path, config_path):
            if not candidate.is_file():
                raise ArtifactValidationError(
                    "LightGCN artifact is incomplete", missing=str(candidate)
                )
        try:
            metadata = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError(
                "LightGCN config is not valid JSON", path=str(config_path)
            ) from exc
        if metadata.get("model") != cls.name:
            raise ArtifactValidationError(
                "Artifact was written by a different model type",
                expected=cls.name,
                found=metadata.get("model"),
            )
        if metadata.get("format_version") != FORMAT_VERSION:
            raise ArtifactValidationError(
                "Unsupported LightGCN artifact format version",
                expected=FORMAT_VERSION,
                found=metadata.get("format_version"),
            )
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ArtifactValidationError(
                "LightGCN state file could not be read; it may be corrupted",
                path=str(state_path),
                reason=str(exc)[:200],
            ) from exc

        model = cls(LightGCNConfig(**metadata["config"]), device=device)
        model._device = resolve_torch_device(device)
        model._user_final = state["user_final"].to(model._device)
        model._item_final = state["item_final"].to(model._device)
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
        model._graph_checksum = metadata.get("graph_checksum", "")
        model._mapping_checksum = metadata.get("mapping_checksum", "")
        model._dataset_identity = metadata.get("dataset_identity", {})
        model._loss_history = list(metadata.get("loss_history", []))
        model._validation_history = list(metadata.get("validation_history", []))
        model._best_epoch = int(metadata.get("best_epoch", 0))
        model._sampler_configuration = metadata.get("negative_sampler", {})
        model._fitted = True
        logger.info("lightgcn.loaded", path=str(source), device=str(model._device))
        return model

    def require_mapping(self, mapping_checksum: str) -> None:
        """Assert this model was fitted against the given item mapping."""
        if self._mapping_checksum and mapping_checksum != self._mapping_checksum:
            raise ArtifactValidationError(
                "Item mapping checksum does not match the one this model was "
                "fitted against. Every recommended id would resolve to the wrong item.",
                expected=self._mapping_checksum,
                found=mapping_checksum,
            )

    def require_graph(self, graph_checksum: str) -> None:
        """Assert this model was fitted on the given graph.

        Raises:
            ArtifactValidationError: The graph differs. Propagated embeddings are
                a function of the adjacency, so a different graph makes them
                meaningless without making them look wrong.
        """
        if self._graph_checksum and graph_checksum != self._graph_checksum:
            raise ArtifactValidationError(
                "Graph checksum does not match the graph this model was trained "
                "on. Its propagated embeddings encode a different adjacency.",
                expected=self._graph_checksum,
                found=graph_checksum,
            )


__all__ = [
    "FORMAT_VERSION",
    "UNKNOWN_ITEM_SCORE",
    "LightGCN",
    "LightGCNConfig",
    "LightGCNFitData",
    "build_normalized_adjacency",
    "propagate",
]
