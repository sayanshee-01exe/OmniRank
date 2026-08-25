"""The two-tower retriever as a :class:`CandidateGenerator`.

Wraps the network in the same interface popularity, BPR, LightGCN and SASRec
already implement, so the shared evaluation harness and the aggregator treat it
identically. That is what makes a four-source-versus-five-source comparison a
comparison rather than a juxtaposition of two measurement systems.

**Public ids are external, internal computation is internal.** Every id crossing
the boundary is translated through the mapping the model was fitted against, and
an unknown id resolves to nothing rather than to item 0.

**No fallback lives here.** An unknown user with no history gets an empty list,
not popularity. Substituting a different model's output inside this class would
make "the two-tower retrieved it" untrue for an unknown fraction of requests,
and every downstream contribution metric would inherit the lie. Fallback is the
orchestration layer's job, where it is explicit and logged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Self

import numpy as np
import torch

from omnirank.core.exceptions import ArtifactValidationError, DataError
from omnirank.core.logging import get_logger
from omnirank.features.multimodal_store import MultimodalFeatureStore
from omnirank.models.base import Candidate, CandidateGenerator
from omnirank.models.baselines.bpr import resolve_torch_device
from omnirank.models.two_tower.catalogue import RetrievalCatalogue, build_catalogue
from omnirank.models.two_tower.config import TwoTowerConfig
from omnirank.models.two_tower.model import MultimodalTwoTower
from omnirank.models.two_tower.persistence import build_metadata, load, save

logger = get_logger(__name__)

SOURCE_NAME: Final = "two_tower"
UNKNOWN_ITEM_SCORE: Final = 0.0
#: Retrieval depth multiplier before seen-item filtering, and the cap on how
#: far it may grow. A user who has seen most of the catalogue must terminate
#: with a short list rather than looping.
DEFAULT_OVERSAMPLING_FACTOR: Final = 2
DEFAULT_MAXIMUM_SEARCH_MULTIPLIER: Final = 16
EMPTY_SLOT: Final = -1


class TwoTowerRetriever(CandidateGenerator):
    """Full-catalogue retrieval over multimodal two-tower embeddings."""

    name = SOURCE_NAME

    def __init__(
        self,
        model: MultimodalTwoTower,
        store: MultimodalFeatureStore,
        *,
        internal_to_external_item: dict[int, str],
        external_to_internal_user: dict[str, int],
        item_tags: np.ndarray,
        histories: dict[int, list[int]],
        warm_items: np.ndarray,
        device: str = "cpu",
        mapping_checksum: str = "",
        dataset_identity: dict[str, Any] | None = None,
        oversampling_factor: int = DEFAULT_OVERSAMPLING_FACTOR,
        maximum_search_multiplier: int = DEFAULT_MAXIMUM_SEARCH_MULTIPLIER,
    ) -> None:
        super().__init__()
        if maximum_search_multiplier < oversampling_factor:
            raise DataError(
                "maximum_search_multiplier must be at least oversampling_factor",
                oversampling_factor=oversampling_factor,
                maximum_search_multiplier=maximum_search_multiplier,
            )
        self.model = model
        self.store = store
        self.config: TwoTowerConfig = model.config
        self._device = resolve_torch_device(device)
        self.model.to(self._device)
        self.model.eval()

        self._internal_to_external = internal_to_external_item
        self._external_to_internal = {v: k for k, v in internal_to_external_item.items()}
        self._external_to_internal_user = external_to_internal_user
        self._tags = np.asarray(item_tags, dtype="int64")
        self._histories = histories
        self._seen = {user: set(items) for user, items in histories.items()}
        self._warm = np.asarray(warm_items, dtype=bool)
        self.mapping_checksum = mapping_checksum
        self.dataset_identity = dataset_identity or {}
        self.oversampling_factor = oversampling_factor
        self.maximum_search_multiplier = maximum_search_multiplier

        self.catalogue: RetrievalCatalogue = build_catalogue(
            warm_items=self._warm,
            text_available=store.has_modality("text"),
            image_available=store.has_modality("image"),
            internal_to_external=internal_to_external_item,
        )
        self._catalogue_ids = self.catalogue.internal_ids
        # Position of each catalogue item within the embedding matrix. Retrieval
        # returns rows; this maps them back to item ids.
        self._row_of = {int(item): row for row, item in enumerate(self._catalogue_ids)}
        self._embeddings: np.ndarray | None = None
        self._fitted = True

    # -- fitting ------------------------------------------------------------ #
    def fit(self, data: Any) -> None:
        """Not supported: the network is trained by :class:`TwoTowerTrainer`.

        Separating them keeps this class about retrieval. A ``fit`` here would
        need a feature store, a sequence table and a device policy, duplicating
        the trainer for no benefit.
        """
        raise DataError(
            "TwoTowerRetriever does not train. Fit the network with "
            "TwoTowerTrainer, then wrap it here.",
        )

    @classmethod
    def from_trained(
        cls,
        model: MultimodalTwoTower,
        store: MultimodalFeatureStore,
        dataset: Any,
        histories: dict[int, list[int]],
        warm_items: np.ndarray,
        item_tags: np.ndarray,
        *,
        device: str = "cpu",
        **kwargs: Any,
    ) -> TwoTowerRetriever:
        """Build a retriever from a trained network and its fitting context."""
        return cls(
            model,
            store,
            internal_to_external_item=dataset.internal_to_external_items(),
            external_to_internal_user=dataset.external_to_internal_users(),
            item_tags=item_tags,
            histories=histories,
            warm_items=warm_items,
            device=device,
            mapping_checksum=dataset.mapping_metadata.get("item_mapping_checksum", ""),
            dataset_identity=dataset.identity.to_dict(),
            **kwargs,
        )

    # -- encoding ----------------------------------------------------------- #
    def encode_items(self, internal_ids: np.ndarray) -> np.ndarray:
        """Encode a batch of items, gating the identity residual on warmth."""
        ids = np.asarray(internal_ids, dtype="int64")
        features = self.store.get_batch(ids)
        with torch.no_grad():
            vectors = self.model.encode_items(
                torch.from_numpy(features.text).to(self._device),
                torch.from_numpy(features.image).to(self._device),
                torch.from_numpy(features.text_mask).to(self._device),
                torch.from_numpy(features.image_mask).to(self._device),
                torch.from_numpy(self._tags[ids]).to(self._device),
                torch.from_numpy(ids).to(self._device),
                torch.from_numpy(self._warm[ids]).to(self._device),
            )
        return vectors.cpu().numpy().astype("float32")

    def export_item_embeddings(self, batch_size: int = 2048) -> np.ndarray:
        """Encode the whole catalogue in bounded memory.

        Batched because the alternative -- one forward pass over 69,347 items
        with 1024-d text and image inputs -- materialises roughly 570 MB of
        input activations before the first projection.
        """
        if batch_size < 1:
            raise DataError("batch_size must be positive", batch_size=batch_size)
        blocks = [
            self.encode_items(self._catalogue_ids[start : start + batch_size])
            for start in range(0, self._catalogue_ids.size, batch_size)
        ]
        embeddings = np.concatenate(blocks, axis=0) if blocks else np.empty((0, 0), "float32")
        if not np.isfinite(embeddings).all():
            raise DataError(
                "Item embeddings contain non-finite values. An index built over "
                "these would return them as neighbours with meaningless scores."
            )
        if embeddings.shape != (len(self.catalogue), self.config.embedding_dim):
            raise DataError(
                "Exported embeddings do not match the catalogue",
                shape=list(embeddings.shape),
                expected=[len(self.catalogue), self.config.embedding_dim],
            )
        self._embeddings = embeddings
        logger.info(
            "two_tower.item_embeddings_exported",
            items=int(embeddings.shape[0]),
            dimension=int(embeddings.shape[1]),
            warm=self.catalogue.warm_count,
            cold=self.catalogue.cold_count,
            megabytes=round(embeddings.nbytes / 1e6, 1),
        )
        return embeddings

    @property
    def item_embeddings_matrix(self) -> np.ndarray:
        """Catalogue embeddings, exporting them on first use."""
        if self._embeddings is None:
            self.export_item_embeddings()
        assert self._embeddings is not None  # noqa: S101
        return self._embeddings

    def item_embeddings(self) -> np.ndarray:
        """Alias matching the other retrievers' export surface."""
        return self.item_embeddings_matrix

    def build_query_embedding(
        self, histories: list[list[int]], user_ids: list[int] | None = None
    ) -> np.ndarray:
        """Encode user queries from item histories.

        ``user_ids=None`` builds the query from history alone, which is the
        unknown-user path: an identity embedding that was never trained would
        contribute noise rather than information.
        """
        width = self.config.maximum_history_length
        padding = self.model.num_items
        rows = len(histories)
        if rows == 0:
            return np.empty((0, self.config.embedding_dim), dtype="float32")

        ids = np.full((rows, width), padding, dtype="int64")
        lengths = np.zeros(rows, dtype="int64")
        for row, history in enumerate(histories):
            # Oldest dropped first; the newest item lands in the final column so
            # recency weighting can index from the end.
            recent = list(history)[-width:]
            lengths[row] = len(recent)
            if recent:
                ids[row, width - len(recent) :] = recent

        mask = ids == padding
        safe = np.where(mask, 0, ids)
        features = self.store.get_batch(safe.ravel())
        text = features.text.reshape(rows, width, -1)
        image = features.image.reshape(rows, width, -1)
        text_mask = features.text_mask.reshape(rows, width) & ~mask
        image_mask = features.image_mask.reshape(rows, width) & ~mask
        tags = np.where(mask, 0, self._tags[safe])

        with torch.no_grad():
            queries = self.model.encode_users(
                torch.from_numpy(text * text_mask[..., None]).to(self._device),
                torch.from_numpy(image * image_mask[..., None]).to(self._device),
                torch.from_numpy(text_mask).to(self._device),
                torch.from_numpy(image_mask).to(self._device),
                torch.from_numpy(tags).to(self._device),
                torch.from_numpy(mask).to(self._device),
                torch.from_numpy(lengths).to(self._device),
                (
                    torch.from_numpy(np.asarray(user_ids, dtype="int64")).to(self._device)
                    if user_ids is not None and self.config.use_user_id_embedding
                    else None
                ),
            )
        return queries.cpu().numpy().astype("float32")

    def encode_users(self, user_ids: list[str]) -> np.ndarray:
        """Query vectors for known external users."""
        internal = [
            self._external_to_internal_user[user]
            for user in user_ids
            if user in self._external_to_internal_user
        ]
        if not internal:
            return np.empty((0, self.config.embedding_dim), dtype="float32")
        return self.build_query_embedding(
            [self._histories.get(user, []) for user in internal], internal
        )

    # -- retrieval ---------------------------------------------------------- #
    def _search(
        self, queries: np.ndarray, k: int, excluded: list[set[int]] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rank the catalogue, filtering seen items with bounded over-retrieval."""
        matrix = self.item_embeddings_matrix
        scores = queries @ matrix.T
        rows, catalogue_size = scores.shape

        if excluded is None:
            take = min(k, catalogue_size)
            order = np.argsort(-scores, axis=1, kind="stable")[:, :take]
            values = np.take_along_axis(scores, order, axis=1)
            return self._pad(self._catalogue_ids[order], values, k)

        # Grow the retrieval depth geometrically rather than re-ranking the
        # whole catalogue per user, and cap it so a user who has seen almost
        # everything terminates with a short list instead of looping.
        depth = min(max(k * self.oversampling_factor, k), catalogue_size)
        cap = min(k * self.maximum_search_multiplier, catalogue_size)
        items = np.full((rows, k), EMPTY_SLOT, dtype="int64")
        kept_scores = np.full((rows, k), -np.inf, dtype="float64")
        underfilled = 0

        while True:
            order = np.argsort(-scores, axis=1, kind="stable")[:, :depth]
            values = np.take_along_axis(scores, order, axis=1)
            candidates = self._catalogue_ids[order]
            incomplete = False
            for row in range(rows):
                blocked = excluded[row]
                keep_items: list[int] = []
                keep_scores: list[float] = []
                for item, score in zip(candidates[row], values[row], strict=True):
                    if int(item) in blocked:
                        continue
                    keep_items.append(int(item))
                    keep_scores.append(float(score))
                    if len(keep_items) == k:
                        break
                items[row, : len(keep_items)] = keep_items
                kept_scores[row, : len(keep_scores)] = keep_scores
                if len(keep_items) < k:
                    incomplete = True
            if not incomplete or depth >= cap:
                underfilled = int((items == EMPTY_SLOT).any(axis=1).sum())
                break
            depth = min(depth * 2, cap)

        if underfilled:
            logger.info(
                "two_tower.underfilled_recommendations",
                users=underfilled,
                requested=k,
                depth=depth,
                detail="Catalogue exhausted after seen-item filtering.",
            )
        return items, kept_scores

    @staticmethod
    def _pad(items: np.ndarray, scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Pad a short result to width k with empty slots."""
        if items.shape[1] >= k:
            return items, scores
        pad = k - items.shape[1]
        return (
            np.concatenate([items, np.full((items.shape[0], pad), EMPTY_SLOT, "int64")], axis=1),
            np.concatenate([scores, np.full((scores.shape[0], pad), -np.inf)], axis=1),
        )

    def recommend(
        self, user_id: str, k: int, context: dict[str, Any] | None = None
    ) -> list[Candidate]:
        """Top-``k`` candidates for one user.

        ``context["history"]`` supplies an external-id history for an unknown
        user. Without it, an unknown user returns an empty list -- the caller's
        fallback chain decides what to do, not this class.
        """
        self.ensure_fitted()
        if k < 1:
            raise DataError("k must be >= 1", k=k)
        internal = self._external_to_internal_user.get(user_id)
        supplied = (context or {}).get("history")

        if internal is not None:
            history, user_ids = self._histories.get(internal, []), [internal]
        elif supplied:
            history = [
                self._external_to_internal[item]
                for item in supplied
                if item in self._external_to_internal
            ]
            # No identity for an unknown user: the query is content only.
            user_ids = None
        else:
            return []
        if not history:
            return []

        filter_seen = True if context is None else bool(context.get("filter_seen", True))
        seen = self._seen.get(internal, set()) if internal is not None else set(history)
        queries = self.build_query_embedding([history], user_ids)
        items, scores = self._search(queries, k, [seen] if filter_seen else None)
        return [
            Candidate(
                item_id=self._internal_to_external[int(item)],
                score=float(score),
                sources=(self.name,),
                source_scores={self.name: float(score)},
            )
            for item, score in zip(items[0], scores[0], strict=True)
            if item != EMPTY_SLOT
        ]

    def recommend_batch(
        self, user_ids: list[str], k: int, *, filter_seen: bool = True
    ) -> dict[str, list[str]]:
        """Top-``k`` external item ids for many users, memory-bounded."""
        self.ensure_fitted()
        if k < 1:
            raise DataError("k must be >= 1", k=k)
        results: dict[str, list[str]] = {}
        known: list[tuple[str, int]] = []
        for user in user_ids:
            internal = self._external_to_internal_user.get(user)
            if internal is None or not self._histories.get(internal):
                results[user] = []
            else:
                known.append((user, internal))

        batch = self.config.evaluation_user_batch_size
        for start in range(0, len(known), batch):
            chunk = known[start : start + batch]
            queries = self.build_query_embedding(
                [self._histories[internal] for _, internal in chunk],
                [internal for _, internal in chunk],
            )
            excluded = (
                [self._seen.get(internal, set()) for _, internal in chunk] if filter_seen else None
            )
            items, _ = self._search(queries, k, excluded)
            for (user, _), row in zip(chunk, items, strict=True):
                results[user] = [
                    self._internal_to_external[int(item)] for item in row if item != EMPTY_SLOT
                ]
        return results

    def score(self, user_id: str, item_ids: list[str]) -> list[float]:
        """Score specific items. Unknown users and items score 0.0."""
        self.ensure_fitted()
        internal = self._external_to_internal_user.get(user_id)
        if internal is None or not self._histories.get(internal):
            return [UNKNOWN_ITEM_SCORE] * len(item_ids)
        query = self.build_query_embedding([self._histories[internal]], [internal])[0]
        matrix = self.item_embeddings_matrix
        scores: list[float] = []
        for item in item_ids:
            item_internal = self._external_to_internal.get(item)
            row = self._row_of.get(item_internal) if item_internal is not None else None
            scores.append(UNKNOWN_ITEM_SCORE if row is None else float(query @ matrix[row]))
        return scores

    # -- introspection ------------------------------------------------------ #
    @property
    def fit_item_catalogue(self) -> set[int]:
        """Internal ids this retriever can return -- warm *and* cold."""
        return set(self._catalogue_ids.tolist())

    @property
    def cold_item_catalogue(self) -> set[int]:
        """Cold items reachable through content. Empty for every other source."""
        return set(self._catalogue_ids[~self.catalogue.warm_mask].tolist())

    @property
    def device(self) -> str:
        """Device the network runs on."""
        return str(self._device)

    def metadata(self) -> dict[str, Any]:
        """Configuration, identity and catalogue composition."""
        return {
            "model": self.name,
            "config": self.config.to_dict(),
            "device": str(self._device),
            "embedding_dim": self.config.embedding_dim,
            "normalization": "l2" if self.config.l2_normalize else "none",
            "temperature": self.config.temperature,
            "modality_schema": self.model.modality_schema(),
            "mapping_checksum": self.mapping_checksum,
            "feature_version": self.store.feature_version,
            "feature_manifest_checksum": self.store.manifest_checksum(),
            "dataset_identity": self.dataset_identity,
            "catalogue_size": len(self.catalogue),
            "warm_items": self.catalogue.warm_count,
            "cold_items": self.catalogue.cold_count,
            "excluded_items": self.catalogue.excluded_count,
            "catalogue_checksum": self.catalogue.checksum(),
            "oversampling_factor": self.oversampling_factor,
            "maximum_search_multiplier": self.maximum_search_multiplier,
        }

    def require_mapping(self, mapping_checksum: str) -> None:
        """Assert this retriever was fitted against the given item mapping."""
        if self.mapping_checksum and mapping_checksum != self.mapping_checksum:
            raise ArtifactValidationError(
                "Item mapping checksum does not match the one this model was "
                "fitted against. Every recommended id would resolve to a "
                "different item.",
                expected=self.mapping_checksum,
                found=mapping_checksum,
            )

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        """Persist the network plus the retrieval context it needs."""
        self.ensure_fitted()
        target = Path(path)
        metadata = build_metadata(
            self.model,
            feature_version=self.store.feature_version,
            feature_manifest_checksum=self.store.manifest_checksum(),
            mapping_checksum=self.mapping_checksum,
            dataset_identity=self.dataset_identity,
        )
        metadata["retriever"] = self.metadata()
        save(self.model, target, metadata=metadata)

        users = sorted(self._histories)
        np.savez_compressed(
            target / "retrieval_context.npz",
            item_tags=self._tags,
            warm_items=self._warm,
            history_users=np.asarray(users, dtype="int64"),
            history_lengths=np.asarray(
                [len(self._histories[user]) for user in users], dtype="int64"
            ),
            history_flat=np.asarray(
                [item for user in users for item in self._histories[user]], dtype="int64"
            ),
            external_items=np.asarray(
                [self._internal_to_external.get(i, "") for i in range(self.model.num_items)],
                dtype=object,
            ),
            external_users=np.asarray(sorted(self._external_to_internal_user), dtype=object),
            internal_users=np.asarray(
                [
                    self._external_to_internal_user[u]
                    for u in sorted(self._external_to_internal_user)
                ],
                dtype="int64",
            ),
            allow_pickle=True,
        )
        logger.info("two_tower.retriever_saved", path=str(target))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        store: MultimodalFeatureStore | None = None,
        device: str = "cpu",
    ) -> Self:
        """Restore a retriever, refusing an incompatible feature store."""
        source = Path(path)
        context_path = source / "retrieval_context.npz"
        if not context_path.is_file():
            raise ArtifactValidationError(
                "Retriever artifact is missing its retrieval context",
                missing=str(context_path),
            )
        if store is None:
            raise DataError(
                "A feature store is required to load a two-tower retriever: its "
                "item vectors come from the store, not from the checkpoint."
            )
        model, metadata = load(
            source,
            device=device,
            expected_mapping_checksum=None,
            expected_feature_version=store.feature_version,
        )
        store.require_compatible(
            mapping_checksum=metadata.get("mapping_checksum", ""),
            feature_version=metadata.get("feature_version", ""),
        )

        context = np.load(context_path, allow_pickle=True)
        flat = context["history_flat"].tolist()
        offset = 0
        histories: dict[int, list[int]] = {}
        for user, length in zip(
            context["history_users"].tolist(), context["history_lengths"].tolist(), strict=True
        ):
            histories[int(user)] = flat[offset : offset + int(length)]
            offset += int(length)

        return cls(
            model,
            store,
            internal_to_external_item={
                index: str(value) for index, value in enumerate(context["external_items"])
            },
            external_to_internal_user={
                str(user): int(internal)
                for user, internal in zip(
                    context["external_users"], context["internal_users"], strict=True
                )
            },
            item_tags=context["item_tags"],
            histories=histories,
            warm_items=context["warm_items"],
            device=device,
            mapping_checksum=metadata.get("mapping_checksum", ""),
            dataset_identity=metadata.get("dataset_identity", {}),
        )


__all__ = [
    "DEFAULT_MAXIMUM_SEARCH_MULTIPLIER",
    "DEFAULT_OVERSAMPLING_FACTOR",
    "EMPTY_SLOT",
    "SOURCE_NAME",
    "UNKNOWN_ITEM_SCORE",
    "TwoTowerRetriever",
]
