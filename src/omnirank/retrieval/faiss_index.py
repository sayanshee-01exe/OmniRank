"""FAISS vector index - component 15.

Implements the Phase 1 :class:`~omnirank.retrieval.base.VectorIndex` protocol.

**Exact search is the correctness reference.** `IndexFlatIP` computes the same
inner products a brute-force matmul would, so its results must match PyTorch
exactly up to floating-point tie order - and a test asserts that on both fixtures
and sampled real users. Nothing approximate is used for a reported model-quality
metric until that equivalence is established, because an approximate index that
quietly drops the true nearest neighbour looks exactly like a worse model.

At ~69,000 items an exact index may simply be sufficient. Approximate backends
(HNSW, IVF) are available behind the same interface, and the decision between
them is made from measurement rather than from which sounds more production-like.

**No normalisation is applied.** LightGCN and SASRec score by dot product, and
L2-normalising the vectors would silently convert the metric to cosine - a
different ranking, arrived at without anyone choosing it. Cosine is available by
asking for it explicitly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Self

import numpy as np

from omnirank.core.exceptions import ArtifactValidationError, VectorIndexError
from omnirank.core.logging import get_logger

logger = get_logger(__name__)

#: Bumped when the build procedure changes in a way that makes an old index
#: incomparable. Checked against artifact metadata (ADR-006).
INDEX_FORMAT_VERSION: Final = 1

FLAT_IP: Final = "flat_ip"
FLAT_L2: Final = "flat_l2"
HNSW: Final = "hnsw"
IVF_FLAT: Final = "ivf_flat"
INDEX_TYPES: Final = (FLAT_IP, FLAT_L2, HNSW, IVF_FLAT)

INNER_PRODUCT: Final = "inner_product"
L2: Final = "l2"

#: Padding for positions with no result, so callers get rectangular output and
#: can filter on the sentinel rather than on list length.
EMPTY_SLOT: Final = -1

_INDEX_FILENAME: Final = "index.faiss"
_METADATA_FILENAME: Final = "index_metadata.json"


def _require_faiss() -> Any:
    """Import faiss, or explain how to get it.

    **OpenMP coexistence.** ``faiss-cpu`` and ``torch`` each bundle their own
    copy of ``libomp.dylib``. On macOS the LLVM OpenMP runtime aborts the
    process when the second copy initialises -- so any code path that builds an
    index from a torch model's embeddings crashes, which is every path that
    matters here. Import order does not help; both copies load either way.

    ``KMP_DUPLICATE_LIB_OK`` permits the second initialisation. LLVM documents
    it as unsafe and warns it "may silently produce incorrect results", which
    would normally rule it out. Two things make it acceptable here rather than a
    hopeful workaround:

    * FAISS is pinned to a single OpenMP thread below, so the duplicated
      runtimes never contend over a shared thread pool -- the mechanism behind
      the warning;
    * every flat index is checked against exact brute force in the test suite,
      in both set and order. Silent corruption is exactly what that test fails
      on, so the claim is verified on each run rather than assumed.

    ``setdefault`` so an operator who has set it deliberately is not overridden.

    Raises:
        VectorIndexError: The retrieval extra is not installed.
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise VectorIndexError(
            "FAISS is not installed. Install the retrieval extra: "
            "`uv pip install -e '.[retrieval,dev]'`.",
            reason=str(exc),
        ) from exc
    faiss.omp_set_num_threads(1)
    return faiss


@dataclass(frozen=True, slots=True)
class IndexMetadata:
    """Everything needed to decide whether an index may be used with a model."""

    index_type: str
    index_version: int
    dimension: int
    metric: str
    num_vectors: int
    model_name: str
    model_version: str
    embedding_checksum: str
    item_mapping_checksum: str
    build_timestamp: str
    faiss_version: str
    format_version: int = INDEX_FORMAT_VERSION
    build_parameters: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Manifest-ready payload."""
        return {
            "index_type": self.index_type,
            "index_version": self.index_version,
            "dimension": self.dimension,
            "metric": self.metric,
            "num_vectors": self.num_vectors,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "embedding_checksum": self.embedding_checksum,
            "item_mapping_checksum": self.item_mapping_checksum,
            "build_timestamp": self.build_timestamp,
            "faiss_version": self.faiss_version,
            "format_version": self.format_version,
            "build_parameters": self.build_parameters or {},
        }


def embedding_checksum(embeddings: np.ndarray) -> str:
    """Content hash of an embedding matrix.

    Recorded in the index metadata and checked at load, so an index built from
    one model's embeddings can never be paired with another's - a mismatch that
    otherwise produces confident nonsense rather than an error.
    """
    import hashlib

    contiguous = np.ascontiguousarray(embeddings, dtype="float32")
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _validate_matrix(embeddings: np.ndarray, *, name: str = "embeddings") -> np.ndarray:
    """Check an embedding matrix and return it as contiguous float32.

    Raises:
        VectorIndexError: Wrong rank, empty, or containing NaN/inf. A single NaN
            makes every comparison against that row false, so the row silently
            never gets retrieved.
    """
    array = np.asarray(embeddings)
    if array.ndim != 2:
        raise VectorIndexError(
            f"{name} must be a 2-D (rows, dimension) matrix",
            found_shape=list(array.shape),
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise VectorIndexError(f"{name} must not be empty", found_shape=list(array.shape))
    contiguous = np.ascontiguousarray(array, dtype="float32")
    if not np.isfinite(contiguous).all():
        nan_rows = int(np.isnan(contiguous).any(axis=1).sum())
        inf_rows = int(np.isinf(contiguous).any(axis=1).sum())
        raise VectorIndexError(
            f"{name} contain non-finite values. A NaN row is never retrieved and "
            "fails silently rather than erroring at query time.",
            nan_rows=nan_rows,
            inf_rows=inf_rows,
        )
    return contiguous


class FaissVectorIndex:
    """A FAISS-backed vector index over item embeddings.

    Row order is the dense item index from the Phase 2 mapping. The index does
    not know about external string ids, so it cannot drift out of sync with the
    mapping in a way that silently resolves to the wrong item.
    """

    def __init__(
        self,
        *,
        index_type: str = FLAT_IP,
        metric: str = INNER_PRODUCT,
        index_version: int = 1,
        oversampling_factor: int = 2,
        maximum_search_multiplier: int = 16,
        build_parameters: dict[str, Any] | None = None,
    ) -> None:
        if index_type not in INDEX_TYPES:
            raise VectorIndexError(
                "Unknown index type", index_type=index_type, available=list(INDEX_TYPES)
            )
        if metric not in (INNER_PRODUCT, L2):
            raise VectorIndexError("Unknown metric", metric=metric)
        if oversampling_factor < 1:
            raise VectorIndexError(
                "Oversampling factor must be >= 1", oversampling_factor=oversampling_factor
            )
        if maximum_search_multiplier < oversampling_factor:
            raise VectorIndexError(
                "Maximum search multiplier must be at least the oversampling factor",
                oversampling_factor=oversampling_factor,
                maximum_search_multiplier=maximum_search_multiplier,
            )
        self.index_type = index_type
        self.metric = metric
        self._index_version = index_version
        self.oversampling_factor = oversampling_factor
        self.maximum_search_multiplier = maximum_search_multiplier
        self.build_parameters = dict(build_parameters or {})
        self._index: Any = None
        self._dimension = 0
        self._num_vectors = 0
        self._embedding_checksum = ""
        self._metadata: IndexMetadata | None = None

    # -- protocol ----------------------------------------------------------- #
    @property
    def dimension(self) -> int:
        """Embedding dimensionality this index was built for."""
        return self._dimension

    @property
    def index_version(self) -> int:
        """Build version, checked against artifact metadata (ADR-006)."""
        return self._index_version

    @property
    def num_vectors(self) -> int:
        """Vectors currently indexed."""
        return self._num_vectors

    @property
    def metadata(self) -> IndexMetadata | None:
        """Index metadata, once built or loaded."""
        return self._metadata

    def build(self, embeddings: Any, *, metric: str = INNER_PRODUCT) -> None:
        """Build the index from a ``(num_items, dimension)`` matrix.

        Embeddings are **not** normalised. LightGCN and SASRec score by dot
        product; normalising would silently convert the metric to cosine.
        """
        faiss = _require_faiss()
        matrix = _validate_matrix(embeddings)
        self.metric = metric
        rows, dimension = matrix.shape

        if self.index_type == FLAT_IP:
            index = faiss.IndexFlatIP(dimension)
        elif self.index_type == FLAT_L2:
            index = faiss.IndexFlatL2(dimension)
        elif self.index_type == HNSW:
            neighbours = int(self.build_parameters.get("hnsw_m", 32))
            index = faiss.IndexHNSWFlat(dimension, neighbours, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = int(self.build_parameters.get("ef_construction", 200))
        else:
            lists = int(self.build_parameters.get("nlist", max(1, int(np.sqrt(rows)))))
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, lists, faiss.METRIC_INNER_PRODUCT)
            index.train(matrix)

        index.add(matrix)
        self._index = index
        self._dimension = dimension
        self._num_vectors = rows
        self._embedding_checksum = embedding_checksum(matrix)
        logger.info(
            "faiss.built",
            index_type=self.index_type,
            metric=metric,
            vectors=rows,
            dimension=dimension,
        )

    def search(self, query: Any, k: int) -> tuple[list[list[int]], list[list[float]]]:
        """Return the ``k`` nearest item indices and their scores.

        Output is rectangular: rows with fewer than ``k`` results are padded with
        :data:`EMPTY_SLOT` and ``-inf``.

        Raises:
            VectorIndexError: The index is unbuilt, ``k`` is invalid, or the
                query dimension does not match.
        """
        if self._index is None:
            raise VectorIndexError("Index has not been built or loaded")
        if k < 1:
            raise VectorIndexError("k must be >= 1", k=k)
        matrix = _validate_matrix(query, name="query")
        if matrix.shape[1] != self._dimension:
            raise VectorIndexError(
                "Query dimension does not match the index",
                query_dimension=int(matrix.shape[1]),
                index_dimension=self._dimension,
            )

        take = min(k, self._num_vectors)
        scores, indices = self._index.search(matrix, take)
        out_indices = indices.astype("int64")
        out_scores = scores.astype("float64")
        # FAISS already pads short results with -1; make the score sentinel match.
        out_scores = np.where(out_indices == EMPTY_SLOT, float("-inf"), out_scores)

        if take < k:
            pad = k - take
            out_indices = np.concatenate(
                [out_indices, np.full((matrix.shape[0], pad), EMPTY_SLOT, dtype="int64")], axis=1
            )
            out_scores = np.concatenate(
                [out_scores, np.full((matrix.shape[0], pad), float("-inf"))], axis=1
            )
        return out_indices.tolist(), out_scores.tolist()

    def search_excluding(
        self, query: Any, k: int, *, excluded: list[set[int]]
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Search, removing each row's excluded items, with bounded over-retrieval.

        FAISS does not know a user's seen items, so the only way to return ``k``
        *unseen* results is to retrieve more and filter. The buffer grows
        geometrically and is capped by ``maximum_search_multiplier``, so a user
        who has seen most of the catalogue terminates with a short list rather
        than looping.

        Args:
            query: ``(rows, dimension)`` query matrix.
            k: Results wanted per row after filtering.
            excluded: Per-row item indices to remove. Must align with ``query``.

        Raises:
            VectorIndexError: ``excluded`` does not align with the query rows.
        """
        matrix = _validate_matrix(query, name="query")
        if len(excluded) != matrix.shape[0]:
            raise VectorIndexError(
                "Excluded-item sets must align with query rows",
                rows=int(matrix.shape[0]),
                excluded=len(excluded),
            )

        rows = matrix.shape[0]
        results: list[list[int]] = [[] for _ in range(rows)]
        scores: list[list[float]] = [[] for _ in range(rows)]
        pending = list(range(rows))
        multiplier = self.oversampling_factor

        while pending and multiplier <= self.maximum_search_multiplier:
            depth = min(self._num_vectors, max(k * multiplier, k + 1))
            found_indices, found_scores = self.search(matrix[pending], depth)
            still_pending: list[int] = []
            for position, row in enumerate(pending):
                keep_items: list[int] = []
                keep_scores: list[float] = []
                blocked = excluded[row]
                row_items = found_indices[position]
                row_scores = found_scores[position]
                for item, score in zip(row_items, row_scores, strict=True):
                    if item == EMPTY_SLOT or item in blocked:
                        continue
                    keep_items.append(item)
                    keep_scores.append(score)
                    if len(keep_items) == k:
                        break
                results[row] = keep_items
                scores[row] = keep_scores
                # Only retry when a deeper search could actually help.
                if len(keep_items) < k and depth < self._num_vectors:
                    still_pending.append(row)
            pending = still_pending
            multiplier *= 2

        if pending:
            logger.debug(
                "faiss.oversampling_exhausted",
                rows=len(pending),
                detail=(
                    "Some rows returned fewer than k unseen items after the "
                    "maximum search multiplier. Their histories cover most of "
                    "the catalogue; a short list is the correct answer."
                ),
            )

        padded_items = [row + [EMPTY_SLOT] * (k - len(row)) for row in results]
        padded_scores = [row + [float("-inf")] * (k - len(row)) for row in scores]
        return padded_items, padded_scores

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        """Persist the index and its metadata to a directory."""
        if self._index is None:
            raise VectorIndexError("Cannot save an index that has not been built")
        faiss = _require_faiss()
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(target / _INDEX_FILENAME))
        if self._metadata is not None:
            (target / _METADATA_FILENAME).write_text(
                json.dumps(self._metadata.to_dict(), indent=2, sort_keys=True)
            )
        logger.info("faiss.saved", path=str(target), vectors=self._num_vectors)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Restore a persisted index.

        Raises:
            ArtifactValidationError: Files missing, or the metadata was written
                by an unsupported format version.
        """
        faiss = _require_faiss()
        source = Path(path)
        index_path = source / _INDEX_FILENAME
        if not index_path.is_file():
            raise ArtifactValidationError("FAISS index file not found", path=str(index_path))

        metadata: IndexMetadata | None = None
        metadata_path = source / _METADATA_FILENAME
        if metadata_path.is_file():
            try:
                payload = json.loads(metadata_path.read_text())
            except json.JSONDecodeError as exc:
                raise ArtifactValidationError(
                    "Index metadata is not valid JSON", path=str(metadata_path)
                ) from exc
            if payload.get("format_version") != INDEX_FORMAT_VERSION:
                raise ArtifactValidationError(
                    "Unsupported index format version",
                    expected=INDEX_FORMAT_VERSION,
                    found=payload.get("format_version"),
                )
            payload.pop("build_parameters", None)
            metadata = IndexMetadata(**payload, build_parameters={})

        index = cls(
            index_type=metadata.index_type if metadata else FLAT_IP,
            metric=metadata.metric if metadata else INNER_PRODUCT,
            index_version=metadata.index_version if metadata else 1,
        )
        try:
            index._index = faiss.read_index(str(index_path))
        except Exception as exc:  # any read failure is the same error
            raise ArtifactValidationError(
                "FAISS index could not be read; it may be corrupted",
                path=str(index_path),
                reason=str(exc)[:200],
            ) from exc
        index._dimension = index._index.d
        index._num_vectors = index._index.ntotal
        index._metadata = metadata
        if metadata is not None:
            index._embedding_checksum = metadata.embedding_checksum
        logger.info("faiss.loaded", path=str(source), vectors=index._num_vectors)
        return index

    def attach_metadata(
        self,
        *,
        model_name: str,
        model_version: str,
        item_mapping_checksum: str,
        build_timestamp: str,
    ) -> IndexMetadata:
        """Record what this index was built from, for compatibility checking."""
        faiss = _require_faiss()
        self._metadata = IndexMetadata(
            index_type=self.index_type,
            index_version=self._index_version,
            dimension=self._dimension,
            metric=self.metric,
            num_vectors=self._num_vectors,
            model_name=model_name,
            model_version=model_version,
            embedding_checksum=self._embedding_checksum,
            item_mapping_checksum=item_mapping_checksum,
            build_timestamp=build_timestamp,
            faiss_version=getattr(faiss, "__version__", "unknown"),
            build_parameters=self.build_parameters,
        )
        return self._metadata

    def require_compatible(
        self,
        *,
        model_name: str,
        model_version: str,
        item_mapping_checksum: str,
    ) -> None:
        """Assert this index belongs with the given model and mapping.

        Raises:
            ArtifactValidationError: Any mismatch. An index paired with the wrong
                model returns confident nonsense rather than an error, so this
                is a hard failure.
        """
        if self._metadata is None:
            raise ArtifactValidationError(
                "Index carries no metadata, so compatibility cannot be checked"
            )
        problems: list[str] = []
        if self._metadata.model_name != model_name:
            problems.append(f"model_name {self._metadata.model_name!r} != {model_name!r}")
        if self._metadata.model_version != model_version:
            problems.append(f"model_version {self._metadata.model_version!r} != {model_version!r}")
        if self._metadata.item_mapping_checksum != item_mapping_checksum:
            problems.append("item_mapping_checksum differs")
        if problems:
            raise ArtifactValidationError(
                "Index is incompatible with this model. Rebuild it from the "
                "model's current embeddings (ADR-006).",
                problems=problems,
            )


def brute_force_top_k(
    embeddings: np.ndarray, query: np.ndarray, k: int, *, metric: str = INNER_PRODUCT
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by direct computation - the reference FAISS is checked against.

    Deliberately naive. Its only job is to be obviously correct, so that any
    disagreement with the index is the index's fault.

    ``metric`` must match the metric the index was built with. Both are
    supported because building an index under one metric and querying as though
    it were the other is a silent failure: every query still returns k
    neighbours with plausible scores, just the wrong ones.
    """
    matrix = _validate_matrix(embeddings)
    queries = _validate_matrix(query, name="query")
    if metric == INNER_PRODUCT:
        # Higher is better, so negate to get descending order from argsort.
        ranked = -(queries @ matrix.T)
    elif metric == L2:
        # Squared euclidean, which is what FAISS's L2 index returns.
        ranked = (
            (queries**2).sum(axis=1)[:, None]
            - 2 * (queries @ matrix.T)
            + (matrix**2).sum(axis=1)[None, :]
        )
    else:
        raise VectorIndexError("Unknown metric", metric=metric, available=[INNER_PRODUCT, L2])
    take = min(k, matrix.shape[0])
    # Ties fall to the lower index, matching FAISS's behaviour.
    order = np.argsort(ranked, axis=1, kind="stable")[:, :take]
    top_scores = np.take_along_axis(ranked, order, axis=1)
    if metric == INNER_PRODUCT:
        top_scores = -top_scores
    return order.astype("int64"), top_scores.astype("float64")


__all__ = [
    "EMPTY_SLOT",
    "FLAT_IP",
    "FLAT_L2",
    "HNSW",
    "INDEX_FORMAT_VERSION",
    "INDEX_TYPES",
    "INNER_PRODUCT",
    "IVF_FLAT",
    "L2",
    "FaissVectorIndex",
    "IndexMetadata",
    "brute_force_top_k",
    "embedding_checksum",
]
