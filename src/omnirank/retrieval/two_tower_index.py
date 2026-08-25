"""Exact FAISS index over two-tower item embeddings.

Thin on purpose. The generic :class:`~omnirank.retrieval.faiss_index.FaissVectorIndex`
already handles building, exact search, bounded exclusion search, persistence
and brute-force verification; duplicating it for one model would mean two
implementations to keep exact.

What this module adds is the **identity a two-tower index needs and a
collaborative one does not**. A LightGCN index is wrong if it is paired with the
wrong model or mapping. A two-tower index has a third way to be wrong: the
embeddings derive from a feature store, so a store with different vectors --
same items, same mapping, different content -- produces a different index that
nothing downstream would notice. Feature version and feature-manifest checksum
therefore travel with it.

It also records **warm and cold counts**, because "the index contains cold
items" is the claim Phase 5 rests on, and an index that quietly contained none
would still answer every query.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from omnirank.core.exceptions import ArtifactValidationError, DataError
from omnirank.core.logging import get_logger
from omnirank.models.two_tower.catalogue import RetrievalCatalogue
from omnirank.retrieval.faiss_index import (
    FLAT_IP,
    INNER_PRODUCT,
    FaissVectorIndex,
    brute_force_top_k,
    embedding_checksum,
)

logger = get_logger(__name__)

EMBEDDING_FILENAME: Final = "item_embeddings.npy"
EMBEDDING_MANIFEST: Final = "embedding_manifest.json"
INDEX_SUBDIR: Final = "index"


def write_item_embeddings(
    directory: Path | str,
    embeddings: np.ndarray,
    catalogue: RetrievalCatalogue,
    *,
    model_version: str,
    model_checksum: str,
    mapping_checksum: str,
    feature_version: str,
    feature_manifest_checksum: str,
    normalization: str,
) -> dict[str, Any]:
    """Write the embedding matrix, item table and manifest.

    Raises:
        DataError: The matrix does not align with the catalogue, is not
            float32, or contains non-finite values.
    """
    if embeddings.shape[0] != len(catalogue):
        raise DataError(
            "Embedding rows do not match the catalogue. Every row would then "
            "describe a different item than the table claims.",
            rows=int(embeddings.shape[0]),
            catalogue=len(catalogue),
        )
    if embeddings.dtype != np.float32:
        raise DataError("Item embeddings must be float32", dtype=str(embeddings.dtype))
    if not np.isfinite(embeddings).all():
        raise DataError("Item embeddings contain non-finite values")

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    # Saved with .npy so it can be memory-mapped back rather than read whole.
    np.save(target / EMBEDDING_FILENAME, embeddings)

    catalogue_manifest = catalogue.save(
        target,
        model_version=model_version,
        mapping_checksum=mapping_checksum,
        feature_version=feature_version,
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "embedding_file": EMBEDDING_FILENAME,
        "items": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "dtype": "float32",
        "storage": "memory_map",
        "normalization": normalization,
        "warm_items": catalogue.warm_count,
        "cold_items": catalogue.cold_count,
        "excluded_items": catalogue.excluded_count,
        "embedding_checksum": embedding_checksum(embeddings),
        "catalogue_checksum": catalogue.checksum(),
        "model_version": model_version,
        "model_checksum": model_checksum,
        "mapping_checksum": mapping_checksum,
        "feature_version": feature_version,
        "feature_manifest_checksum": feature_manifest_checksum,
    }
    (target / EMBEDDING_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    logger.info(
        "two_tower.embeddings_written",
        path=str(target),
        items=manifest["items"],
        dimension=manifest["dimension"],
        warm=catalogue.warm_count,
        cold=catalogue.cold_count,
        megabytes=round(embeddings.nbytes / 1e6, 1),
    )
    return {**manifest, "catalogue": catalogue_manifest}


def load_item_embeddings(
    directory: Path | str, *, verify: bool = True
) -> tuple[np.ndarray, RetrievalCatalogue, dict[str, Any]]:
    """Memory-map a saved embedding matrix beside its catalogue.

    Raises:
        ArtifactValidationError: Files are missing or the checksum disagrees.
    """
    source = Path(directory)
    matrix_path, manifest_path = source / EMBEDDING_FILENAME, source / EMBEDDING_MANIFEST
    for path in (matrix_path, manifest_path):
        if not path.is_file():
            raise ArtifactValidationError(
                "Item embedding artifact is incomplete", missing=str(path)
            )
    manifest = json.loads(manifest_path.read_text())
    embeddings = np.load(matrix_path, mmap_mode="r")
    catalogue, _ = RetrievalCatalogue.load(source)

    if verify:
        recorded = manifest.get("embedding_checksum")
        actual = embedding_checksum(np.asarray(embeddings))
        if recorded and recorded != actual:
            raise ArtifactValidationError(
                "Embedding checksum does not match its manifest; the matrix "
                "changed after it was written.",
                expected=recorded,
                found=actual,
            )
    return np.asarray(embeddings), catalogue, manifest


def build_two_tower_index(
    embeddings: np.ndarray,
    catalogue: RetrievalCatalogue,
    *,
    model_version: str,
    model_checksum: str,
    mapping_checksum: str,
    feature_version: str,
    feature_manifest_checksum: str,
    normalization: str,
    index_type: str = FLAT_IP,
) -> tuple[FaissVectorIndex, dict[str, Any]]:
    """Build an exact index and the metadata that keeps it usable.

    Inner product is the metric because the towers are L2-normalised, which
    makes a dot product a cosine similarity. Building under one convention and
    querying under the other returns confident nonsense, so the normalisation
    rule is recorded rather than assumed.
    """
    if normalization != "l2" and index_type == FLAT_IP:
        logger.warning(
            "two_tower.unnormalised_inner_product",
            normalization=normalization,
            detail=(
                "Inner product over unnormalised vectors ranks by magnitude as "
                "much as by direction."
            ),
        )
    index = FaissVectorIndex(index_type=index_type, metric=INNER_PRODUCT)
    index.build(embeddings, metric=INNER_PRODUCT)

    metadata = {
        "index_type": index_type,
        "metric": INNER_PRODUCT,
        "dimension": int(embeddings.shape[1]),
        "number_of_items": int(embeddings.shape[0]),
        "warm_item_count": catalogue.warm_count,
        "cold_item_count": catalogue.cold_count,
        "model_version": model_version,
        "model_checksum": model_checksum,
        "embedding_checksum": embedding_checksum(embeddings),
        "mapping_checksum": mapping_checksum,
        "feature_version": feature_version,
        "feature_manifest_checksum": feature_manifest_checksum,
        "catalogue_checksum": catalogue.checksum(),
        "normalization_policy": normalization,
        "created_at": datetime.now(UTC).isoformat(),
    }
    built_at = str(metadata["created_at"])
    metadata["index_checksum"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()

    index.attach_metadata(
        model_name="two_tower",
        model_version=model_version,
        item_mapping_checksum=mapping_checksum,
        build_timestamp=built_at,
    )
    logger.info(
        "two_tower.index_built",
        items=metadata["number_of_items"],
        warm=catalogue.warm_count,
        cold=catalogue.cold_count,
        dimension=metadata["dimension"],
    )
    return index, metadata


#: Score gap below which two items are numerically indistinguishable in
#: float32. A 128-dimensional dot product accumulates rounding error of roughly
#: this magnitude, and FAISS and numpy do not sum in the same order.
TIE_TOLERANCE: Final = 1e-5


def verify_index_against_brute_force(
    index: FaissVectorIndex,
    embeddings: np.ndarray,
    queries: np.ndarray,
    k: int = 20,
    *,
    tie_tolerance: float = TIE_TOLERANCE,
) -> dict[str, Any]:
    """Check exact search reproduces brute force, allowing float32 ties.

    An index built with the wrong metric or over a transposed matrix still
    returns k neighbours with plausible scores for every query, and nothing
    raises. This is the only check that separates "fast" from "fast and wrong".

    **Bit-exact ordering is the wrong bar.** Two items whose true scores differ
    by less than float32 accumulation error can be ordered either way, and
    FAISS and numpy do not sum in the same order. Demanding identical orderings
    therefore fails on correct indexes at catalogue scale. So a disagreement is
    only counted against the index when the two scores are *distinguishable*:
    ``matches_brute_force`` means every position either agrees or is a genuine
    numerical tie.
    """
    found, scores = index.search(queries, k)
    expected, expected_scores = brute_force_top_k(embeddings, queries, k)
    expected_rows = expected.tolist()
    actual_scores = np.array(scores)

    exact_order = 0
    tie_explained = 0
    unexplained: list[dict[str, Any]] = []
    for row, (returned, reference) in enumerate(zip(found, expected_rows, strict=True)):
        if returned == reference:
            exact_order += 1
            tie_explained += 1
            continue
        # Every position where the two disagree must be a pair of scores too
        # close to order reliably. Anything else is the index being wrong.
        gaps = [
            abs(float(actual_scores[row][position]) - float(expected_scores[row][position]))
            for position in range(len(returned))
            if returned[position] != reference[position]
        ]
        if gaps and max(gaps) <= tie_tolerance:
            tie_explained += 1
        else:
            unexplained.append({"query": row, "max_gap": max(gaps) if gaps else None})

    overlap = float(
        np.mean(
            [
                len(set(row) & set(reference)) / max(len(reference), 1)
                for row, reference in zip(found, expected_rows, strict=True)
            ]
        )
    )
    result: dict[str, Any] = {
        "queries": len(found),
        "k": k,
        "exact_order_agreement": round(exact_order / max(len(found), 1), 6),
        "order_agreement_within_ties": round(tie_explained / max(len(found), 1), 6),
        "set_overlap": round(overlap, 6),
        "max_score_difference": float(np.abs(actual_scores - expected_scores).max()),
        "tie_tolerance": tie_tolerance,
        "unexplained_disagreements": len(unexplained),
        "matches_brute_force": not unexplained,
    }
    if unexplained:
        result["examples"] = unexplained[:5]
        logger.error(
            "two_tower.index_disagrees_with_brute_force",
            detail=(
                "Ordering differs at positions whose scores are distinguishable, "
                "so this is not float32 tie-breaking."
            ),
            **result,
        )
    else:
        logger.info("two_tower.index_verified", **result)
    return result


__all__ = [
    "EMBEDDING_FILENAME",
    "EMBEDDING_MANIFEST",
    "INDEX_SUBDIR",
    "TIE_TOLERANCE",
    "build_two_tower_index",
    "load_item_embeddings",
    "verify_index_against_brute_force",
    "write_item_embeddings",
]
