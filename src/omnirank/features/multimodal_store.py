"""Memory-mapped access to the aligned PixelRec content vectors.

The two aligned matrices are 69,347 x 1024 float32 -- about 284 MB each, so
both would fit in memory here. They are memory-mapped anyway, for two reasons
that outlast this catalogue size: the store is read in batches during training,
where mapping avoids holding a second copy per worker, and the same code has to
keep working when the catalogue is ten times larger without anyone revisiting
this decision.

**Identity is checked, not assumed.** A feature matrix is a list of vectors with
no intrinsic connection to any item id. Loaded against a different mapping it
does not fail -- it silently describes the wrong items, and every downstream
recommendation is confidently wrong in a way no metric reveals. So the manifest
carries the mapping checksum, the feature version and the dimensions, and
:meth:`MultimodalFeatureStore.require_compatible` refuses a mismatch. This is
ADR-006 applied to features rather than to indexes.

**Absence is a state, not an error.** An item may have text, image, both or
neither. The store returns explicit masks; it never substitutes a zero vector
for a missing modality without saying so, and never borrows another item's.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from omnirank.core.exceptions import ArtifactValidationError, DataError, DataSourceError
from omnirank.core.logging import get_logger

logger = get_logger(__name__)

MANIFEST_FILENAME: Final = "multimodal_feature_manifest.json"
MASK_FILENAME: Final = "modality_mask.parquet"
TEXT: Final = "text"
IMAGE: Final = "image"
MODALITIES: Final = (TEXT, IMAGE)


@dataclass(frozen=True, slots=True)
class ItemFeatures:
    """One item's content vectors and what is actually present."""

    internal_item_id: int
    text: np.ndarray | None
    image: np.ndarray | None

    @property
    def has_text(self) -> bool:
        """Whether a real text vector is present."""
        return self.text is not None

    @property
    def has_image(self) -> bool:
        """Whether a real image vector is present."""
        return self.image is not None

    @property
    def has_any(self) -> bool:
        """Whether the item can be represented from content at all."""
        return self.has_text or self.has_image


@dataclass(frozen=True, slots=True)
class ItemFeatureBatch:
    """A batch of content vectors with per-item, per-modality masks.

    Missing rows are zero-filled *and* flagged. The zeros are a placeholder the
    model must gate on the mask -- a model that reads the vectors and ignores
    the masks is training on fabricated content, which is the failure this
    structure exists to make impossible to reach by accident.
    """

    internal_item_ids: np.ndarray
    text: np.ndarray
    image: np.ndarray
    text_mask: np.ndarray
    image_mask: np.ndarray

    def __len__(self) -> int:
        return int(self.internal_item_ids.size)

    @property
    def coverage(self) -> dict[str, float]:
        """Fraction of the batch carrying each modality."""
        rows = max(len(self), 1)
        return {
            "text": float(self.text_mask.sum()) / rows,
            "image": float(self.image_mask.sum()) / rows,
            "both": float((self.text_mask & self.image_mask).sum()) / rows,
            "neither": float((~self.text_mask & ~self.image_mask).sum()) / rows,
        }


def _checksum(path: Path, *, block_bytes: int = 8 << 20) -> str:
    """SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


class MultimodalFeatureStore:
    """Batch-oriented, memory-mapped reader over the aligned feature matrices."""

    def __init__(
        self,
        features_dir: Path | str,
        *,
        verify_checksums: bool = False,
        modalities: tuple[str, ...] = MODALITIES,
    ) -> None:
        self.root = Path(features_dir)
        manifest_path = self.root / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise DataSourceError(
                "No multimodal feature manifest. Run "
                "`python scripts/prepare_multimodal_features.py` first.",
                expected=str(manifest_path),
            )
        try:
            self.manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError(
                "Feature manifest is not valid JSON", path=str(manifest_path)
            ) from exc

        self._matrices: dict[str, np.ndarray] = {}
        self._masks: dict[str, np.ndarray] = {}
        self._dimensions: dict[str, int] = {}

        declared = self.manifest.get("modalities", {})
        for modality in modalities:
            block = declared.get(modality)
            if not block or not block.get("available"):
                logger.info("features.modality_absent", modality=modality)
                continue
            matrix_file = block.get("matrix_file")
            if not matrix_file:
                continue
            path = self.root / matrix_file
            if not path.is_file():
                raise DataSourceError(
                    "Feature manifest names a matrix that does not exist",
                    modality=modality,
                    expected=str(path),
                )
            if verify_checksums and (recorded := block.get("matrix_sha256")):
                actual = _checksum(path)
                if actual != recorded:
                    raise ArtifactValidationError(
                        "Feature matrix checksum does not match the manifest. The "
                        "file changed after alignment; re-run feature preparation.",
                        modality=modality,
                        expected=recorded,
                        found=actual,
                    )
            # mmap_mode="r" keeps the array on disk and read-only, so a bug that
            # writes to it fails loudly instead of corrupting the store.
            self._matrices[modality] = np.load(path, mmap_mode="r")
            self._dimensions[modality] = int(self._matrices[modality].shape[1])

        self._load_masks()
        self._validate_shapes()
        logger.info(
            "features.store_loaded",
            root=str(self.root),
            modalities=sorted(self._matrices),
            items=self.catalogue_size,
            dimensions=self._dimensions,
            feature_version=self.feature_version,
        )

    def _load_masks(self) -> None:
        """Read the per-item availability flags."""
        mask_path = self.root / MASK_FILENAME
        if not mask_path.is_file():
            raise DataSourceError("Modality mask table is missing", expected=str(mask_path))
        frame = pd.read_parquet(mask_path).sort_values("internal_item_id")
        self._catalogue_size = len(frame)
        for modality in MODALITIES:
            column = f"has_{modality}_feature"
            values = (
                frame[column].to_numpy(dtype=bool)
                if column in frame.columns
                else np.zeros(self._catalogue_size, dtype=bool)
            )
            # A mask claiming a modality the store did not load would let the
            # model read zero rows as though they were real vectors.
            self._masks[modality] = values & (modality in self._matrices)

    def _validate_shapes(self) -> None:
        """Assert every loaded matrix covers the whole catalogue."""
        for modality, matrix in self._matrices.items():
            if matrix.shape[0] != self._catalogue_size:
                raise ArtifactValidationError(
                    "Feature matrix does not cover the catalogue",
                    modality=modality,
                    matrix_rows=int(matrix.shape[0]),
                    catalogue_items=self._catalogue_size,
                )
            if matrix.dtype != np.float32:
                raise ArtifactValidationError(
                    "Feature matrix is not float32",
                    modality=modality,
                    dtype=str(matrix.dtype),
                )

    # -- identity ----------------------------------------------------------- #
    @property
    def feature_version(self) -> str:
        """Schema version of the stored features."""
        return str(self.manifest.get("feature_version", ""))

    @property
    def mapping_checksum(self) -> str:
        """Item mapping these features were aligned against."""
        return str(self.manifest.get("item_mapping_checksum", ""))

    @property
    def catalogue_size(self) -> int:
        """Number of catalogue items the store covers."""
        return self._catalogue_size

    @property
    def available_modalities(self) -> tuple[str, ...]:
        """Modalities actually loaded, in a stable order."""
        return tuple(sorted(self._matrices))

    def dimension(self, modality: str) -> int:
        """Vector width for one modality.

        Raises:
            DataError: The modality is not loaded.
        """
        if modality not in self._dimensions:
            raise DataError(
                "Modality is not available in this store",
                modality=modality,
                available=list(self.available_modalities),
            )
        return self._dimensions[modality]

    def manifest_checksum(self) -> str:
        """Stable hash of the manifest's identity-bearing fields.

        Covers what makes two stores interchangeable, and deliberately not the
        timestamp or runtime measurements -- re-running alignment on the same
        inputs must produce the same identity.
        """
        payload = {
            "feature_version": self.feature_version,
            "item_mapping_checksum": self.mapping_checksum,
            "catalogue_items": self._catalogue_size,
            "modalities": {
                modality: {
                    "dimension": self._dimensions.get(modality),
                    "source_sha256": self.manifest["modalities"][modality].get("source_sha256"),
                    "rows_matched": self.manifest["modalities"][modality].get("rows_matched"),
                }
                for modality in sorted(self._matrices)
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def require_compatible(self, *, mapping_checksum: str, feature_version: str) -> None:
        """Assert this store belongs with the given mapping and schema.

        Raises:
            ArtifactValidationError: Either identity differs. A mismatched store
                does not fail at read time; it returns another item's content.
        """
        problems: list[str] = []
        if mapping_checksum and mapping_checksum != self.mapping_checksum:
            problems.append("item_mapping_checksum differs")
        if feature_version and feature_version != self.feature_version:
            problems.append(f"feature_version {self.feature_version!r} != {feature_version!r}")
        if problems:
            raise ArtifactValidationError(
                "Feature store is incompatible with this model. Its vectors "
                "describe different items than the model was fitted against.",
                problems=problems,
            )

    # -- reads -------------------------------------------------------------- #
    def _check_ids(self, item_ids: np.ndarray) -> None:
        """Reject ids outside the catalogue before they index a matrix."""
        if item_ids.size and (item_ids.min() < 0 or item_ids.max() >= self._catalogue_size):
            raise DataError(
                "Item id outside the catalogue",
                minimum=int(item_ids.min()),
                maximum=int(item_ids.max()),
                catalogue_items=self._catalogue_size,
            )

    def has_modality(self, modality: str) -> np.ndarray:
        """Per-item availability flags for one modality."""
        if modality not in self._masks:
            raise DataError("Unknown modality", modality=modality, available=list(MODALITIES))
        return self._masks[modality]

    @property
    def content_representable(self) -> np.ndarray:
        """Items carrying at least one modality.

        The catalogue a content model can encode. An item with neither modality
        cannot be represented from content, and saying so is what keeps a cold
        item that is genuinely unreachable from being counted as a retrieval
        failure.
        """
        combined: np.ndarray = self._masks[TEXT] | self._masks[IMAGE]
        return combined

    def get_item(self, internal_item_id: int) -> ItemFeatures:
        """Fetch one item's vectors, omitting modalities it does not have."""
        ids = np.asarray([internal_item_id], dtype="int64")
        self._check_ids(ids)
        return ItemFeatures(
            internal_item_id=int(internal_item_id),
            text=(
                np.asarray(self._matrices[TEXT][internal_item_id], dtype="float32")
                if self._masks[TEXT][internal_item_id]
                else None
            ),
            image=(
                np.asarray(self._matrices[IMAGE][internal_item_id], dtype="float32")
                if self._masks[IMAGE][internal_item_id]
                else None
            ),
        )

    def get_batch(self, internal_item_ids: np.ndarray) -> ItemFeatureBatch:
        """Fetch a batch, preserving the requested order.

        Order is preserved rather than sorted: callers align these rows against
        their own label tensors positionally, so reordering here would silently
        pair the wrong content with the wrong item.
        """
        ids = np.asarray(internal_item_ids, dtype="int64").ravel()
        self._check_ids(ids)
        return ItemFeatureBatch(
            internal_item_ids=ids,
            text=self._read(TEXT, ids),
            image=self._read(IMAGE, ids),
            text_mask=self._masks[TEXT][ids] if ids.size else np.zeros(0, dtype=bool),
            image_mask=self._masks[IMAGE][ids] if ids.size else np.zeros(0, dtype=bool),
        )

    def _read(self, modality: str, ids: np.ndarray) -> np.ndarray:
        """Read rows for one modality, zero-filling absent ones."""
        width = self._dimensions.get(modality, 0)
        if modality not in self._matrices or not ids.size:
            return np.zeros((ids.size, width), dtype="float32")
        rows = np.asarray(self._matrices[modality][ids], dtype="float32")
        # Zero the rows the mask says are absent, so a caller that ignores the
        # mask gets a neutral vector rather than whatever alignment left behind.
        rows[~self._masks[modality][ids]] = 0.0
        return rows

    def describe(self) -> dict[str, Any]:
        """Report-ready description of what the store holds."""
        return {
            "feature_version": self.feature_version,
            "item_mapping_checksum": self.mapping_checksum,
            "manifest_checksum": self.manifest_checksum(),
            "catalogue_items": self._catalogue_size,
            "modalities": {
                modality: {
                    "dimension": self._dimensions[modality],
                    "items_with_feature": int(self._masks[modality].sum()),
                    "coverage": float(self._masks[modality].mean()),
                }
                for modality in sorted(self._matrices)
            },
            "items_with_any_modality": int(self.content_representable.sum()),
            "items_with_no_modality": int((~self.content_representable).sum()),
            "storage": "memory_map",
            "dtype": "float32",
        }


__all__ = [
    "IMAGE",
    "MANIFEST_FILENAME",
    "MODALITIES",
    "TEXT",
    "ItemFeatureBatch",
    "ItemFeatures",
    "MultimodalFeatureStore",
]
