"""The retrieval catalogue: which items the two-tower model can return.

Every earlier retriever's catalogue is "items with a fitting interaction".
This one is different, and the difference is the phase's purpose:

    warm items  -- seen during fitting; content plus a learned id residual
    cold items  -- never seen, but carrying text or image content
    excluded    -- neither warm nor content-representable

An item with no content *and* no interaction history genuinely cannot be
represented by anything, and including it would put a meaningless vector in the
index that some query would eventually rank first. Excluding it is correct; not
saying how many were excluded is not.

The order is deterministic (ascending internal id) because the embedding matrix,
the index and the item table are all positional. A catalogue that reordered
between builds would silently pair every embedding with the wrong item.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from omnirank.core.exceptions import ArtifactValidationError, DataError
from omnirank.core.logging import get_logger

logger = get_logger(__name__)

CATALOGUE_FILENAME: Final = "catalogue_manifest.json"
ITEM_INDEX_FILENAME: Final = "item_index.parquet"

COLUMNS: Final = (
    "internal_item_id",
    "external_item_id",
    "warm_item_flag",
    "text_available",
    "image_available",
    "both_modalities_available",
    "content_representable",
)


@dataclass(frozen=True, slots=True)
class RetrievalCatalogue:
    """Items the model may return, and what each one can be built from."""

    items: pd.DataFrame
    warm_count: int
    cold_count: int
    excluded_count: int
    #: Internal ids of items left out, so the exclusion is auditable rather
    #: than merely counted.
    excluded_item_ids: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.items)

    @property
    def internal_ids(self) -> np.ndarray:
        """Catalogue item ids, in the order embeddings are written."""
        return self.items["internal_item_id"].to_numpy(dtype="int64")

    @property
    def warm_mask(self) -> np.ndarray:
        """Per-row warm flag, aligned with :attr:`internal_ids`."""
        return self.items["warm_item_flag"].to_numpy(dtype=bool)

    def checksum(self) -> str:
        """Content hash over the identity-bearing columns.

        Covers what makes two catalogues interchangeable -- which items, in
        which order, with which capabilities -- and nothing else.
        """
        frame = self.items.loc[:, list(COLUMNS)]
        digest = hashlib.sha256()
        digest.update(
            pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="uint64").tobytes()
        )
        return digest.hexdigest()

    def describe(self, **identity: Any) -> dict[str, Any]:
        """Manifest-ready description."""
        both = int(self.items["both_modalities_available"].sum())
        return {
            "catalogue_checksum": self.checksum(),
            "created_at": datetime.now(UTC).isoformat(),
            "items": len(self),
            "warm_items": self.warm_count,
            "cold_items": self.cold_count,
            "excluded_items": self.excluded_count,
            "excluded_reason": (
                "no content in any modality and no fitting interaction, so the "
                "item cannot be represented by content or by identity"
            ),
            "items_with_both_modalities": both,
            "items_with_text_only": int(
                (self.items["text_available"] & ~self.items["image_available"]).sum()
            ),
            "items_with_image_only": int(
                (~self.items["text_available"] & self.items["image_available"]).sum()
            ),
            "cold_items_with_both_modalities": int(
                (~self.items["warm_item_flag"] & self.items["both_modalities_available"]).sum()
            ),
            "ordering": "ascending internal_item_id",
            "cold_item_policy": (
                "content only; the item-id residual is gated to zero for any "
                "item with no fitting interaction"
            ),
            **identity,
        }

    def save(self, directory: Path | str, **identity: Any) -> dict[str, Any]:
        """Write the item table and its manifest."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.items.to_parquet(target / ITEM_INDEX_FILENAME, index=False)
        manifest = self.describe(**identity)
        (target / CATALOGUE_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
        )
        logger.info(
            "two_tower.catalogue_saved",
            path=str(target),
            **{
                key: manifest[key]
                for key in ("items", "warm_items", "cold_items", "excluded_items")
            },
        )
        return manifest

    @classmethod
    def load(cls, directory: Path | str) -> tuple[RetrievalCatalogue, dict[str, Any]]:
        """Read a saved catalogue and verify its checksum."""
        source = Path(directory)
        table, manifest_path = source / ITEM_INDEX_FILENAME, source / CATALOGUE_FILENAME
        for path in (table, manifest_path):
            if not path.is_file():
                raise ArtifactValidationError("Catalogue is incomplete", missing=str(path))
        manifest = json.loads(manifest_path.read_text())
        items = pd.read_parquet(table)
        catalogue = cls(
            items=items,
            warm_count=int(items["warm_item_flag"].sum()),
            cold_count=int((~items["warm_item_flag"]).sum()),
            excluded_count=int(manifest.get("excluded_items", 0)),
            excluded_item_ids=(),
        )
        recorded = manifest.get("catalogue_checksum")
        if recorded and catalogue.checksum() != recorded:
            raise ArtifactValidationError(
                "Catalogue checksum does not match its manifest. The item table "
                "changed after the embeddings were written, so every embedding "
                "row may now describe a different item.",
                expected=recorded,
                found=catalogue.checksum(),
            )
        return catalogue, manifest


def build_catalogue(
    *,
    warm_items: np.ndarray,
    text_available: np.ndarray,
    image_available: np.ndarray,
    internal_to_external: dict[int, str],
) -> RetrievalCatalogue:
    """Assemble the retrieval catalogue from warmth and content availability.

    Args:
        warm_items: Per-item boolean, indexed by internal id.
        text_available: Per-item boolean, indexed by internal id.
        image_available: Per-item boolean, indexed by internal id.
        internal_to_external: The mapping public ids are resolved through.

    Raises:
        DataError: The masks disagree in length, or nothing is representable.
    """
    lengths = {len(warm_items), len(text_available), len(image_available)}
    if len(lengths) != 1:
        raise DataError(
            "Warmth and modality masks must cover the same catalogue",
            warm=len(warm_items),
            text=len(text_available),
            image=len(image_available),
        )

    content = text_available | image_available
    # Warm items survive without content because the id residual can represent
    # them; cold items need content because nothing else can.
    keep = content | warm_items
    if not keep.any():
        raise DataError(
            "No item is representable. Every item lacks both content and a "
            "fitting interaction, so the catalogue would be empty."
        )

    ids = np.flatnonzero(keep).astype("int64")
    excluded = np.flatnonzero(~keep).astype("int64")
    items = pd.DataFrame(
        {
            "internal_item_id": ids,
            "external_item_id": [internal_to_external.get(int(i), "") for i in ids],
            "warm_item_flag": warm_items[ids],
            "text_available": text_available[ids],
            "image_available": image_available[ids],
            "both_modalities_available": text_available[ids] & image_available[ids],
            "content_representable": content[ids],
        }
    ).sort_values("internal_item_id", ignore_index=True)

    catalogue = RetrievalCatalogue(
        items=items,
        warm_count=int(items["warm_item_flag"].sum()),
        cold_count=int((~items["warm_item_flag"]).sum()),
        excluded_count=int(excluded.size),
        excluded_item_ids=tuple(int(value) for value in excluded[:100]),
    )
    logger.info(
        "two_tower.catalogue_built",
        items=len(catalogue),
        warm=catalogue.warm_count,
        cold=catalogue.cold_count,
        excluded=catalogue.excluded_count,
        cold_content_representable=int(
            (~items["warm_item_flag"] & items["content_representable"]).sum()
        ),
    )
    return catalogue


__all__ = [
    "CATALOGUE_FILENAME",
    "COLUMNS",
    "ITEM_INDEX_FILENAME",
    "RetrievalCatalogue",
    "build_catalogue",
]
