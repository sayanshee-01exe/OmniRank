"""Stable string-to-dense-index mapping - component 4.

Every embedding-based model needs contiguous integer indices; every contract
outside the model boundary uses opaque string identifiers. This module owns the
translation, and owning it in exactly one place is what prevents the classic
training/serving skew where a model trained on one index order is served
against another.

Guarantees:

* **Append-only.** ``add`` never renumbers an existing identifier, so an
  embedding matrix trained against version *n* stays valid against version
  *n + k* for every id it already knew.
* **Persisted with a fingerprint.** The saved file carries a content hash that
  the artifact registry records; a model whose mapping fingerprint disagrees
  with the loaded one is a hard failure, not a silent mis-lookup (ADR-006).
* **No I/O dependencies.** Plain JSON, no pandas, no torch - so serving can
  load a mapping without importing the training stack.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from omnirank.core.exceptions import IdMappingError

# Reserved dense index for "identifier not known to this mapping". Sequence
# models use it as their padding/unknown row, so it must never be assigned.
UNKNOWN_INDEX = -1

MAPPING_FORMAT_VERSION = 1


class IdMapping:
    """Bidirectional, append-only map between string ids and dense indices."""

    __slots__ = ("_id_to_index", "_index_to_id", "entity")

    def __init__(self, entity: str, ids: Iterable[str] = ()) -> None:
        """Create a mapping.

        Args:
            entity: What is being mapped (``"user"``, ``"item"``, ...). Used in
                error messages and in the persisted file.
            ids: Initial identifiers, assigned indices in iteration order.
                Duplicates are collapsed, preserving first appearance.
        """
        self.entity = entity
        self._index_to_id: list[str] = []
        self._id_to_index: dict[str, int] = {}
        for identifier in ids:
            self.add(identifier)

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_ids(cls, entity: str, ids: Iterable[str], *, sort: bool = True) -> IdMapping:
        """Build a mapping from an id collection.

        Args:
            entity: Entity name.
            ids: Identifiers to include.
            sort: Assign indices in sorted order rather than iteration order.
                Defaults to ``True`` so that the same set of ids always yields
                byte-identical mappings regardless of upstream row order - a
                precondition for reproducible training runs.
        """
        unique = set(ids)
        return cls(entity, sorted(unique) if sort else ids)

    # -- mutation ----------------------------------------------------------- #
    def add(self, identifier: str) -> int:
        """Return the index for ``identifier``, assigning a new one if needed."""
        if not identifier:
            raise IdMappingError("Cannot map an empty identifier", entity=self.entity)
        existing = self._id_to_index.get(identifier)
        if existing is not None:
            return existing
        index = len(self._index_to_id)
        self._index_to_id.append(identifier)
        self._id_to_index[identifier] = index
        return index

    def extend(self, ids: Iterable[str]) -> None:
        """Append every unseen identifier, preserving existing indices."""
        for identifier in ids:
            self.add(identifier)

    # -- lookup ------------------------------------------------------------- #
    def to_index(self, identifier: str, *, default: int | None = None) -> int:
        """Map an identifier to its dense index.

        Args:
            identifier: The string id.
            default: Returned when unknown. Pass :data:`UNKNOWN_INDEX` for
                serving paths that must tolerate cold ids.

        Raises:
            IdMappingError: Unknown identifier and no ``default`` given.
        """
        index = self._id_to_index.get(identifier)
        if index is not None:
            return index
        if default is not None:
            return default
        raise IdMappingError(
            f"Unknown {self.entity} identifier", entity=self.entity, identifier=identifier
        )

    def to_id(self, index: int) -> str:
        """Map a dense index back to its identifier.

        Raises:
            IdMappingError: Index outside the mapping.
        """
        if not 0 <= index < len(self._index_to_id):
            raise IdMappingError(
                f"{self.entity} index out of range",
                entity=self.entity,
                index=index,
                size=len(self._index_to_id),
            )
        return self._index_to_id[index]

    def to_indices(self, ids: Iterable[str], *, default: int | None = None) -> list[int]:
        """Vectorised :meth:`to_index`."""
        return [self.to_index(identifier, default=default) for identifier in ids]

    def to_ids(self, indices: Iterable[int]) -> list[str]:
        """Vectorised :meth:`to_id`."""
        return [self.to_id(index) for index in indices]

    # -- introspection ------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._index_to_id)

    def __contains__(self, identifier: object) -> bool:
        return identifier in self._id_to_index

    def __iter__(self) -> Iterator[str]:
        """Iterate identifiers in index order."""
        return iter(self._index_to_id)

    def __repr__(self) -> str:
        return f"IdMapping(entity={self.entity!r}, size={len(self)})"

    @property
    def ids(self) -> tuple[str, ...]:
        """All identifiers, in index order."""
        return tuple(self._index_to_id)

    @property
    def fingerprint(self) -> str:
        """SHA-256 over entity name and id order.

        Two mappings with the same fingerprint are interchangeable; two that
        differ are not, even if they contain the same ids in a different order.
        """
        digest = hashlib.sha256()
        digest.update(self.entity.encode())
        for identifier in self._index_to_id:
            digest.update(b"\x00")
            digest.update(identifier.encode())
        return digest.hexdigest()

    # -- persistence -------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        """Serialisable representation, including the integrity fingerprint."""
        return {
            "format_version": MAPPING_FORMAT_VERSION,
            "entity": self.entity,
            "size": len(self._index_to_id),
            "fingerprint": self.fingerprint,
            # Index is the list position; storing only ids halves the file and
            # makes an inconsistent map structurally impossible.
            "ids": self._index_to_id,
        }

    def save(self, path: Path | str) -> Path:
        """Write the mapping as JSON, creating parent directories."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return target

    @classmethod
    def load(cls, path: Path | str) -> IdMapping:
        """Read a mapping and verify its fingerprint.

        Raises:
            IdMappingError: File missing, malformed, or fingerprint mismatched.
        """
        source = Path(path)
        if not source.is_file():
            raise IdMappingError("Mapping file not found", path=str(source))
        try:
            payload = json.loads(source.read_text())
        except json.JSONDecodeError as exc:
            raise IdMappingError(
                "Mapping file is not valid JSON", path=str(source), reason=str(exc)
            ) from exc

        if payload.get("format_version") != MAPPING_FORMAT_VERSION:
            raise IdMappingError(
                "Unsupported mapping format version",
                path=str(source),
                found=payload.get("format_version"),
                expected=MAPPING_FORMAT_VERSION,
            )

        mapping = cls(payload["entity"], payload["ids"])
        expected = payload.get("fingerprint")
        if expected and expected != mapping.fingerprint:
            raise IdMappingError(
                "Mapping fingerprint mismatch: the file was modified after it was "
                "written, and any model trained against it would mis-resolve ids.",
                path=str(source),
                expected=expected,
                actual=mapping.fingerprint,
            )
        return mapping


__all__ = ["MAPPING_FORMAT_VERSION", "UNKNOWN_INDEX", "IdMapping"]
