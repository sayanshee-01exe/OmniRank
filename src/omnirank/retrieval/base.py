"""Candidate aggregation and vector-index contracts - components 11 and 15.

Retrieval is the stage where several independent generators become one list.
Two contracts live here:

* :class:`CandidateAggregator` - merge, deduplicate, and truncate.
* :class:`VectorIndex` - the approximate-nearest-neighbour store the embedding
  generators query. FAISS first (ADR-004); the interface exists so that moving
  to pgvector or a managed service later touches one file.

PHASE 1 STATUS: contracts only. Both land in Phase 2-3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

from omnirank.models.base import Candidate


@dataclass(frozen=True, slots=True)
class AggregationResult:
    """Merged candidates plus a per-source audit trail.

    ``contributions`` records how many candidates each generator supplied
    *after* deduplication. It is the diagnostic that answers "why did recall
    drop" when one generator silently stops producing.
    """

    candidates: tuple[Candidate, ...]
    contributions: dict[str, int]
    # Generators that failed or timed out. Non-empty means the response is
    # degraded even when it is not a full fallback.
    degraded_sources: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """True when nothing survived, which must trigger the fallback chain."""
        return not self.candidates


class CandidateAggregator(ABC):
    """Merges per-generator candidate lists into one ordered set.

    The hard part is not merging but *comparison*: generator scores live on
    incomparable scales (a dot product, a popularity count, a softmax
    probability). Implementations must normalise within source before any
    cross-source comparison, or blend by rank rather than by score.
    """

    @abstractmethod
    def aggregate(
        self,
        per_source: dict[str, Sequence[Candidate]],
        *,
        limit: int,
    ) -> AggregationResult:
        """Merge per-source lists into at most ``limit`` unique candidates.

        Args:
            per_source: Generator name to its candidates, best first.
            limit: Maximum candidates to emit, i.e. the ranker's input budget.

        Implementations must preserve every contributing source on the merged
        candidate (see :meth:`omnirank.models.base.Candidate.merged_with`), and
        must be deterministic given identical inputs.
        """


@runtime_checkable
class VectorIndex(Protocol):
    """Approximate nearest-neighbour store over item embeddings.

    Row order is the dense item index from
    :class:`~omnirank.data.id_mapping.IdMapping`; the index deliberately does
    not know about string ids, so it cannot drift out of sync with the mapping
    in a way that silently resolves.
    """

    @property
    def dimension(self) -> int:
        """Embedding dimensionality this index was built for."""
        ...

    @property
    def index_version(self) -> int:
        """Build version, checked against artifact metadata (ADR-006)."""
        ...

    def build(self, embeddings: Any, *, metric: str = "inner_product") -> None:
        """Build the index from a ``(num_items, dimension)`` embedding matrix."""
        ...

    def search(self, query: Any, k: int) -> tuple[list[list[int]], list[list[float]]]:
        """Return the ``k`` nearest item indices and their similarity scores.

        Returns:
            ``(indices, scores)``, each a list per query row. Padding with
            ``-1`` is required when fewer than ``k`` results exist, so callers
            can rely on rectangular output.
        """
        ...

    def save(self, path: str | Path) -> None:
        """Persist the index."""
        ...

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Restore a persisted index."""
        ...


__all__ = ["AggregationResult", "CandidateAggregator", "VectorIndex"]
