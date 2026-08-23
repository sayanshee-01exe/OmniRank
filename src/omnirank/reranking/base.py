"""Post-ranking contracts - component 13.

Two distinct concerns share this stage, and conflating them is a common design
error:

* :class:`PostRankingFilter` - **correctness**. Remove what must not be shown:
  unavailable items, already-purchased items, blocked categories. Filters may
  shorten the list and are never optional.
* :class:`Reranker` - **quality**. Reorder what may be shown, trading a little
  relevance for diversity, freshness, or business objectives. MMR is the first
  implementation (Phase 5).

Filters run before rerankers: reordering a list that still contains items you
are about to drop wastes the diversity budget on items nobody will see.

PHASE 1 STATUS: contracts only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from omnirank.models.base import RankedItem


class PostRankingFilter(ABC):
    """Removes items that must not be shown, for reasons outside relevance."""

    #: Stable name, reported in serving diagnostics when a filter empties a list.
    name: str = "post_ranking_filter"

    @abstractmethod
    def apply(
        self,
        items: Sequence[RankedItem],
        context: dict[str, Any] | None = None,
    ) -> list[RankedItem]:
        """Return the surviving items, order preserved.

        Implementations must not renumber ``rank``; renumbering happens once,
        after the whole post-ranking chain, so that each filter's effect stays
        attributable.
        """


class Reranker(ABC):
    """Reorders a ranked list to optimise something beyond pointwise relevance."""

    #: Stable name, recorded in the response so an A/B arm is identifiable.
    name: str = "reranker"

    @abstractmethod
    def rerank(
        self,
        items: Sequence[RankedItem],
        k: int,
        context: dict[str, Any] | None = None,
    ) -> list[RankedItem]:
        """Return the top ``k`` items in their new order, ranks renumbered from 1.

        Args:
            items: Ranked candidates, best first.
            k: How many to return.
            context: Request-time signals. Similarity-based rerankers such as
                MMR require item embeddings here or via an injected provider;
                they must never fetch them synchronously per request (ADR-003).

        Implementations must be deterministic and must return at most ``k``
        items, never more.
        """


__all__ = ["PostRankingFilter", "Reranker"]
