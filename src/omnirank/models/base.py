"""Core model interfaces - components 8, 10, 12.

Two abstractions carry the whole pipeline:

* :class:`CandidateGenerator` - "given a user, propose items". Popularity,
  matrix factorization, LightGCN, the two-tower retriever and SASRec are all
  interchangeable behind it, which is what lets the aggregator treat them as a
  list rather than as five special cases.
* :class:`Ranker` - "given candidates and their features, order them".

Both are abstract base classes rather than bare protocols, deliberately: the
shared ``fit``-then-``recommend`` state machine (``_fitted``) and the
``save``/``load`` round-trip are behaviour every implementation needs and none
should reimplement.

PHASE 4 STATUS: :class:`CandidateGenerator` has five implementations --
popularity and BPR matrix factorization (Phase 3), LightGCN and SASRec
(Phase 4), and the composite ``omnirank.retrieval.blended.BlendedRetriever``.
:class:`Ranker` still has none; it lands in Phase 6. The interface has not been
widened for any of them (ADR-001): the memory-bounded ``recommend_batch`` the
evaluation harness uses is a separate protocol, deliberately not a method here,
because a generator that will never be evaluated in bulk should not have to
implement it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from omnirank.core.exceptions import ModelNotFittedError


@dataclass(frozen=True, slots=True)
class Candidate:
    """One proposed item on its way through the pipeline.

    ``sources`` is a tuple rather than a single string because the aggregator
    merges the same item arriving from several generators, and both the ranker's
    features and the API's explanation need to know it was multiply-nominated.
    """

    item_id: str
    # Generator-local score. NOT comparable across generators - each has its own
    # scale. The aggregator normalises before any cross-source comparison.
    score: float
    sources: tuple[str, ...] = ()
    # Per-source raw scores, kept for ranking features and for debugging why an
    # item surfaced.
    source_scores: dict[str, float] = field(default_factory=dict)

    def merged_with(self, other: Candidate) -> Candidate:
        """Combine two nominations of the same item from different generators.

        Raises:
            ValueError: The two candidates are not the same item.
        """
        if self.item_id != other.item_id:
            raise ValueError(
                f"cannot merge candidates for different items: "
                f"{self.item_id!r} and {other.item_id!r}"
            )
        return Candidate(
            item_id=self.item_id,
            score=max(self.score, other.score),
            sources=tuple(dict.fromkeys((*self.sources, *other.sources))),
            source_scores={**self.source_scores, **other.source_scores},
        )


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """One retrieved item with the score *and* rank its source actually gave it.

    Both are carried because they answer different questions and a ranker wants
    both. The rank says where a source placed an item relative to its own other
    candidates; the score says how strongly, on that source's own scale.

    Critically, ``score`` is the model's genuine output -- never a value
    reconstructed from ``rank``. A reciprocal of the rank looks like a score,
    lands in a column named score, and silently replaces the signal a ranker was
    supposed to learn from with a monotone function of information it already
    has. Sources that cannot produce a score record ``None`` rather than a
    stand-in.
    """

    item_id: str
    rank: int
    score: float | None
    source: str

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"rank is 1-based; got {self.rank}")


@dataclass(frozen=True, slots=True)
class RankedItem:
    """A candidate after ranking, carrying its final position."""

    item_id: str
    rank: int
    score: float
    sources: tuple[str, ...] = ()
    # Human-readable justification surfaced in the API response. Generated from
    # the sources and features - never invented.
    reason: str | None = None


class CandidateGenerator(ABC):
    """Proposes a set of items for a user.

    Implementations are retrieval-stage models: cheap enough to run over the
    whole catalogue, and judged on recall rather than on precise ordering.
    """

    #: Stable identifier used in configuration, in ``Candidate.sources``, and in
    #: artifact names. Subclasses must override.
    name: str = "candidate_generator"

    def __init__(self) -> None:
        self._fitted = False

    # -- lifecycle ---------------------------------------------------------- #
    @abstractmethod
    def fit(self, data: Any) -> None:
        """Train on a preprocessed dataset.

        Args:
            data: An :class:`~omnirank.data.preprocessing.PreprocessedDataset`.
                Typed loosely here so that this module stays importable without
                the training stack installed.

        Implementations must set ``self._fitted = True`` on success.
        """

    @abstractmethod
    def recommend(
        self,
        user_id: str,
        k: int,
        context: dict[str, Any] | None = None,
    ) -> list[Candidate]:
        """Return up to ``k`` candidates for a user, best first.

        Args:
            user_id: Opaque user identifier.
            k: Maximum number to return. Implementations may return fewer -
                a cold user with no history legitimately yields nothing, and the
                fallback chain, not the generator, is responsible for that case.
            context: Request-time signals (session items, category filter,
                device, locale). Unknown keys must be ignored, not rejected, so
                that adding a signal does not break every generator at once.

        Raises:
            ModelNotFittedError: Called before ``fit``/``load``.
        """

    @abstractmethod
    def score(self, user_id: str, item_ids: list[str]) -> list[float]:
        """Score specific items for a user, in the order given.

        Used by the ranking stage to turn "this generator nominated it" into a
        feature for *every* candidate, including ones this generator did not
        nominate. Items unknown to the model score as ``0.0`` rather than
        raising, because an unknown item is a legitimate outcome here.

        Returns:
            One score per input item, same length and order as ``item_ids``.
        """

    # -- persistence -------------------------------------------------------- #
    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist model state to ``path``.

        Implementations must write everything needed by ``load`` and must not
        depend on ambient state (working directory, environment variables).
        """

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Self:
        """Restore a model previously written by :meth:`save`.

        The returned instance must be immediately usable: ``is_fitted`` is True
        and no further ``fit`` call is required.
        """

    # -- shared behaviour --------------------------------------------------- #
    @property
    def is_fitted(self) -> bool:
        """Whether this instance can serve requests."""
        return self._fitted

    def ensure_fitted(self) -> None:
        """Guard for use at the top of ``recommend``/``score``.

        Raises:
            ModelNotFittedError: The model has not been fitted or loaded.
        """
        if not self._fitted:
            raise ModelNotFittedError(
                f"{type(self).__name__} must be fitted or loaded before use",
                generator=self.name,
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, fitted={self._fitted})"

    def recommend_batch_scored(
        self, user_ids: list[str], k: int, *, filter_seen: bool = True
    ) -> dict[str, list[ScoredCandidate]]:
        """Top-``k`` candidates for many users, keeping each source's real score.

        The default implementation loops :meth:`recommend`, which already
        carries genuine scores. Generators that score a whole user batch against
        the catalogue in one pass should override this -- most of them compute
        exactly these numbers and then discard them.

        Returning ``score=None`` is legitimate for a generator that genuinely
        has no meaningful score. Returning a value derived from the rank is not:
        it would occupy the score column with a restatement of the rank.
        """
        results: dict[str, list[ScoredCandidate]] = {}
        for user_id in user_ids:
            context = None if filter_seen else {"filter_seen": False}
            results[user_id] = [
                ScoredCandidate(
                    item_id=candidate.item_id,
                    rank=position,
                    score=float(candidate.score),
                    source=self.name,
                )
                for position, candidate in enumerate(self.recommend(user_id, k, context), start=1)
            ]
        return results


class Ranker(ABC):
    """Orders a candidate list using richer features than retrieval can afford.

    The ranking stage sees hundreds of candidates instead of the whole
    catalogue, which buys the budget for cross features, real-time context, and
    a gradient-boosted model (ADR-008).
    """

    #: Stable identifier used in configuration and artifact names.
    name: str = "ranker"

    def __init__(self) -> None:
        self._fitted = False

    @abstractmethod
    def fit(self, features: Any, labels: Any, groups: Any | None = None) -> None:
        """Train the ranking model.

        Args:
            features: Feature matrix, one row per (user, candidate) pair.
            labels: Relevance labels aligned with ``features``.
            groups: Query-group sizes, i.e. how many rows belong to each user's
                candidate list. Required by pairwise/listwise objectives such as
                LambdaRank; ``None`` only for pointwise training.
        """

    @abstractmethod
    def rank(
        self,
        candidates: list[Candidate],
        context: dict[str, Any] | None = None,
    ) -> list[RankedItem]:
        """Order candidates, best first, assigning 1-based ranks.

        Implementations must be order-stable for equal scores, so that a rerun
        against identical inputs produces an identical response - otherwise
        cached and freshly computed responses disagree.
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the trained ranker."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Self:
        """Restore a ranker previously written by :meth:`save`."""

    @property
    def is_fitted(self) -> bool:
        """Whether this ranker can serve requests."""
        return self._fitted

    def ensure_fitted(self) -> None:
        """Raise :class:`ModelNotFittedError` unless fitted or loaded."""
        if not self._fitted:
            raise ModelNotFittedError(
                f"{type(self).__name__} must be fitted or loaded before use", ranker=self.name
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, fitted={self._fitted})"


__all__ = [
    "Candidate",
    "CandidateGenerator",
    "RankedItem",
    "Ranker",
    "ScoredCandidate",
]
