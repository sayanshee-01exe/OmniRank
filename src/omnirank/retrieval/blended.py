"""A composite retriever that fuses several generators into one.

This exists so a *combination* of retrievers can be measured by exactly the
same driver that measures a single one. Phase 3 established that the harness is
shared; a hybrid that produced its recommendations through some bespoke path
would reintroduce the comparability problem the shared harness was built to
avoid.

Two Phase 4 deliverables reduce to this one class:

* the popularity + BPR hybrid baseline, and
* the aggregation experiments, which are the same construction over LightGCN,
  SASRec, BPR and popularity.

**Over-retrieval.** Each source is asked for ``over_retrieval_factor * k``
candidates rather than ``k``. Fusing top-``k`` lists and then truncating back to
``k`` yields fewer than ``k`` distinct items whenever the sources agree, so the
deeper request is what makes the output length independent of how much the
sources overlap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from omnirank.core.exceptions import DataError, ModelNotFittedError
from omnirank.core.logging import get_logger
from omnirank.models.base import Candidate, CandidateGenerator
from omnirank.retrieval.base import AggregationResult, CandidateAggregator

logger = get_logger(__name__)

#: Ask each source for this multiple of k, so fusion has depth to work with.
DEFAULT_OVER_RETRIEVAL_FACTOR = 3


@runtime_checkable
class BlendableSource(Protocol):
    """What a retriever must provide to be usable as a fusion source.

    Deliberately a local protocol rather than a widening of
    :class:`~omnirank.models.base.CandidateGenerator`. ``recommend_batch`` is
    the memory-bounded evaluation path, which the base contract leaves to
    implementations (ADR-001); requiring it here states the blend's actual
    dependency without imposing it on every generator that will never be
    blended.
    """

    name: str

    @property
    def is_fitted(self) -> bool:
        """Whether the source is ready to serve."""

    @property
    def fit_item_catalogue(self) -> set[int]:
        """Internal ids this source can return."""

    def recommend(
        self, user_id: str, k: int, context: dict[str, Any] | None = None
    ) -> list[Candidate]:
        """Top-k candidates for one user."""

    def recommend_batch(
        self, user_ids: list[str], k: int, *, filter_seen: bool = True
    ) -> dict[str, list[str]]:
        """Top-k external item ids for many users."""

    def score(self, user_id: str, item_ids: list[str]) -> list[float]:
        """Scores for specific items."""


class BlendedRetriever(CandidateGenerator):
    """Fuse several fitted candidate generators through one aggregator."""

    name = "blended"

    def __init__(
        self,
        sources: Mapping[str, BlendableSource],
        aggregator: CandidateAggregator,
        *,
        name: str | None = None,
        over_retrieval_factor: int = DEFAULT_OVER_RETRIEVAL_FACTOR,
    ) -> None:
        super().__init__()
        if not sources:
            raise DataError("A blended retriever needs at least one source")
        if over_retrieval_factor < 1:
            raise DataError("over_retrieval_factor must be >= 1", factor=over_retrieval_factor)
        unfitted = sorted(source for source, model in sources.items() if not model.is_fitted)
        if unfitted:
            raise ModelNotFittedError(
                "Every source must be fitted before it can be blended", unfitted=unfitted
            )
        self.sources = dict(sources)
        self.aggregator = aggregator
        self.over_retrieval_factor = over_retrieval_factor
        if name:
            self.name = name
        self._fitted = True

    def fit(self, data: Any) -> None:
        """Not supported: the sources are fitted independently, then combined.

        Fitting here would have to decide what "training a fusion" means for
        four models with four objectives, and Phase 4 makes no such claim -- the
        aggregators are all parameter-free given their weights.
        """
        raise DataError(
            "BlendedRetriever does not fit. Fit each source, then blend them.",
            sources=sorted(self.sources),
        )

    @property
    def fit_item_catalogue(self) -> set[int]:
        """Union of what the sources can return.

        A union, not an intersection: an item any source can retrieve is
        reachable through the blend.
        """
        catalogue: set[int] = set()
        for model in self.sources.values():
            catalogue |= model.fit_item_catalogue
        return catalogue

    def _per_source(self, user_id: str, k: int) -> dict[str, Sequence[Candidate]]:
        """Collect each source's list for one user."""
        depth = k * self.over_retrieval_factor
        return {source: model.recommend(user_id, depth) for source, model in self.sources.items()}

    def aggregate_for(self, user_id: str, k: int) -> AggregationResult:
        """Full aggregation result for one user, including the audit trail."""
        return self.aggregator.aggregate(self._per_source(user_id, k), limit=k)

    def recommend(
        self, user_id: str, k: int, context: dict[str, Any] | None = None
    ) -> list[Candidate]:
        """Top-``k`` fused candidates for one user."""
        return list(self.aggregate_for(user_id, k).candidates)

    def recommend_batch(
        self, user_ids: list[str], k: int, *, filter_seen: bool = True
    ) -> dict[str, list[str]]:
        """Fuse per user, using each source's own batch path.

        The sources are asked in batch -- one pass over the catalogue per source
        rather than one per user -- and only the fusion runs per user. This is
        the path the evaluation harness uses.

        **Scores are rank-derived here, not real.** The batch protocol returns
        ranked ids without scores, so this reconstructs a descending placeholder
        from each item's position. For the rank-based strategies -- round robin,
        RRF, and ``normalized_score_union`` under ``rank_percentile`` -- that is
        exactly equivalent to the real thing, because those strategies read only
        the ordering, which is preserved.

        It is *not* equivalent for ``min_max`` or ``z_score`` normalisation,
        which read magnitude: under those, this path silently fuses uniform rank
        gaps instead of a source's actual score spread, and would disagree with
        :meth:`recommend`. Those normalisations are therefore not used in the
        Phase 4 blend grid. Making them correct here means having sources return
        scores in batch, which is an interface change across every generator and
        is deliberately not made as a side effect of blending.
        """
        depth = k * self.over_retrieval_factor
        per_source_batches = {
            source: model.recommend_batch(user_ids, depth, filter_seen=filter_seen)
            for source, model in self.sources.items()
        }
        results: dict[str, list[str]] = {}
        for user_id in user_ids:
            per_source: dict[str, Sequence[Candidate]] = {
                source: [
                    # Descending placeholder: preserves rank order for the
                    # score-reading aggregators without implying a real score.
                    # See the docstring on when this is and is not equivalent.
                    Candidate(
                        item_id=item,
                        score=float(depth - position),
                        sources=(source,),
                        source_scores={source: float(depth - position)},
                    )
                    for position, item in enumerate(batch.get(user_id, []))
                ]
                for source, batch in per_source_batches.items()
            }
            results[user_id] = [
                candidate.item_id
                for candidate in self.aggregator.aggregate(per_source, limit=k).candidates
            ]
        return results

    def score(self, user_id: str, item_ids: list[str]) -> list[float]:
        """Mean of the sources' scores.

        Only meaningful when the sources share a scale, which in general they do
        not -- popularity returns a decayed count and BPR a dot product. Ranking
        goes through :meth:`recommend`; this exists to satisfy the interface.
        """
        per_source = [model.score(user_id, item_ids) for model in self.sources.values()]
        return [sum(values) / len(values) for values in zip(*per_source, strict=True)]

    def metadata(self) -> dict[str, Any]:
        """What was blended, and how."""
        return {
            "model": self.name,
            "kind": "blended",
            "sources": sorted(self.sources),
            "aggregator": type(self.aggregator).__name__,
            "over_retrieval_factor": self.over_retrieval_factor,
            "source_metadata": {
                source: model.metadata()
                for source, model in self.sources.items()
                if hasattr(model, "metadata")
            },
        }

    def save(self, path: Any) -> None:
        """Not supported: save the sources, then reconstruct the blend.

        Persisting a blend would duplicate every source's weights inside it and
        create a second copy that can drift from the registered originals.
        """
        raise DataError(
            "BlendedRetriever is not persisted. Save each source and rebuild "
            "the blend from the registry.",
            sources=sorted(self.sources),
        )

    @classmethod
    def load(cls, path: Any) -> BlendedRetriever:
        """Not supported, symmetrically with :meth:`save`.

        A blend is reconstructed by loading its sources from the registry and
        passing them here, which keeps one authoritative copy of each model.
        """
        raise DataError(
            "BlendedRetriever is not loaded from disk. Load each source from "
            "the registry and construct the blend from them.",
            path=str(path),
        )


__all__ = ["DEFAULT_OVER_RETRIEVAL_FACTOR", "BlendableSource", "BlendedRetriever"]
