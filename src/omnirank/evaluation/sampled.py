"""Sampled-negative evaluation - for development loops only.

Ranking a positive against 100 sampled items answers a much easier question than
ranking it against 69,347 real ones, and the two numbers differ by roughly an
order of magnitude. Published recommender results are frequently not comparable
for exactly this reason.

So this exists to make a development loop fast, and everything about it is built
to stop the resulting numbers escaping into a report:

* Every result carries ``protocol="sampled"`` and the negative count.
* :func:`assert_not_final` raises if a sampled result reaches a reporting path.
* :func:`warn_if_incomparable` refuses to line a sampled number up beside a
  full-catalogue one without saying so.

Before Phase 4 this module did not exist, and ``protocol: sampled`` was an
accepted config value that silently ran full-catalogue evaluation - a worse
failure than either implementing it or rejecting it, because the label said one
thing and the number meant another.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from omnirank.core.exceptions import DataError
from omnirank.core.logging import get_logger

logger = get_logger(__name__)

SAMPLED = "sampled"
FULL_CATALOGUE = "full_catalogue"

#: Sampled results may never be the headline. Raised by :func:`assert_not_final`.
_FINAL_STAGES = frozenset({"final", "test", "report"})


@dataclass(frozen=True, slots=True)
class SampledCandidateSet:
    """The per-user candidate pool a sampled evaluation ranks within.

    Holds the user's true target plus ``num_negatives`` items they have not
    interacted with. A model is scored only against this pool, so the metric
    means "did it rank the target above these particular negatives".
    """

    #: user id -> the candidate pool, target first for reproducible construction.
    candidates: Mapping[str, tuple[str, ...]]
    num_negatives: int
    seed: int
    #: True when every model being compared saw the identical pool, which is the
    #: only way two sampled numbers are comparable to each other.
    shared_across_models: bool = True

    def pool_for(self, user_id: str) -> tuple[str, ...]:
        """The candidate pool for one user; empty when the user has none."""
        return self.candidates.get(user_id, ())

    def describe(self) -> dict[str, Any]:
        """Report-ready description. Always travels with the metrics."""
        return {
            "protocol": SAMPLED,
            "num_negatives": self.num_negatives,
            "sampling": "uniform over items the user has not interacted with",
            "seed": self.seed,
            "users": len(self.candidates),
            "shared_across_models": self.shared_across_models,
            "comparable_to_full_catalogue": False,
        }


def build_sampled_candidates(
    *,
    targets: Mapping[str, str],
    seen_by_user: Mapping[str, set[str]],
    catalogue: Sequence[str],
    num_negatives: int = 100,
    seed: int = 42,
) -> SampledCandidateSet:
    """Build one candidate pool per user: their target plus sampled negatives.

    Args:
        targets: user id -> their held-out target item.
        seen_by_user: user id -> items they interacted with in the fit data.
        catalogue: Items the model may recommend.
        num_negatives: Negatives per user.
        seed: Fixed, so the pools are reproducible.

    Returns:
        A :class:`SampledCandidateSet`.

    Raises:
        DataError: ``num_negatives`` is not positive, or the catalogue is too
            small to draw that many negatives for some user.

    The same seed produces the same pools, so two models compared under this
    protocol rank within an identical set - without which sampled numbers are not
    even comparable to each other, let alone to a full-catalogue run.
    """
    if num_negatives < 1:
        raise DataError("num_negatives must be positive", num_negatives=num_negatives)
    items = np.asarray(catalogue, dtype=object)
    if items.size < num_negatives + 1:
        raise DataError(
            "Catalogue is too small to sample this many negatives",
            catalogue_size=int(items.size),
            num_negatives=num_negatives,
        )

    generator = np.random.default_rng(seed)
    pools: dict[str, tuple[str, ...]] = {}
    # Users are sorted so the draw depends on the seed alone, not on dict order.
    for user_id in sorted(targets):
        target = targets[user_id]
        excluded = set(seen_by_user.get(user_id, set()))
        excluded.add(target)
        negatives: list[str] = []
        # Oversample then filter: cheaper than rejection-testing one at a time,
        # and bounded because the excluded set is tiny next to the catalogue.
        attempts = 0
        while len(negatives) < num_negatives and attempts < 32:
            draw = items[generator.integers(0, items.size, size=num_negatives * 2)]
            for candidate in draw:
                if candidate not in excluded:
                    negatives.append(str(candidate))
                    excluded.add(str(candidate))
                    if len(negatives) == num_negatives:
                        break
            attempts += 1
        if len(negatives) < num_negatives:
            raise DataError(
                "Could not draw enough negatives for a user; their seen set covers "
                "too much of the catalogue",
                user_id=user_id,
                drawn=len(negatives),
                requested=num_negatives,
            )
        pools[user_id] = (target, *negatives)

    result = SampledCandidateSet(candidates=pools, num_negatives=num_negatives, seed=seed)
    logger.info("evaluation.sampled_pools_built", **result.describe())
    return result


def restrict_to_pool(
    recommendations: Mapping[str, Sequence[str]], pool: SampledCandidateSet
) -> dict[str, list[str]]:
    """Keep only items inside each user's candidate pool, order preserved.

    This is what makes the evaluation "sampled": the model still ranks the whole
    catalogue, but is scored on how it ordered the pool.
    """
    restricted: dict[str, list[str]] = {}
    for user_id, items in recommendations.items():
        allowed = set(pool.pool_for(user_id))
        restricted[user_id] = [item for item in items if item in allowed]
    return restricted


def assert_not_final(protocol: str, stage: str) -> None:
    """Refuse to let a sampled result become a reported one.

    Raises:
        DataError: A sampled protocol was used at a final/reporting stage.
    """
    if protocol == SAMPLED and stage in _FINAL_STAGES:
        raise DataError(
            "Sampled-negative evaluation cannot produce a reported result. It "
            "ranks against a handful of sampled items rather than the catalogue, "
            "so its numbers are roughly an order of magnitude higher and are not "
            "comparable to anything published. Use protocol=full for stage "
            f"{stage!r}.",
            protocol=protocol,
            stage=stage,
        )


def warn_if_incomparable(protocols: Sequence[str]) -> None:
    """Log loudly when sampled and full-catalogue numbers appear side by side."""
    distinct = set(protocols)
    if len(distinct) > 1 and SAMPLED in distinct:
        logger.warning(
            "evaluation.incomparable_protocols",
            protocols=sorted(distinct),
            detail=(
                "Sampled and full-catalogue metrics are being compared. They "
                "measure different tasks and differ by roughly an order of "
                "magnitude; the comparison is not meaningful."
            ),
        )


__all__ = [
    "FULL_CATALOGUE",
    "SAMPLED",
    "SampledCandidateSet",
    "assert_not_final",
    "build_sampled_candidates",
    "restrict_to_pool",
    "warn_if_incomparable",
]
