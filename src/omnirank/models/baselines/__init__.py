"""Non-neural baselines: popularity and BPR matrix factorization.

Phase 3. Both implement :class:`~omnirank.models.base.CandidateGenerator`.

``popularity`` needs only the ``data`` extra - it is the terminal stage of the
serving fallback chain and must work when nothing else does. ``bpr`` needs the
``baseline`` extra (torch) and is therefore **not** imported here; import it
directly so that a torch-free environment can still use popularity and the
evaluator.
"""

from __future__ import annotations

from omnirank.models.baselines.negative_sampling import (
    UniformNegativeSampler,
    build_positives_by_user,
)
from omnirank.models.baselines.popularity import (
    GLOBAL_COUNT,
    TIME_DECAY,
    PopularityConfig,
    PopularityFitData,
    PopularityRecommender,
    build_seen_by_user,
)

__all__ = [
    "GLOBAL_COUNT",
    "TIME_DECAY",
    "PopularityConfig",
    "PopularityFitData",
    "PopularityRecommender",
    "UniformNegativeSampler",
    "build_positives_by_user",
    "build_seen_by_user",
]
