"""Training-only user and item statistics - component 6.

Everything here is computed from ``split == "train"`` rows and nothing else.
That single rule is what separates a feature from a leak: an item's popularity
counted over the whole log encodes which items become popular *after* the
training window, and a model given that feature reports offline numbers it
cannot reproduce online.

The rule is not merely intended, it is verified - :func:`.leakage.check_popularity_is_training_only`
independently recounts both tables from the training split and fails the build
on any disagreement.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from omnirank.core.logging import get_logger
from omnirank.data.splitters import TRAIN

logger = get_logger(__name__)

USER_STATISTIC_COLUMNS: Final = (
    "internal_user_id",
    "training_interaction_count",
    "unique_training_items",
    "first_training_interaction_order",
    "last_training_interaction_order",
    "first_training_timestamp",
    "last_training_timestamp",
    "mean_item_popularity",
    "sequence_length",
)

ITEM_POPULARITY_COLUMNS: Final = (
    "internal_item_id",
    "training_interaction_count",
    "training_unique_user_count",
    "training_popularity_rank",
    "training_popularity_percentile",
    "long_tail_flag",
)

#: Items outside the head that accounts for this share of training interactions
#: are long tail. 0.8 follows the common Pareto convention; it is configurable
#: because the right cut depends on the catalogue, and it is recorded in the
#: report so a number is never quoted without the threshold that produced it.
DEFAULT_LONG_TAIL_QUANTILE: Final = 0.8


def build_item_popularity(
    frame: pd.DataFrame, *, long_tail_quantile: float = DEFAULT_LONG_TAIL_QUANTILE
) -> pd.DataFrame:
    """Compute item popularity from training interactions only.

    Args:
        frame: Split-labelled interactions with internal ids.
        long_tail_quantile: Cumulative interaction share defining the head.

    Returns:
        One row per item that appears in training. Items absent from training
        are deliberately omitted rather than given a zero row: they are cold
        items, enumerated by the cold-start slice, and a zero here would make
        them look merely unpopular.
    """
    train = frame[frame["split"] == TRAIN]
    if train.empty:
        return pd.DataFrame(columns=list(ITEM_POPULARITY_COLUMNS))

    grouped = train.groupby("internal_item_id", observed=True)
    popularity = pd.DataFrame(
        {
            "training_interaction_count": grouped.size(),
            "training_unique_user_count": grouped["internal_user_id"].nunique(),
        }
    ).reset_index()

    # Rank 1 is the most popular. `method="first"` keeps ranks unique and
    # deterministic; ties break by internal id, which is stable across runs.
    popularity = popularity.sort_values(
        ["training_interaction_count", "internal_item_id"], ascending=[False, True]
    ).reset_index(drop=True)
    popularity["training_popularity_rank"] = range(1, len(popularity) + 1)
    popularity["training_popularity_percentile"] = (
        popularity["training_interaction_count"].rank(pct=True, method="average").astype("float64")
    )

    # Head = the most popular items whose cumulative interactions first reach the
    # quantile. Everything after that point is long tail.
    cumulative = popularity["training_interaction_count"].cumsum()
    total = int(popularity["training_interaction_count"].sum())
    head_mask = cumulative <= long_tail_quantile * total
    # Always keep at least one head item, otherwise a uniform catalogue makes
    # every item long tail and the slice stops being informative.
    if not head_mask.any():
        head_mask.iloc[0] = True
    popularity["long_tail_flag"] = ~head_mask

    logger.info(
        "statistics.item_popularity",
        items=len(popularity),
        long_tail_items=int(popularity["long_tail_flag"].sum()),
        long_tail_quantile=long_tail_quantile,
    )
    return popularity.loc[:, list(ITEM_POPULARITY_COLUMNS)]


def build_user_statistics(frame: pd.DataFrame, item_popularity: pd.DataFrame) -> pd.DataFrame:
    """Compute per-user profile statistics from training interactions only.

    ``mean_item_popularity`` uses the training-only popularity table, so a
    user's taste-for-popular-items feature cannot encode future popularity.
    """
    train = frame[frame["split"] == TRAIN]
    if train.empty:
        return pd.DataFrame(columns=list(USER_STATISTIC_COLUMNS))

    if not item_popularity.empty:
        train = train.merge(
            item_popularity[["internal_item_id", "training_interaction_count"]].rename(
                columns={"training_interaction_count": "_item_popularity"}
            ),
            on="internal_item_id",
            how="left",
        )
    else:
        # float NaN rather than pd.NA: the column is averaged below, and a
        # masked NA would make the mean itself NA for every user.
        train = train.assign(_item_popularity=float("nan"))

    grouped = train.groupby("internal_user_id", observed=True)
    statistics = pd.DataFrame(
        {
            "training_interaction_count": grouped.size(),
            "unique_training_items": grouped["internal_item_id"].nunique(),
            "first_training_interaction_order": grouped["interaction_order"].min(),
            "last_training_interaction_order": grouped["interaction_order"].max(),
            "first_training_timestamp": grouped["timestamp"].min(),
            "last_training_timestamp": grouped["timestamp"].max(),
            "mean_item_popularity": grouped["_item_popularity"].mean(),
        }
    ).reset_index()
    # Identical to the interaction count today, and kept as its own column
    # because it stops being identical the moment sequence truncation applies.
    statistics["sequence_length"] = statistics["training_interaction_count"]

    logger.info("statistics.user_profiles", users=len(statistics))
    return statistics.loc[:, list(USER_STATISTIC_COLUMNS)]


__all__ = [
    "DEFAULT_LONG_TAIL_QUANTILE",
    "ITEM_POPULARITY_COLUMNS",
    "USER_STATISTIC_COLUMNS",
    "build_item_popularity",
    "build_user_statistics",
]
