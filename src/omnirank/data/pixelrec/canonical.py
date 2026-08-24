"""PixelRec50K to OmniRank canonical mapping.

The full field-by-field table lives in ``docs/data/source_to_canonical_mapping.md``.
This module is its executable form.

Three decisions are load-bearing and are enforced here rather than left to
convention:

**Interactions are ``interaction``, not ``click`` or ``view``.** PixelRec records
that a user engaged with an item and nothing finer. Naming it anything more
specific would assert intent the source never measured, and every downstream
weighting scheme would inherit that fiction.

**Engagement counters are metadata, never labels and never features.**
``view_number`` and friends describe the *whole platform's* lifetime behaviour,
not this dataset's 50,000 users, and they carry no timestamp - so they cannot be
point-in-time bounded and would leak future popularity into any training feature
built from them. They are preserved verbatim in ``source_metadata`` and excluded
from the feature path. See ``docs/data/leakage_prevention.md``.

**Absent fields stay absent.** PixelRec has no user table, no item publication
date, no price, no brand, and no rating. Those columns are either omitted or
null. Inventing them would be undetectable downstream, which is exactly why it
is forbidden.

Everything here operates on pandas frames rather than the Pydantic records in
:mod:`omnirank.data.schemas`. The records remain the schema authority - a test
asserts that emitted frames satisfy them - but instantiating a million frozen
models to validate a million rows costs minutes and gigabytes for no additional
guarantee.
"""

from __future__ import annotations

import json
from typing import Final

import pandas as pd

from omnirank.data.pixelrec.loaders import ENGAGEMENT_COLUMNS

#: Every interaction in PixelRec is the same, unlabelled implicit signal.
DEFAULT_EVENT_TYPE: Final = "interaction"
DEFAULT_INTERACTION_WEIGHT: Final = 1.0

#: Prefix for the derived interaction identifier. PixelRec supplies none, so one
#: is derived from the source row index: deterministic, reproducible, and
#: traceable back to a specific line of the original CSV. This is a surrogate
#: key, not fabricated data.
INTERACTION_ID_PREFIX: Final = "pr50k"

#: Cover images are named ``<item_id>.jpg`` in the official cover archive.
IMAGE_EXTENSION: Final = ".jpg"

CANONICAL_USER_COLUMNS: Final = ("external_user_id",)

CANONICAL_ITEM_COLUMNS: Final = (
    "external_item_id",
    "title",
    "description",
    "category",
    "image_reference",
    "text_feature_reference",
    "image_feature_reference",
    "source_metadata",
)

CANONICAL_INTERACTION_COLUMNS: Final = (
    "interaction_id",
    "external_user_id",
    "external_item_id",
    "event_type",
    "timestamp",
    "event_timestamp_utc",
    "interaction_weight",
    "source_row_id",
)


def canonicalize_interactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw interaction rows to the canonical interaction frame.

    Args:
        raw: Frame with PixelRec's ``item_id``/``user_id``/``timestamp`` columns
            plus the loader's ``source_row_id``.

    Returns:
        A canonical frame. ``timestamp`` is kept as the source's integer epoch
        seconds because it is the authoritative ordering key; a parallel
        ``event_timestamp_utc`` carries the same instant as a tz-aware datetime
        for humans and for the Pydantic contract. Keeping both means ordering
        never depends on datetime parsing being correct.
    """
    if raw.empty:
        return pd.DataFrame(columns=list(CANONICAL_INTERACTION_COLUMNS))

    frame = pd.DataFrame(
        {
            "interaction_id": INTERACTION_ID_PREFIX
            + "-"
            + raw["source_row_id"].astype("int64").astype(str),
            # `.str.strip()` on identifiers: leading/trailing whitespace in a CSV
            # id silently creates a second, distinct user.
            "external_user_id": raw["user_id"].astype("string").str.strip(),
            "external_item_id": raw["item_id"].astype("string").str.strip(),
            "event_type": DEFAULT_EVENT_TYPE,
            "timestamp": pd.to_numeric(raw["timestamp"], errors="coerce").astype("Int64"),
            "interaction_weight": DEFAULT_INTERACTION_WEIGHT,
            "source_row_id": raw["source_row_id"].astype("int64"),
        }
    )
    frame["event_timestamp_utc"] = pd.to_datetime(
        frame["timestamp"], unit="s", utc=True, errors="coerce"
    )
    return frame.loc[:, list(CANONICAL_INTERACTION_COLUMNS)]


def canonicalize_items(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw item information rows to the canonical item frame.

    ``tag`` becomes ``category``: PixelRec assigns each item exactly one tag
    from a 108-value vocabulary, which is a category in everything but name.
    The seven engagement counters are folded into ``source_metadata`` as JSON -
    preserved, greppable, and structurally excluded from the feature path.
    """
    if raw.empty:
        return pd.DataFrame(columns=list(CANONICAL_ITEM_COLUMNS))

    item_id = raw["item_id"].astype("string").str.strip()

    def _clean_text(column: str) -> pd.Series:
        """Strip whitespace and normalise empty strings to missing."""
        if column not in raw.columns:
            return pd.Series([pd.NA] * len(raw), dtype="string")
        cleaned = raw[column].astype("string").str.strip()
        return cleaned.mask(cleaned.eq(""), pd.NA)

    counters = [column for column in ENGAGEMENT_COLUMNS if column in raw.columns]
    metadata = (
        raw[counters].to_dict(orient="records") if counters else [{} for _ in range(len(raw))]
    )

    frame = pd.DataFrame(
        {
            "external_item_id": item_id,
            "title": _clean_text("title"),
            "description": _clean_text("description"),
            "category": _clean_text("tag"),
            # A reference, not a path: resolving it against a cover directory is
            # the caller's business, and storing an absolute path here would make
            # the artifact unusable on any other machine.
            "image_reference": item_id + IMAGE_EXTENSION,
            # The official feature files are JSON objects keyed by item id, so
            # the item id *is* the lookup key into both.
            "text_feature_reference": item_id,
            "image_feature_reference": item_id,
            "source_metadata": [
                json.dumps(
                    {key: (None if pd.isna(value) else float(value)) for key, value in row.items()},
                    sort_keys=True,
                )
                for row in metadata
            ],
        }
    )
    return frame.loc[:, list(CANONICAL_ITEM_COLUMNS)]


def derive_users(interactions: pd.DataFrame) -> pd.DataFrame:
    """Derive the canonical user table from the interaction log.

    PixelRec50K ships no user file. Users are exactly the distinct
    ``user_id`` values in the interaction log, and nothing more is known about
    them - no signup date, no demographics, no locale. Emitting a table with
    one honest column is preferable to inventing columns that look informative.
    """
    if interactions.empty:
        return pd.DataFrame(columns=list(CANONICAL_USER_COLUMNS))
    users = (
        interactions["external_user_id"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    return pd.DataFrame({"external_user_id": users})


__all__ = [
    "CANONICAL_INTERACTION_COLUMNS",
    "CANONICAL_ITEM_COLUMNS",
    "CANONICAL_USER_COLUMNS",
    "DEFAULT_EVENT_TYPE",
    "DEFAULT_INTERACTION_WEIGHT",
    "IMAGE_EXTENSION",
    "INTERACTION_ID_PREFIX",
    "canonicalize_interactions",
    "canonicalize_items",
    "derive_users",
]
