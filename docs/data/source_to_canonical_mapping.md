# Source-to-canonical mapping

How PixelRec50K's columns become OmniRank canonical records. Implemented in
[`src/omnirank/data/pixelrec/canonical.py`](../../src/omnirank/data/pixelrec/canonical.py).

## Users

PixelRec ships **no user table**. Users are derived from the distinct `user_id`
values in the interaction log.

| Source file | Source field | Canonical field | Transformation | Required | Notes |
|---|---|---|---|---|---|
| `interaction.csv` | `user_id` | `external_user_id` | strip whitespace, keep as string | ✅ | Whitespace stripping matters: a padded id would become a second, distinct user |
| — | — | `internal_user_id` | dense rank over sorted external ids | ✅ | Assigned after filtering |
| — | — | `created_at` | **not available** | — | Never fabricated |
| — | — | demographics | **not available** | — | Never fabricated |

## Items

| Source file | Source field | Canonical field | Transformation | Required | Notes |
|---|---|---|---|---|---|
| `item_info.csv` | `item_id` | `external_item_id` | strip whitespace | ✅ | |
| — | — | `internal_item_id` | dense rank over sorted external ids | ✅ | |
| `item_info.csv` | `title` | `title` | strip; `""` → null | ✖ | 192 items (0.23%) have none |
| `item_info.csv` | `description` | `description` | strip; `""` → null | ✖ | 19,758 items (23.8%) have none |
| `item_info.csv` | `tag` | `category` | strip; `""` → null | ✖ | Renamed: one tag per item from a 108-value vocabulary *is* a category |
| — | — | `image_reference` | `f"{item_id}.jpg"` | ✅ | An identifier, not a path — keeps artifacts portable |
| — | — | `text_feature_reference` | `item_id` | ✅ | The feature files are keyed by item id |
| — | — | `image_feature_reference` | `item_id` | ✅ | |
| `item_info.csv` | `view_number` | `source_metadata.view_number` | JSON, float or null | ✖ | **Metadata only — see below** |
| `item_info.csv` | `comment_number` | `source_metadata.comment_number` | JSON | ✖ | |
| `item_info.csv` | `thumbup_number` | `source_metadata.thumbup_number` | JSON | ✖ | |
| `item_info.csv` | `share_number` | `source_metadata.share_number` | JSON | ✖ | |
| `item_info.csv` | `coin_number` | `source_metadata.coin_number` | JSON | ✖ | |
| `item_info.csv` | `favorite_number` | `source_metadata.favorite_number` | JSON | ✖ | |
| `item_info.csv` | `barrage_number` | `source_metadata.barrage_number` | JSON | ✖ | |
| — | — | `created_at` | **not available** | — | No publication date exists |
| — | — | `available` | defaults `true` | — | Means "no availability signal", not "verified available" |

## Interactions

| Source file | Source field | Canonical field | Transformation | Required | Notes |
|---|---|---|---|---|---|
| `interaction.csv` | *(row index)* | `source_row_id` | 0-based position in the file | ✅ | Lets a rejected record point at a source line |
| — | — | `interaction_id` | `f"pr50k-{source_row_id}"` | ✅ | **Derived surrogate key**, deterministic — see below |
| `interaction.csv` | `user_id` | `external_user_id` | strip | ✅ | |
| `interaction.csv` | `item_id` | `external_item_id` | strip | ✅ | |
| `interaction.csv` | `timestamp` | `timestamp` | int64 epoch seconds, unchanged | ✅ | **The ordering key** |
| `interaction.csv` | `timestamp` | `event_timestamp_utc` | `to_datetime(unit="s", utc=True)` | ✅ | Human/contract view of the same instant |
| — | — | `event_type` | constant `"interaction"` | ✅ | **See below** |
| — | — | `interaction_weight` | constant `1.0` | ✅ | |
| — | — | `interaction_order` | per-user rank by `(timestamp, source_row_id)` | ✅ | Derived at the splitting stage |
| — | — | `split` | assigned by the splitter | ✅ | |
| — | — | `session_id` | **not available** | — | |
| — | — | `event_value` | **not available** | — | No rating or duration exists |

---

## Three decisions worth defending

### `event_type = "interaction"`

PixelRec records that a user engaged with an item and **nothing finer**. There
is no column distinguishing a view from a click from a like.

Calling it `click` would be a fabrication that propagates: the domain profile
assigns weights per event type, so a mislabelled event silently acquires a
weight that encodes intent the source never measured. `EventType.INTERACTION`
was added to the canonical vocabulary in Phase 2 precisely so this case has an
honest name.

### Engagement counters are metadata, never features

`view_number` and its six siblings look like excellent popularity features. They
are excluded from the feature path, for two independent reasons:

1. **They describe the wrong population.** They are the *whole platform's*
   lifetime totals — tens of millions of users — not the 50,000 users in this
   dataset. An item's `view_number` says nothing about its popularity within the
   corpus being modelled.
2. **They cannot be point-in-time bounded.** They carry no timestamp, so there
   is no way to know what they were during the training window. Using them
   attaches an end-of-2024 popularity value to a 2012 interaction — future
   information, by definition. See [`leakage_prevention.md`](leakage_prevention.md).

They are preserved verbatim in `source_metadata` so nothing is lost, and are
absent from `item_training_popularity.parquet`, which is computed from training
interactions only.

### `interaction_id` is derived, and that is not fabrication

PixelRec assigns no interaction id, but the canonical contract needs one for
deduplication reporting and for the "no interaction in two splits" leakage check.

`pr50k-<source_row_id>` is a **surrogate key**: deterministic, reproducible from
the source file alone, and traceable back to a specific line. It asserts nothing
about the world. That is categorically different from inventing a `price` or a
`created_at`, which would assert facts the source never recorded.

---

## Fields that must never be fabricated

Absent in PixelRec, and absent from every output. No null-filled column, no
default, no imputation:

**Items:** `price`, `brand`, `rating`, `inventory`, `created_at`, cart/purchase/
wishlist status.
**Users:** any demographic attribute, `created_at`, locale, segment.
**Interactions:** `session_id`, `event_value`, watch duration, any event type
other than `interaction`.

Enforced by tests in
[`tests/unit/data/test_canonical.py`](../../tests/unit/data/test_canonical.py)
and [`tests/integration/test_pipeline.py`](../../tests/integration/test_pipeline.py),
which assert these column names never appear in any output.
