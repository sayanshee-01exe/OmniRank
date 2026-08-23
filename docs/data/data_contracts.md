# Data contracts

Three entities are the interface between raw data of any vertical and everything
downstream. Defined in [`src/omnirank/data/schemas.py`](../../src/omnirank/data/schemas.py),
validated in batch by [`validation.py`](../../src/omnirank/data/validation.py).

Both are ✅ implemented and tested.

## Two rules that apply to all three

**Identifiers are opaque strings.** Dense integer indices belong to `IdMapping`
and never appear in a contract, an API payload, or a database row. A second
identity space would have to be kept in sync with the mappings, and every
desync is a silent mis-recommendation rather than an error.

**Timestamps are timezone-aware UTC.** Naive datetimes are *rejected*, not
assumed local. A temporal split silently corrupted by mixed offsets is the
easiest way to leak the future into training. Aware non-UTC values are
normalised.

---

## Users

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | str (1–128) | ✅ | Opaque, whitespace-stripped |
| `created_at` | datetime (UTC) | ✅ | Must be timezone-aware |
| `attributes` | dict | — | Domain-specific context (locale, segment, cohort) |

Deliberately thin. Demographics are optional and live in `attributes`, because
most verticals have none, and because anything personal not needed for ranking
should not be copied into the feature store at all.

## Items

| Field | Type | Required | Notes |
|---|---|---|---|
| `item_id` | str (1–128) | ✅ | Opaque |
| `title` | str (1–1024) | ✅ | Feeds text embeddings |
| `description` | str (≤20 000) | — | Feeds text embeddings |
| `category` | str (≤256) | — | Used by category-popularity fallback |
| `brand` | str (≤256) | — | |
| `price` | float ≥ 0, finite | — | Currency is a domain-profile concern, not per-row |
| `image_id` | str (≤512) | — | Asset **identifier**, not a filesystem path |
| `available` | bool | — (default `true`) | Enforced in post-ranking |
| `created_at` | datetime (UTC) | ✅ | |
| `attributes` | dict | — | |

`image_id` is an identifier rather than a path so artifacts stay portable
between a laptop and a cloud training host. Text and image fields are both
optional: catalogues are incomplete in practice, and a missing modality must
degrade rather than crash ([ADR-003](../adr/ADR-003-offline-embeddings.md)).

## Interactions

| Field | Type | Required | Notes |
|---|---|---|---|
| `interaction_id` | str (1–128) | ✅ | Source-assigned; **not** the dedup key |
| `user_id` | str | ✅ | |
| `item_id` | str | ✅ | |
| `event_type` | enum | ✅ | See vocabulary below |
| `timestamp` | datetime (UTC) | ✅ | |
| `session_id` | str | — | |
| `event_value` | float | — | Required for `rating`; must not be NaN |
| `weight` | float ≥ 0 | — | Overrides the domain profile's default weight |

### Event vocabulary

Ordered weakest to strongest intent. The enum fixes the *names* so feature code
and evaluation can reason about intent across verticals; a domain profile
assigns the weights.

| Event | e-commerce weight | Positive? |
|---|---|---|
| `view` | 1.0 | ✗ |
| `click` | 2.0 | ✓ |
| `wishlist` | 3.0 | ✓ |
| `cart` | 4.0 | ✓ |
| `purchase` | 5.0 | ✓ |
| `rating` | 3.0 | ✓ |

"Positive" means weight ≥ `data.positive_event_threshold` (2.0). Configured in
[`configs/data/ecommerce.yaml`](../../configs/data/ecommerce.yaml).

### Deduplication key

```python
(user_id, item_id, event_type, timestamp)
```

`interaction_id` is **deliberately excluded**: upstream systems routinely
re-emit an event with a fresh id, and counting that twice inflates every
implicit-feedback signal. The same key backs the `uq_interactions_event`
database index, so ingestion is idempotent at both layers.

---

## Validation rules

Record-level rules come from Pydantic; batch-level rules from `validate_batch`.
Every rejection is tagged with a stable `ValidationRule` identifier that appears
in logs, run reports, and the interaction-ingestion API response.

| Rule | Level | Trigger |
|---|---|---|
| `missing_id` | record | An `*_id` field absent or empty after stripping |
| `malformed_record` | record | Any other schema violation |
| `invalid_timestamp` | batch | Naive, unparseable, or before `data.validation.min_timestamp` |
| `future_timestamp` | batch | After "now" while `allow_future_timestamps` is false |
| `invalid_price` | both | Negative, non-finite, or outside `[min_price, max_price]` |
| `invalid_rating` | batch | `rating` event value outside `[min_rating, max_rating]` |
| `unknown_event_type` | both | Not in the enum, or not declared by the domain profile |
| `duplicate_event` | batch | Dedup key already seen in this batch |
| `duplicate_entity_id` | batch | A `user_id`/`item_id` appears twice |
| `unknown_item_reference` | batch | Interaction references an item that did not survive validation |
| `unknown_user_reference` | batch | Interaction references an unknown user |
| `invalid_availability` | record | `available` not coercible to bool |

### Behaviour

**Non-throwing by default.** Every real catalogue contains bad rows; failing the
whole job on the first is useless. Rejections accumulate in a `ValidationReport`
with `total`, `valid`, `rejected`, `rejection_rate`, and `counts_by_rule`; the
caller decides whether the rate is acceptable. `strict=True` restores fail-fast
for CI fixtures. `report.raise_if_failed()` converts to `SchemaValidationError`.

**Reports are log-safe.** `report.summary()` contains counts only, never record
content, so it can be logged without copying user data into log storage.

**"Now" is injectable.** Every validator takes `now=`, so the test suite never
reads the wall clock and future-timestamp behaviour is deterministic.

### Referential integrity

`validate_batch` validates users and items first, then passes the surviving id
sets to interaction validation. An interaction pointing at a rejected item is
therefore also rejected.

Set `check_references=False` when users or items stream separately and the
reference cannot be resolved yet.

**Note:** the database deliberately carries **no foreign keys** from
`interactions` to `users`/`items`. Events arrive before (or without) their
entities in every real ingestion pipeline, and a rejected event is lost signal.
Integrity is *reported* here rather than *enforced* by a constraint that would
destroy data. See [`database_schema.md`](database_schema.md).

---

## Adding a vertical

1. Write `configs/data/<domain>.yaml` — event vocabulary and weights, validation
   bounds, k-core thresholds, split parameters, sequence lengths.
2. Point `configs/base.yaml`'s `include:` at it.
3. Implement one `DatasetLoader` emitting `User` / `Item` / `Interaction`.

Nothing else changes. Vertical-specific fields ride in `attributes` and become
features, not schema columns.

Worked example — a **courses** profile would map `enroll` → `cart`, `complete` →
`purchase`, and `rating` unchanged; set `min_price: 0` since many courses are
free; and raise `sequences.max_length`, because course histories are shorter but
span longer periods.
