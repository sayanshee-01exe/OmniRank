# Cleaning rules

Implemented in [`src/omnirank/data/cleaning.py`](../../src/omnirank/data/cleaning.py).

## Two governing principles

**Nothing is dropped silently.** Every rejected row is written to
`data/interim/pixelrec50k/rejected_records.parquet` with its source file, source
row identifier, entity type, rejection reason, and original identifier.

**Every stage reconciles.** Each step asserts
`input_rows == output_rows + removed_rows` *and* that the recorded reasons
account for every removal. Arithmetic that does not balance means a join fanned
out or a filter matched something unintended — cheap to catch here, expensive to
discover as an unexplained metric shift later.

## Rules, in execution order

Order is fixed. Identifier checks precede timestamp checks precede vocabulary
checks precede referential integrity precede deduplication.

| # | Rule | Reason code | Action |
|---:|---|---|---|
| 1 | `user_id` null or blank after stripping | `missing_user_id` | reject |
| 2 | `item_id` null or blank after stripping | `missing_item_id` | reject |
| 3 | `timestamp` null | `missing_timestamp` | reject |
| 4 | `timestamp <= 0` | `invalid_timestamp` | reject |
| 5 | `timestamp` before `data.validation.min_timestamp` | `timestamp_out_of_range` | reject |
| 6 | `timestamp` after "now" | `future_timestamp` | reject |
| 7 | `event_type` not declared by the domain profile | `unknown_event_type` | reject |
| 8 | `interaction_weight` null, negative, or non-finite | `invalid_weight` | reject |
| 9 | `item_id` not in the cleaned item table | `unknown_item_reference` | reject |
| 10 | duplicate business key | `duplicate_interaction` | reject all but the first |

Item-side rules:

| # | Rule | Reason code | Action |
|---:|---|---|---|
| 11 | `item_id` null or blank | `missing_item_id` | reject |
| 12 | `item_id` appears twice | `duplicate_item_id` | keep first, reject the rest |

### Why deduplication runs last

A row failing rule 1 is removed before rule 10 sees it, so it is counted once
under one reason. Running deduplication first would let a single bad row be
attributed to two reasons and break reconciliation.

## Deduplication policy

The business key is:

```text
(external_user_id, external_item_id, event_type, timestamp)
```

`interaction_id` is **deliberately excluded**. Upstream systems routinely
re-emit an event with a fresh id, and counting that twice inflates every
implicit-feedback signal downstream. This is the same key enforced by the
`uq_interactions_event` index in the PostgreSQL schema, so ingestion is
idempotent at both layers.

Keep-first is deterministic because the source file order is stable and its
SHA-256 is recorded in the manifest.

## What is deliberately *not* cleaned

| Condition | Why it survives |
|---|---|
| Item with no title (192) | Still recommendable from collaborative signal |
| Item with no description (19,758) | A missing modality must degrade, not crash (ADR-003) |
| Item with no category (5) | Only affects the category-popularity fallback for that item |
| Item with null engagement counters (5) | Counters are metadata, not features |
| Item with one interaction (13,518) | **Filtering's** decision, not cleaning's — see [`filtering_policy.md`](filtering_policy.md) |

Cleaning removes what is *unusable*. Filtering removes what is *too sparse to
model*. Conflating them would make the sparsity threshold look like a data-quality
rule and hide it from the report that exists to expose it.

## Normalisation applied

- Identifier whitespace stripped (a padded id would become a distinct entity).
- Text fields stripped; empty strings normalised to null so "absent" has one
  representation rather than two.
- `timestamp` kept as int64 epoch seconds *and* materialised as a tz-aware UTC
  datetime, so ordering never depends on datetime parsing being correct.

Nothing else is altered. No case folding, no unicode normalisation, no
truncation — those would change the text the Phase 4 encoders will see.

## Result on PixelRec50K

| Step | Input | Output | Removed |
|---|---:|---:|---:|
| `clean_items` | 82,865 | 82,865 | **0** |
| `clean_interactions` | 989,494 | 989,494 | **0** |

**Zero rejections.** PixelRec50K is genuinely clean: no duplicates, no dangling
references, no malformed timestamps. The rules are exercised instead by
`tests/unit/data/test_cleaning.py`, which injects each violation and asserts the
matching reason code fires.

`rejected_records.parquet` is written regardless — an empty, correctly-typed
table is the honest representation of "nothing was rejected".
