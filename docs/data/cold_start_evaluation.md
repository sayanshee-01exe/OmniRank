# Cold-start and evaluation slices

A single aggregate metric hides the failures that matter. A model can look
strong overall while being useless for users with three interactions, or for the
majority of the catalogue nobody has clicked.

These slices exist so Phase 3 reports per-population numbers from its **first**
baseline, rather than discovering the gap after a model has been chosen.

All slices are derived from **training-only** statistics, so membership never
depends on the labels being predicted.

## Format

`data/processed/pixelrec50k/evaluation_slices/<name>.parquet`, each with
`slice_name`, `entity_type`, `entity_id`, plus `slice_manifest.json` listing
every slice with its rule and size.

## Sparse users

Bucketed by **training** interaction count. Boundaries are contiguous, so every
user lands in exactly one bucket — a test asserts both properties.

| Slice | Rule | Users |
|---|---|---:|
| `users_activity_1-3` | 1–3 training interactions | **25** |
| `users_activity_4-10` | 4–10 | **19,901** |
| `users_activity_11-30` | 11–30 | **24,028** |
| `users_activity_31+` | 31 or more | **6,046** |

Total 50,000. The 1–3 bucket is small because every raw user had ≥6
interactions; those 25 fell below after singleton-item filtering and the removal
of their two held-out events.

## Long-tail items

**Rule:** rank items by training interaction count; the head is the most popular
items whose cumulative interactions first reach **80%** of the training total.
Everything after that point is long tail.

| Slice | Items | Share |
|---|---:|---:|
| `items_head` | **27,799** | 40.5% |
| `items_long_tail` | **40,778** | 59.5% |

The threshold is configurable (`data.slices.long_tail_quantile`) and is recorded
in the slice manifest, so no long-tail number is ever quoted without the rule
that produced it.

The head is guaranteed non-empty: a uniform catalogue would otherwise make every
item long tail and the slice would stop being informative.

## New-item cold start

| Slice | Items |
|---|---:|
| `items_cold_start` | **770** |

Items appearing in validation or test but never in training. Produced naturally
by the leave-last-N protocol, and reported as a warning by leakage check L11.

This is the ceiling on any purely collaborative model: 770 items cannot be
retrieved by LightGCN or matrix factorization at all, because those models have
no representation for them. They are exactly the population Phase 4's content
and multimodal features exist to serve — which makes this slice the honest
measure of whether that work paid off.

## New-user cold start

| Slice | Users |
|---|---:|
| `users_cold_start` | **0** |

Empty **by construction**: under per-user leave-last-N every eligible user keeps
training history, so no user is evaluated without being seen.

The slice is emitted empty rather than omitted, so a future split strategy that
does produce cold users surfaces them without a schema change.

**No new-user scenario is fabricated.** Synthesising one — by hiding random
users' histories — would produce a benchmark that measures a situation this
split does not create. If cold-user evaluation is needed later, it requires a
documented alternative split, reported as such.

## Missing modalities

| Slice | Items |
|---|---:|
| `items_missing_text_features` | **69,347** |
| `items_missing_image_features` | **69,347** |
| `items_missing_both_modalities` | **69,347** |
| `items_both_modalities` | **0** |

Every item, because the 17.3 GB of vectors is not downloaded by default. These
numbers change the moment `--with-features` is used; they are computed from the
aligned index tables, never assumed. See
[`multimodal_feature_alignment.md`](multimodal_feature_alignment.md).

## How Phase 3 should use these

Report every metric **per slice**, not only in aggregate. Specifically:

- A popularity baseline will score well overall and near-zero on
  `items_long_tail`. That gap is the point of building anything else.
- Any collaborative model scores **exactly zero** on `items_cold_start`.
- `users_activity_1-3` is where a sequential model has almost nothing to attend
  to, and where the fallback chain earns its place.

A model that improves the aggregate while regressing on `items_long_tail` has
usually just learned popularity better — which the slices make visible instead
of letting it pass as progress.
