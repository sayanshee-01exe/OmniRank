# Leakage prevention

Implemented in [`src/omnirank/data/leakage.py`](../../src/omnirank/data/leakage.py).
Tested in [`tests/unit/data/test_leakage.py`](../../tests/unit/data/test_leakage.py).

## Why this runs on every build

Leakage is the failure mode that offline metrics cannot detect, **because it
makes them better**. A leaked dataset produces a model that looks excellent
offline and disappoints online, and by then the cause is weeks behind you.

So the checks are not an optional audit. They run inside the pipeline, and a
critical failure aborts the build with a non-zero exit code. A dataset with
critical leakage is worse than no dataset.

## Severity

| Severity | Effect |
|---|---|
| **critical** | Pipeline aborts. The dataset is not written as valid. |
| **warning** | Recorded and reported. Not automatically wrong. |

## The checks

| ID | Check | Severity | What it catches |
|---|---|---|---|
| L01 | No interaction in more than one split | critical | A row duplicated across splits — the label is also a feature |
| L02 | Train precedes validation, per user | critical | A future event left in the training history |
| L03 | Validation precedes test, per user | critical | Validation tuning on test-period behaviour |
| L04 | Train precedes test, per user | critical | The direct train/test ordering violation |
| L05 | Sequence histories are strictly past, per split | critical | A future item in the input, **or the target inside its own input** |
| L07 | Training graph contains only training edges | critical | Validation/test edges in the LightGCN graph |
| L08 | Item popularity matches a training-only recount | critical | Popularity counted over the whole log |
| L09 | User statistics match a training-only recount | critical | User activity counted over the whole log |
| L10 | One mapping resolves every id in every split | critical | Per-split or train-only mappings leaving rows unmappable |
| L11 | Items in held-out splits but not in training | **warning** | Genuine new-item cold start |
| L12 | No split label or target column in a feature table | critical | A `split` column surviving a join — a perfect predictor |

L05 runs once per split, so a full run performs **13 checks**.

### L08 and L09 recount independently

These do not inspect how the statistic was produced; they recompute it from the
training split and compare. Trusting the producing code to have filtered
correctly is exactly the assumption the check exists to test.

### L11 is a warning, deliberately

An item first seen at test time is real cold start, not a bug. PixelRec50K
produces 770 of them. Failing the build would be wrong; hiding them would be
worse, because they bound what any collaborative model can achieve.

## Design decisions that prevent leakage upstream

The checks are a safety net. These prevent the leak in the first place:

**Engagement counters are excluded from features.** `view_number` and friends
are platform-wide lifetime totals with no timestamp, so they cannot be
point-in-time bounded. Using them would attach a 2024 popularity value to a 2012
interaction. They live in `source_metadata` and never reach a feature table.
See [`source_to_canonical_mapping.md`](source_to_canonical_mapping.md).

**Statistics are training-only by construction.** `build_item_popularity` and
`build_user_statistics` filter to `split == "train"` as their first operation,
and `mean_item_popularity` is joined from the training-only popularity table so
a user's taste-for-popular-items feature cannot encode future popularity.

**Sequence histories use strict inequality.** `orders < target_order` is the
whole guarantee, and L05 verifies it.

**Graph edges come from training rows only**, and L07 verifies it — while
correctly *not* flagging a `(user, item)` pair that legitimately appears in both
training and test, since re-interacting with a known item is normal behaviour.

**No feature normalisation is fitted.** Nothing in Phase 2 computes a mean or
scale to apply across splits. The multimodal vectors are fixed pretrained
representations and are not renormalised
([`multimodal_feature_alignment.md`](multimodal_feature_alignment.md)).

## On ID mappings

Mappings are fitted on the **full post-filtering population**, not on training
alone. This is not leakage: a mapping is an identifier registry, not a learned
statistic. It encodes *which entities exist*, which the evaluation protocol
already assumes when it asks a model to rank the catalogue. Fitting per split
would leave held-out rows unmappable — the failure L10 exists to catch.

The distinction that matters: identifier registries may span splits; **learned
statistics may not**, and L08/L09 enforce that.

## Result on PixelRec50K

```text
PASSED — 12/13 checks passed · 0 critical failures · 1 warning
```

The single warning is L11: 770 cold-start items, expected and enumerated.

## Reports

- `reports/data_quality/pixelrec50k/leakage/leakage_report.json`
- `reports/data_quality/pixelrec50k/leakage/leakage_report.md`

## How the checks are validated

Every check has a test that **injects the specific leak it is meant to catch**
and asserts it fires — a target spliced into its own sequence, popularity
counted over all splits, a test edge in the training graph, a training event
reordered past a validation target. A leakage check that has never been shown to
fail on bad data is not evidence of anything.
