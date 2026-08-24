# Popularity baseline

[`src/omnirank/models/baselines/popularity.py`](../../src/omnirank/models/baselines/popularity.py)

## Why popularity is required, not optional

Two reasons, and the second is the one people forget.

**It is the yardstick.** A personalised model that cannot beat "show everyone the
same trending items" is not earning its complexity. A substantial share of
published neural recommender results do not beat well-tuned non-neural baselines
under a common protocol ([ADR-007](../adr/ADR-007-baselines-before-advanced-models.md)),
and without this number in the same codebase, under the same split, with the same
metrics, a later result is unfalsifiable.

**It is the fallback floor.** It is the terminal stage of the serving fallback
chain: no user-side input, no embeddings, no index, no dependency beyond an
item-count table. It can answer when everything else cannot, which is why it
needs only the `data` extra and not `baseline`.

## Variants

**`global_count`** — score is the item's fit-interaction count.

**`time_decay`** — each interaction contributes `0.5 ** (age_days / half_life)`:

```text
score(i) = sum over e in interactions(i) of exp(-ln(2) * age_days(e) / half_life_days)
```

`age_days` is measured against the **maximum timestamp in the fit data**, not the
wall clock. Using "now" would make scores change every time the model was loaded
— irreproducible, and a subtle leak when the fit window ended long ago.

## Deliberate omission: engagement counters

PixelRec ships `view_number`, `thumbup_number`, `favorite_number` and four more.
They look like ideal popularity features and are **not used**, for two
independent reasons:

1. They are the **whole platform's** lifetime totals — tens of millions of users
   — not the 50,000 in this dataset.
2. They carry **no timestamp**, so they cannot be bounded to the fit window.
   Using them attaches an end-of-2024 popularity value to a 2012 interaction.

They are preserved verbatim in `source_metadata` and excluded from the feature
path. See [`../data/leakage_prevention.md`](../data/leakage_prevention.md).

## Interface

Implements `CandidateGenerator`. Public methods take and return **external** ids;
dense indices are used only internally, against the mapping the model was fitted
with, and `require_mapping()` refuses a mismatched one.

- `recommend(user_id, k, context)` — top-k, seen items excluded by default.
- `recommend_batch(user_ids, k)` — the ranking is global and computed once, so
  each user costs only their seen-set scan.
- `score(user_id, item_ids)` — user-independent by construction; unknown items
  score 0.0 rather than raising.

**An unknown user receives the ordinary popularity list.** That is correct for a
non-personalised fallback: there is nothing user-specific to lose, and returning
nothing would leave the fallback chain with no terminal stage.

**Ties break on ascending item id**, so two runs produce identical output.

## Persistence

JSON for configuration and metadata, NPZ for arrays. **No pickle** — an artifact
read from disk should not be able to execute code. Round-trip tests assert
recommendations and scores are identical after loading, and that a corrupted
file, a missing file, and a wrong model type each fail with a clear error.

## Hyperparameter selection (validation only)

The prescribed grid was 7/30/90/365-day half-lives. It was **extended** to
730 and 1825 because PixelRec50K training interactions have a median age of
**485 days** and span **3,788** — a grid stopping at 365 would not have bracketed
the corpus.

The measured result then contradicted that reasoning, which is why the grid was
run rather than argued about: validation targets are each user's *second-to-last*
event and cluster at the recent end (median age **172 days** against the training
reference), so short half-lives win.

Selection is by validation NDCG@20, with Recall@20 as the tiebreak. Full results
are in [`../phase_reports/phase_03_report.md`](../phase_reports/phase_03_report.md).
