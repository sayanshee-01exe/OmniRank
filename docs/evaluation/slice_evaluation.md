# Slice-based evaluation

A single aggregate hides the failures that matter. A model can look respectable
overall while collapsing on sparse users or the long tail.

Implemented in [`slices.py`](../../src/omnirank/evaluation/slices.py), using the
slices Phase 2 already wrote.

## Slicing has two meanings, never mixed

| Kind | Selects | Metric definition |
|---|---|---|
| **user** | which users are averaged over | unchanged |
| **target_item** | which users are averaged over, **by a property of their held-out target** | unchanged |

Both produce a *user average*. In the target-item case the item property picks
the users. Neither filters recommended items — nothing in this codebase does
that, because it would change the metric rather than the population.

Every `SliceResult` carries its `kind`, so a reader never has to guess.

## User-activity slices

By **training** interaction count. Boundaries are contiguous, so every user lands
in exactly one bucket — a test asserts both the partition and the contiguity.

| Slice | Users |
|---|---:|
| `users_activity_1-3` | 25 |
| `users_activity_4-10` | 19,901 |
| `users_activity_11-30` | 24,028 |
| `users_activity_31+` | 6,046 |

## Target-item slices

| Slice | Items | Selects users whose target… |
|---|---:|---|
| `items_head` | 27,799 | is in the popularity head |
| `items_long_tail` | 40,778 | is in the long tail |
| `items_cold_start` | 770 | never appeared in training |

Plus two computed per fit boundary rather than read from a file, because the fit
catalogue differs between selection and final:

- `targets_reachable_warm` — target is in the fit catalogue
- `targets_unreachable_cold` — target is not

## Sample size is always reported

`users` appears on every slice, and slices below **100 users** are flagged
`small_sample: true`. The flag is reported rather than the metric suppressed —
`users_activity_1-3` has 25 users, and its number is interesting as long as
nobody mistakes it for a stable estimate.

## Empty slices are reported, not omitted

`users_cold_start` is **empty by construction** under leave-last-N: every
eligible user keeps training history. It is emitted with `empty: true` and zero
users rather than dropped, so a future split strategy that does produce cold
users surfaces them without a schema change.

**No new-user slice is fabricated.** Synthesising one by hiding random users'
histories would produce a benchmark measuring a situation this split does not
create.

## How to read the results

- A popularity baseline scores well overall and near zero on `items_long_tail`.
  That gap is the entire argument for building anything else.
- Any purely collaborative model scores **exactly zero** on `items_cold_start`.
- `users_activity_1-3` is where a personalised model has almost nothing to work
  with, and where the serving fallback chain earns its place.

A model that improves the aggregate while regressing on `items_long_tail` has
usually just learned popularity better. The slices make that visible instead of
letting it pass as progress.
