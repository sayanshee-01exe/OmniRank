# Metric definitions

Implemented in [`metrics.py`](../../src/omnirank/evaluation/metrics.py) and
[`beyond_accuracy.py`](../../src/omnirank/evaluation/beyond_accuracy.py). Every
ranking metric is verified against hand-computed values in
[`tests/unit/evaluation/test_metrics.py`](../../tests/unit/evaluation/test_metrics.py) —
89 cases, none captured from a previous run.

Cut-offs: **K = 5, 10, 20, 50**. Primary metrics: **NDCG@20** and **Recall@20**.

## Conventions

- `k` counts positions, 1-based. A list shorter than `k` is scored as-is, never padded.
- A user with no recommendations scores **0.0** — never `nan`, never excluded.
- A target outside the model's catalogue is simply not retrieved and scores 0.0
  through the ordinary path. No special case, because a special case would hide
  the failure.

## Ranking metrics

**Recall@K** — `hits@K / |relevant|`.

**Precision@K** — `hits@K / K`. The denominator is `K`, not the list length: a
model returning three items where twenty were asked for is penalised for the
seventeen it did not supply, which is the honest accounting when the list is what
gets shown.

**HitRate@K** — 1.0 if any relevant item appears in the top K.

**MRR@K** — reciprocal of the first relevant item's rank, 0.0 if none.

**MAP@K** — mean of `precision@i` over positions holding a relevant item,
normalised by `min(|relevant|, K)` so a user with more relevant items than
positions is not penalised for the shortfall.

**NDCG@K** — `DCG@K / IDCG@K` with the standard `log2(rank + 1)` discount. Ideal
DCG uses the best `min(|relevant|, K)` gains, so a perfect ranking scores exactly
1.0. Relevance is binary (1.0): PixelRec records one undifferentiated implicit
signal, and grading it would mean inventing a preference strength the source
never measured.

## One-positive redundancy — read this before quoting four metrics

PixelRec50K holds out **exactly one item per user**. Under that condition:

```text
Recall@K   == HitRate@K
MAP@K      == MRR@K
Precision@K == Recall@K / K
```

These are algebraic identities, not empirical agreement, and they are asserted by
tests. Reporting all six as though they corroborated one another would be
presenting one measurement four times.

All six are implemented correctly because a future dataset may hold out several
items per user — at which point they genuinely diverge.

## Beyond-accuracy metrics

Accuracy alone rewards learning popularity. These expose the cost.

**Coverage@K** = `unique items recommended / eligible catalogue`.

The denominator is the model's **fit-item catalogue**, not the full dataset
catalogue. Using the larger one would penalise a model for failing to recommend
items it has never seen, which is the cold-start metric's job.

**Novelty@K** = mean of `-log2 p(i)` over recommended items, where `p(i)` comes
from **fit-window** interaction counts only. Computing it over the whole log
would encode which items become popular after the training window.

Zero-count items would make `-log2(0)` infinite, so additive smoothing
(`novelty_smoothing`, default 1.0) is applied and **recorded with the metric**.
Averaging is within-list first, then across users, so every user weighs equally.

**Gini@K** — inequality of exposure counts across the eligible catalogue. 0.0 is
perfectly even; 1.0 is all exposure on one item. Recommenders sit high, and
watching it move is how a diversity intervention is judged.

Eligible items with **zero exposure are included** in the denominator by default
(`gini_includes_zero_exposure`). Excluding them measures inequality only among
the items a model already likes, which flatters a model that ignores the tail.

**Intra-list diversity — unavailable in this phase.**

The literature definition is embedding similarity within a list. PixelRec's
1024-d text and image vectors are two ~8.6 GiB JSON files covering all 408,374
full-PixelRec items; they are not downloaded, so measured feature coverage is
**0.0**. Returning 0.0 for the metric would be indistinguishable from a genuinely
undiverse recommender, so it is marked unavailable with a stated reason instead.
It lands with the multimodal work in Phase 5.

**`category_diversity@K`** is reported in its place: the mean fraction of
distinct categories within a list, over PixelRec's 108-value item tag vocabulary
(99.99% coverage). Deliberately named differently so the two are never conflated.

## Confidence intervals

User-level bootstrap, 1,000 resamples, 95%, fixed seed. Reported for the primary
metrics.

For model comparison, a **paired** bootstrap resamples the same users for both
models and reports the interval on the difference. Pairing removes the
between-user variance that dominates the individual intervals, which is why two
overlapping individual intervals can accompany a delta that excludes zero.

**A delta whose interval contains zero is reported as inconclusive, never as a
win.**
