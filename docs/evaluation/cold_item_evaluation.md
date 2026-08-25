# Cold-item evaluation

Phase 5's purpose is new-item cold start. This document is how that claim is
tested, and how it can fail while every other number looks healthy.

## What "cold" means here

An item with **no interaction in the fitting split**. Not an unpopular item, not
a long-tail item — an item no user in the training data touched.

On PixelRec50K's official split there are **770 such items**, and they are the
held-out target for **880 users**. Every Phase 3–4 model treats those 880 users
as guaranteed misses, because a collaborative model has no representation for an
item nobody interacted with. It is not badly represented; it is absent from
`fit_item_catalogue` and unreachable at any retrieval depth.

The definition is fixed by the split, not by the model. Widening it — counting
"rare" items as cold, or measuring cold recall only over items the model happens
to reach — would produce a nonzero number that means nothing.

## Why warm metrics cannot tell you

A two-tower model can have healthy aggregate NDCG and **zero** cold recall, and
nothing in the aggregate reveals it. Cold items are a small fraction of the
catalogue, so their contribution to a global metric is within noise.

Measured on this project, before the identity residual was gated:

| | warm mean score | cold mean score | top-20 that were warm |
|---|---:|---:|---:|
| With ID residual | +0.3616 | +0.3603 | **100%** |
| Content only | +0.3626 | +0.3603 | 9.1% |

Cold items were not scored badly — they were within 0.0013 of warm items, and
the median best cold item ranked 230th. What excluded them was a *systematic*
advantage: 8,902 warm items each receiving a small consistent nudge from a
learned identity embedding, which is enough to fill twenty slots.

The aggregate NDCG barely moved. Cold recall went from something to exactly
zero. **This is why cold recall is reported independently and never inferred.**

## The evaluation views

| View | Population | Cold targets |
|---|---|---|
| **Strict** | All held-out targets | Counted as misses when not retrieved |
| **Warm** | Targets in the fit catalogue | Excluded |
| **Cold** (`items_cold_start`) | Targets absent from the fit catalogue | The whole population |
| **Unreachable cold** | Cold targets no source could return | Should be empty for a content model |

The last row is the one Phase 5 changed. For every collaborative model
`targets_unreachable_cold` holds 880 users. For the two-tower it holds **0** —
its catalogue covers all 69,347 items, so no test user has a mathematically
unreachable target.

That is a structural result, not a metric improvement: it raises the ceiling
every downstream stage inherits, whether or not the model ranks well today.

## Metrics

Reported at K ∈ {5, 10, 20, 50}, over the cold population only:

```text
Cold Recall@K       did the cold target appear at all
Cold NDCG@K         and how highly
Cold HitRate@K      fraction of cold-target users served
Cold feature coverage    what fraction of cold items are representable
```

Recall@50 matters more here than Recall@20. A content model competing against a
warm catalogue is usually close but edged out, so depth reveals whether the
signal exists at all before the ranking stage compresses it.

## The failure checklist

If cold recall is zero, the ordered diagnosis is:

1. Do cold items exist in the split? (`items_cold_start` slice, non-empty)
2. Are they in the retrieval catalogue? (`catalogue.cold_count > 0`)
3. Are they in the index? (`cold_item_count` in the index manifest)
4. Do they have content? (feature coverage per modality)
5. Is the identity residual gated to zero for them?
6. Do the query and item towers use the same normalisation?
7. Is seen-item filtering removing them?
8. Is the evaluation slice built from the *fit* catalogue, not the full one?

Each is a separate, checkable claim. Working through them is what turned a zero
into a measurement on this project; the answer was (5), and it was visible only
by comparing the gated and ungated encodings directly.

**A zero that survives the checklist is the honest result.** It is not fixed by
redefining "cold".

## Related

- [Missing-modality evaluation](missing_modality_evaluation.md)
- [Strict vs warm evaluation](strict_vs_warm_evaluation.md)
- [Multimodal two-tower](../models/multimodal_two_tower_core.md)
