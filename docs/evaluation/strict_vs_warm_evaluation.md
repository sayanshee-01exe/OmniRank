# Strict and warm-item evaluation

## The problem

Phase 2's leave-last-N split produces held-out items that never appear in
training. Measured on PixelRec50K:

| Boundary | Targets | Warm | Cold | Reachable |
|---|---:|---:|---:|---:|
| validation vs train-only fit | 50,000 | 49,120 | **880** | 98.24% |
| test vs train+validation fit | 50,000 | 49,276 | **724** | 98.55% |

A collaborative model has no representation for a cold item. It **cannot**
retrieve it, no matter how good its ranking is. So a single number conflates two
different questions:

1. How well does the end-to-end system serve users? (cold targets are failures)
2. How good is the collaborative ranking itself? (cold targets are unanswerable)

## Two views, always reported together

### Strict

Every held-out user is evaluated. A target the model could never retrieve counts
as a **miss**.

This is end-to-end system performance without a cold-start content model. It is
the honest headline number, and it is the one that improves when Phase 5's
multimodal retrieval starts serving cold items.

### Warm

Only users whose target is in the model's fit catalogue are evaluated.

This isolates collaborative ranking quality — it answers "when the model *could*
have found it, did it?" It is diagnostic, not a headline.

## The rules

- **Never report warm alone.** Restricting to reachable targets can only raise
  the number; presenting it as the model's performance overstates the system by
  the cold-target rate.
- **Never silently remove unreachable targets.** They stay in the ground truth
  and are classified, not filtered.
- **Always report the cold count and the reachable fraction.** They bound what
  any collaborative model can achieve on this split.

Both views come from the same `EvaluationGroundTruth`; `view="strict"` and
`view="warm"` select the population and nothing else. A test asserts warm is
never below strict for the same model.

## Relationship to the Phase 2 slice

Phase 2 ships `items_cold_start`: **770 distinct items** appearing in a held-out
split but never in training.

That is a different quantity from the row counts above, and the two are not
interchangeable:

- `items_cold_start` counts **items**, across validation and test together.
- 880 / 724 count **target rows**, against a specific fit boundary.

The fit catalogue differs between selection (train) and final (train+validation),
so reachability is recomputed per boundary rather than read from the slice. The
slice is used for item-target slicing; reachability is computed from the fit
catalogue the run actually used.

## Why cold targets are legitimate misses

They are not a data-quality problem and not a bug. A real system meets items it
has never served, and a leave-last-N split over a real interaction log naturally
produces them. Excluding them would measure a system that does not exist.

They are exactly the population that content and multimodal features exist to
serve, which makes the cold slice the honest measure of whether that work paid
off — and the 1.45–1.76% ceiling it imposes on any purely collaborative model is
a Phase 5 target, not a Phase 3 failure.
