# ADR-002: Temporal splitting for interaction data

## Status

Accepted — 2026-08-24.

## Context

A recommendation model predicts what a user will do **next**. Offline evaluation
must therefore reproduce that setting: train on the past, evaluate on the future.

Random splitting of an interaction log breaks this. If user *u*'s events at times
*t₁ < t₂ < t₃* are scattered across train and test, the model can observe *t₃*
while being asked to predict *t₂*. Reported metrics rise substantially, the
model looks like an improvement, and it underperforms online. The gap is not
subtle and it is not detectable from the offline numbers alone — which is what
makes it dangerous.

There is a second, subtler leak: **global popularity signals**. Even with
per-user temporal ordering, a global item-popularity feature computed over the
entire log encodes which items become popular later.

## Decision

**All splitting is temporal, and split integrity is verified rather than
assumed.**

Configured in `data.splitting`:

| Strategy | Rule |
|---|---|
| `temporal_global` (default) | One global cut. Everything before `validation_start` trains. Most faithful to production, where a model is trained once and serves everyone. |
| `temporal_per_user` | Per-user quantile cuts. Keeps sparse users represented in test. |
| `leave_one_out` | Each user's last interaction is the test target. Common in sequential-recommendation literature; kept for comparability with published numbers. |

Supporting rules:

- **Every feature is `as_of`-bounded.** `FeatureStore` and `FeatureBuilder`
  thread an `as_of` timestamp through every signature, so respecting the cut is
  the default and violating it requires deliberately ignoring a parameter.
- **`embargo_days`** drops interactions immediately before each boundary,
  removing near-boundary leakage from events that straddle the cut.
- **`check_split_integrity`** is runnable today and asserts: no interaction in
  two splits, every event inside its window, embargo respected. Phase 2's
  splitter is tested against it.
- **Timestamps are timezone-aware UTC**, enforced by the data contracts. A split
  corrupted by mixed offsets is a leak that looks like clean data.

## Alternatives considered

**Random split.** Standard in generic ML, and simple. Rejected: it leaks the
future, and the resulting metrics are not comparable to online performance. It is
the single most common source of unreproducible recommender results.

**Leave-one-out only.** Widely used in sequential-recommendation papers.
Rejected as the default: it evaluates one prediction per user, so it says little
about list quality at k=20, and it still leaks global popularity unless
timestamps bound it. Kept as an option for literature comparison.

**Group k-fold by user.** Prevents per-user leakage. Rejected: it evaluates
cold-start performance for *every* test user, which is a different and much
harder question than the one production asks.

## Consequences

**Positive.** Offline numbers are comparable to online behaviour. Cold-start is
evaluated honestly, since test-window users may genuinely be new. Split
integrity is machine-checkable.

**Negative.** Metrics are **lower** than randomly split baselines from other
projects and papers — this is correct, but it must be stated whenever a number is
reported, or it reads as underperformance. Test-set size depends on the temporal
distribution, so a bursty log yields an uneven split. Some users appear only in
test, and no personalised model can serve them — which is precisely why the
fallback chain exists and why per-user coverage is a reported metric
(`evaluation.min_user_coverage`).
