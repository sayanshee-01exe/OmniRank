# Offline evaluation protocol

The rules every reported OmniRank number obeys. Implemented in
[`src/omnirank/evaluation/`](../../src/omnirank/evaluation/).

## Full-catalogue evaluation is the primary protocol

For each user the model scores **every item it can legitimately recommend**,
already-seen items are removed, and the top *K* are kept.

Why this and not sampled negatives: ranking a positive against 100 sampled
items answers a different, much easier question than ranking it against 69,347
real ones, and the two numbers differ by an order of magnitude. Published
recommender results are frequently not comparable for exactly this reason. The
sampled protocol exists in the config for development speed and is **never** used
for a reported result — any report produced with it is labelled with the negative
count and sampling strategy.

### Memory

A 50,000 × 69,347 score matrix is 13.9 GB in float32. It is never built.
Retrieval scores one **user batch** at a time (`evaluation_user_batch_size`,
default 512) and reduces immediately with `torch.topk`, so peak memory is set by
the batch size rather than the population. A test asserts the batched result is
identical to the naive one, and that batch size does not change the output.

## The denominator

**Every user in the ground truth counts.** A user who received no
recommendations scores zero; they are not dropped.

Dropping them is the most common way an offline number ends up flattering a
model that cannot serve part of its traffic — a model answering 60% of users
perfectly would post a perfect score. `user_coverage` is reported alongside every
metric so the two facts travel together.

## Seen-item exclusion

Items in the model's **fit** interactions are removed before the top *K* is
taken (`filter_seen: true`). The fit boundary defines this, which is why it is
always explicit.

## Two views, always reported together

| View | Population | Answers |
|---|---|---|
| **strict** | every held-out user | end-to-end system performance |
| **warm** | users whose target is in the fit catalogue | collaborative ranking quality |

Neither is reported alone. See
[`strict_vs_warm_evaluation.md`](strict_vs_warm_evaluation.md).

## Validation and test discipline

```text
selection   fit: train              targets: validation   history: train
final       fit: train+validation   targets: test         history: train+validation
```

The boundary follows the split rather than a flag — `scripts/evaluate.py`
derives it from `--split`, so an optimistic combination cannot be requested by
accident.

Hyperparameters are chosen on **validation only**. The chosen configuration is
written to `reports/metrics/phase_03/selected_configuration.json` *before* any
test metric is computed; `compare_baselines.py --stage final` refuses to run
without that file. The final models are refitted from a clean initialisation and
test is evaluated **once**.

Re-running test evaluation against the identical frozen artifact for software
verification is legitimate and is distinguished from selection in the reports.

## The evaluator never touches the model

`OfflineEvaluator` receives finished recommendation lists. It does not reorder,
filter, extend, or call back into a model. A metric that can re-rank the thing it
measures is not a measurement.

## Determinism

Metrics are pure functions of (recommendations, ground truth). Bootstrap
resampling is seeded. Recommendation generation is deterministic given a fitted
model, and tie-breaking is by item id. The evaluator's single-scan fast path is
tested against the reference metric functions on randomised inputs, so the
optimisation cannot silently change a reported number.

## What is measured, and what is not

Runtime is reported per stage — fit, recommend, evaluate — with peak Python
memory where `tracemalloc` can see it (torch tensor memory is not included, and
is labelled as such).

Recommendation throughput is **offline batch throughput**: a vectorised sweep
over the whole population with no request overhead, no network, and no
per-request model loading. It is not a serving-latency figure and is never
quoted as one.
