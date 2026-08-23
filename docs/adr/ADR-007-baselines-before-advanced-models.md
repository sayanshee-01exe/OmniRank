# ADR-007: Baselines before advanced models

## Status

Accepted — 2026-08-24.

## Context

The target architecture includes LightGCN, SASRec, and a multimodal two-tower
retriever. It is tempting to build the interesting model first.

The argument against is empirical and well documented in the recommender
literature: a substantial share of published neural recommender results do not
beat well-tuned non-neural baselines when both are evaluated under the same
protocol. The usual causes are not fraud — they are an under-tuned baseline, a
different evaluation protocol, or sampled-negative evaluation that flatters the
more complex model.

Without a baseline in the *same* codebase under the *same* split and the *same*
metrics, "LightGCN achieves NDCG@20 = 0.31" is unfalsifiable. It could be
excellent, or it could be worse than counting purchases.

There is a second, operational reason. Popularity is the terminal stage of the
serving fallback chain. Until it exists, the system has **no guaranteed way to
answer a request**, and every other model is built on a serving path that cannot
degrade safely.

## Decision

**Models ship in strict order, and each must beat its predecessor under the same
protocol before the next begins.**

| Order | Model | Phase | Must beat |
|---|---|---|---|
| 1 | Time-decayed popularity | 2 | — (it is the floor) |
| 2 | Implicit matrix factorization | 2 | popularity |
| 3 | LightGCN | 3 | matrix factorization |
| 4 | SASRec | 3 | matrix factorization |
| 5 | Two-tower multimodal | 4 | LightGCN, on cold items specifically |
| 6 | LightGBM ranker | 5 | best single retriever |
| 7 | MMR reranking | 5 | ranker, on diversity at equal NDCG |

Supporting rules:

- **One evaluation protocol for all of them.** Same temporal split
  ([ADR-002](ADR-002-temporal-splitting.md)), same `k_values`, same
  `filter_seen`, same `protocol: full`. Enforced by all models sharing one
  `Evaluator` contract that never calls a model itself.
- **`protocol: sampled` is for development loops only.** It is faster and biased,
  and must never produce a reported number.
- **Beyond-accuracy metrics from the first baseline onward** — coverage, novelty,
  intra-list diversity, gini. Popularity scores well on accuracy and badly on
  coverage; recording both from the start makes that visible instead of
  discovering it after the reranker is built.
- **Every result is registered as an artifact** with its metrics, so comparisons
  are between recorded numbers, not remembered ones.
- **The baseline is tuned before it is beaten.** An untuned baseline is not a
  baseline; it is a strawman.

## Alternatives considered

**Build the interesting models first, baselines later "for the paper".**
Rejected: a baseline built after the fact is invariably under-tuned, and by then
the architecture has been shaped around the complex model. It also leaves the
fallback chain without its terminal stage for several phases.

**Use published benchmark numbers as the baseline.** Rejected: different dataset,
different split, different filtering, different metric definitions. Not
comparable, and comparing against them anyway is how unreproducible claims are
made.

**Skip matrix factorization and go straight from popularity to LightGCN.**
Tempting — one fewer model. Rejected: MF is the standard collaborative baseline
and is cheap to build. If LightGCN cannot beat it, that is the most informative
possible early result, and it is worth far more than the week it saves to skip.

## Consequences

**Positive.** Every claim is comparative and reproducible. The fallback chain
gets its terminal stage first, so the serving path can degrade safely from
Phase 2. Cheap models arrive early, so the end-to-end pipeline is exercised
before the expensive parts exist. The evaluation harness is validated against a
model whose behaviour is easy to reason about.

**Negative.** Slower to a headline result — LightGCN is a phase later than it
would otherwise be. Requires the discipline to actually tune the baseline. And it
creates a real possibility that must be accepted in advance: **a neural model may
fail to beat MF on this dataset**, in which case the honest outcome is to report
that and investigate, not to quietly re-tune the evaluation until it wins.
