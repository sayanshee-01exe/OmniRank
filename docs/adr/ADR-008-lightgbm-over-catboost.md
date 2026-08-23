# ADR-008: LightGBM over CatBoost for the ranking stage

## Status

Accepted — 2026-08-24. Low cost to revisit: both sit behind the `Ranker`
interface.

## Context

The stack permitted "LightGBM **or** CatBoost" for the learning-to-rank stage.
Both are strong gradient-boosting implementations with ranking objectives. The
choice needed making once so the ranker's training and serving paths are not
written twice, and so a decision is not made by whoever writes the code first.

The deciding constraints are the development environment (Apple Silicon, 16 GB,
CPU-only) and the fact that the ranker is on the **request path**, where model
load time and inference latency matter directly.

## Decision

**LightGBM, with `lambdarank` as the objective.** Recorded in
`models.ranker.implementation`, which also accepts `catboost` — the interface
does not care.

Reasons, in order:

1. **Native listwise ranking.** `lambdarank` with query groups is what the
   ranking stage needs. `Ranker.fit(features, labels, groups)` carries `groups`
   precisely for this. CatBoost has `YetiRank`, which is also good; this is
   close to a tie.
2. **Smaller footprint.** The LightGBM wheel and its models are considerably
   smaller, which matters for a 16 GB laptop that also holds embeddings and an
   index, and for artifact size.
3. **Faster CPU inference on numeric features.** The ranking features here are
   dense and numeric — retrieval scores, counts, recencies, price ratios. That is
   LightGBM's strongest case.
4. **Straightforward arm64 build**, needing only `libomp` from Homebrew.

CatBoost's genuine advantage — excellent native categorical handling without
manual encoding — applies least here: high-cardinality ids (`user_id`, `item_id`)
are handled by embeddings in the retrieval stage, not fed raw to the ranker.

## Alternatives considered

**CatBoost.** Better categorical handling, ordered boosting reduces target
leakage, often strong with less tuning. Rejected: its main advantage is not
needed given the feature design, and it is heavier on both counts that matter
here. Remains a valid config value.

**XGBoost.** Mature, `rank:pairwise` available. Rejected: no clear advantage over
LightGBM for this workload, and its ranking support is less ergonomic.

**A neural ranker (DLRM / DCN-style).** Rejected for the first ranker: gradient
boosting is the stronger starting point on tabular features at this data scale,
trains in minutes on CPU, and gives interpretable feature importances — which
matter a great deal when debugging why the ranker prefers something odd. A neural
ranker becomes worth evaluating once the GBDT is tuned and the feature set is
stable.

**Skip ranking; serve retrieval scores directly.** Rejected as an end state, but
noted as the **degraded mode**: if the ranker is unavailable, serving falls back
to aggregated retrieval order rather than failing.

## Consequences

**Positive.** One ranking implementation to build and maintain. Fast CPU
training, so the ranker can be retrained frequently. Feature importances come
free and are genuinely useful for debugging. Small artifacts.

**Negative.** Categorical features need explicit encoding, which is work the
`FeatureBuilder` must do — and must do **identically offline and online**, which
is exactly the training/serving skew risk the `FeatureBuilder` contract exists to
contain. LightGBM needs `libomp` on macOS, an extra install step documented in
the Phase 2 setup. Switching to CatBoost later means re-tuning, though not
re-architecting.
