# Phase 3 report — offline evaluation and baselines

**Date:** 2026-08-24 · **Dataset:** `pixelrec50k@v1` / split v1 / mapping v1 ·
**Manifest:** `9d1eabb7977d24ac…` · **Status:** complete

---

## Headline

Two baselines were built and measured under one full-catalogue protocol. **The
result reverses between validation and test**, and that reversal is the most
important finding in this phase:

| Stage | Fit → target | Popularity NDCG@20 | BPR NDCG@20 | Winner |
|---|---|---:|---:|---|
| Selection | train → validation | 0.00215 | **0.00331** ± 0.00019 | **BPR** |
| Final | train+validation → test | **0.00407** | 0.00284 | **Popularity** |

On test, the paired bootstrap delta is **−0.00124 NDCG@20 [−0.00175, −0.00076]**
and **−0.00432 Recall@20 [−0.00552, −0.00314]** — both intervals exclude zero, so
popularity's win on test is statistically supported, not noise.

The reversal is explained in §23 and is a property of the data, not a bug. It is
reported rather than resolved in BPR's favour.

---

## 1. Repository state before Phase 3

Phase 2 intact: 679 tests passing, ruff and mypy-strict clean, all 34 manifest
outputs present and checksummed. Two changes since the Phase 2 report:

- **It is now a git repository** (`main`, clean tree). The commit titled
  *"Phase 3 is completed"* actually contained the Phase 2 work; no Phase 3 code
  existed. `models/baselines/` and `evaluation/` held only `__init__.py` and
  `base.py`.
- **1.4 MB of PixelRec-derived tables had been committed** — `user_activity.csv`
  (50,001 rows of real `u…` ids) and `item_popularity.csv` (82,866 rows). The
  Phase 1 `.gitignore` covered `reports/figures/` and `reports/metrics/` but not
  `reports/data_quality/`. Since the licence grants no redistribution rights,
  the rule was extended and the files untracked (they remain on disk).

Reused rather than rebuilt: `core/config.py`, `core/logging.py`,
`core/exceptions.py`, `artifacts/` (registry + metadata), the `data/processed`
tables, `IdMapping`, the Phase 2 evaluation slices, and the Phase 1
`CandidateGenerator` / `Evaluator` / `GroundTruth` contracts — none of which were
weakened.

### Contract findings

**The split tables carry no `timestamp` column** (`internal_user_id`,
`internal_item_id`, `interaction_order`, `event_type`, `interaction_weight`,
`split`). Time-decayed popularity needs real ages. Rather than regenerate
anything, timestamps are joined from `interim/canonical_interactions.parquet` —
itself a manifest-listed, checksummed Phase 2 output. This is not a Phase 2
defect; the processed schema is documented.

**Zero repeated `(user, item)` pairs** in the training split
(`interaction_count` maxes at 1), measured not assumed. The binary-vs-confidence
decision for BPR is therefore moot on this dataset; unique binary positives are
used and the policy is recorded in the artifact.

---

## 2. Files created

**Source (16).** `data/processed.py`; `evaluation/{recommendations,ground_truth,
metrics,beyond_accuracy,evaluator,slices,bootstrap,experiment,reporting}.py`;
`models/baselines/{popularity,negative_sampling,bpr,runner,registry_support}.py`.

**Scripts (1).** `scripts/compare_baselines.py`.

**Tests (10).** `tests/unit/evaluation/{test_metrics,test_recommendations,
test_evaluator,test_ground_truth,test_bootstrap}.py`;
`tests/unit/models/{test_popularity,test_negative_sampling,test_bpr,
test_runner_examples}.py`; `tests/integration/test_baseline_pipeline.py`.

**Docs (10).** `docs/evaluation/{offline_evaluation_protocol,metric_definitions,
strict_vs_warm_evaluation,slice_evaluation,README}.md`;
`docs/models/{popularity,bpr_matrix_factorization,negative_sampling,
model_selection,README}.md`; this report.

## 3. Files modified

`pyproject.toml` (new `baseline` extra; `ml` now includes it), `Makefile`
(7 targets), `.github/workflows/ci.yml` (second job), `README.md`,
`configs/models/retrieval.yaml`, `configs/evaluation/default.yaml`,
`configs/data/ecommerce.yaml`, `configs/serving/local.yaml`,
`core/config.py` (`BootstrapConfig`, `BeyondAccuracyConfig`, model
hyperparameters), `evaluation/base.py`, `models/baselines/__init__.py`,
`models/{lightgcn,sasrec,two_tower}/__init__.py`, `scripts/{train,evaluate}.py`
(rewritten), `docs/architecture/{offline_training_flow,component_contracts}.md`,
`docs/adr/ADR-007-baselines-before-advanced-models.md`,
`tests/integration/test_repository_smoke.py`, `.gitignore`.

## 4. Phase-label corrections

Normalised everywhere to: 1 foundation · 2 data · 3 evaluation+baselines ·
4 LightGCN/SASRec/aggregation · 5 multimodal two-tower · 6 LTR/MMR/serving ·
7 monitoring.

| Location | Before | After |
|---|---|---|
| `retrieval.yaml` popularity, matrix_factorization | 2 | **3** |
| `retrieval.yaml` lightgcn, sasrec | 3 | **4** |
| `retrieval.yaml` two_tower | 4 | **5** |
| `retrieval.yaml` ranker, reranker | 5 | **6** |
| `models/lightgcn`, `models/sasrec` docstrings | 3 | **4** |
| `models/two_tower` docstring | 4 | **5** |
| ADR-007 delivery table | 2/3/4/5 | **3/4/5/6** |
| README roadmap, pipeline diagram, model table | mixed | corrected |
| `evaluation/base.py`, `evaluation/default.yaml` headers | "Phase 2" | Phase 3 |
| `scripts/{train,evaluate}.py` | placeholder text | rewritten |

An enforcement test asserts every ADR still carries its five required sections.

---

## 5. Evaluation architecture

```text
ProcessedDataset (manifest-verified)
        │
        ├─ build_ground_truth ──► EvaluationGroundTruth (external ids, warm/cold)
        │
        └─ model.recommend_batch ──► RecommendationSet (ordered, dup-rejecting)
                                            │
                                    OfflineEvaluator
                                     ├─ strict view
                                     ├─ warm view
                                     ├─ 9 slices
                                     └─ beyond-accuracy
                                            │
                                    bootstrap ──► reporting
```

The evaluator receives finished lists and never calls, reorders, or filters a
model. Metric functions are pure and independently testable.

**Dataset identity is verified before any model sees a row**: schema version 2,
split version 1, mapping version 1, all required files present, and all 34
manifest checksums re-hashed (0.5 s). The identity is recorded in every artifact
and every report.

## 6. Metric definitions

Recall, Precision, NDCG, MAP, MRR, HitRate at K = 5, 10, 20, 50. Primary:
**NDCG@20** and **Recall@20**. Full definitions in
[`../evaluation/metric_definitions.md`](../evaluation/metric_definitions.md).

**One-positive redundancy.** PixelRec holds out exactly one item per user, so
`recall@k == hit_rate@k`, `map@k == mrr@k`, and `precision@k == recall@k / k`.
These are algebraic identities, asserted by tests. All six are implemented
because a future dataset may hold out several items — but they are **not**
independent evidence and are not presented as such.

## 7. Ground-truth construction

Built once, by one function, for every model. External ids at the boundary;
internal ids stay inside models. Records the target split and the fit boundary.

Cold targets are **classified, never removed** — removing them would turn a real
end-to-end failure into an invisible one.

A guard asserts every user's fit history strictly precedes their target, so a
mis-specified fit boundary fails loudly rather than producing an optimistic
number.

## 8–9. Strict and warm protocols

| Boundary | Targets | Warm | Cold | Reachable |
|---|---:|---:|---:|---:|
| validation vs train fit | 50,000 | 49,120 | 880 | **98.24%** |
| test vs train+validation fit | 50,000 | 49,276 | 724 | **98.55%** |

Final test, both views:

| Model | View | Recall@20 | NDCG@20 |
|---|---|---:|---:|
| popularity | strict | 0.01148 | 0.00407 |
| popularity | warm | 0.01165 | 0.00413 |
| BPR | strict | 0.00716 | 0.00284 |
| BPR | warm | 0.00726 | 0.00288 |

The warm/strict gap is small (≈1.5%) because reachability is high. Both are
always reported; warm is never reported alone.

---

## 10–12. Popularity

Two variants: `global_count`, and `time_decay` with
`0.5 ** (age_days / half_life)` measured against the **maximum fit timestamp**
(not the wall clock, which would make scores change on every load).

Engagement counters (`view_number`, `thumbup_number`, …) are **deliberately
unused**: they are platform-wide lifetime totals with no timestamp, so they
cannot be bounded to the fit window.

### Hyperparameter search (validation, full-catalogue, strict)

| Variant | Half-life | NDCG@20 | Recall@20 | Coverage@20 |
|---|---:|---:|---:|---:|
| global_count | — | 0.00038 | 0.00088 | 0.00035 |
| time_decay | 7 | 0.00016 | 0.00058 | 0.00035 |
| time_decay | 30 | 0.00213 | 0.00548 | 0.00034 |
| time_decay | **45** | **0.00215** | 0.00558 | 0.00035 |
| time_decay | 90 | 0.00199 | 0.00564 | 0.00036 |
| time_decay | 365 | 0.00131 | 0.00364 | 0.00036 |
| time_decay | 730 | 0.00089 | 0.00246 | 0.00036 |
| time_decay | 1825 | 0.00055 | 0.00150 | 0.00035 |

The prescribed grid (7/30/90/365) was **extended to 730 and 1825** because
training interactions have a median age of 485 days and span 3,788 — a grid
stopping at 365 would not have bracketed the corpus.

The measurement then **contradicted that reasoning**, which is why the grid was
run rather than argued: validation targets are each user's second-to-last event
and sit at the recent end (median age 172 days vs 485 for training events), so
short half-lives win. Recency matters more than corpus span.

**Selected: `time_decay`, half-life 45 days** (validation NDCG@20 = 0.00215).
30–60 days are within noise of one another; 45 wins by the stated rule.

Time decay beats raw counts by **5.7×** — the single largest effect in the phase.

---

## 13–16. BPR matrix factorization

PyTorch, sparse embeddings with `SparseAdam`, `softplus(-x)` for numerical
stability, L2 over batch-touched embeddings only. CPU and MPS, never CUDA
implicitly. Unique binary positives. Finite-loss and finite-gradient checks abort
training rather than producing a NaN model.

### Negative sampling

Uniform over items the user has not interacted with in the fit data. Guarantees
(all tested): never samples a known positive; same seed gives identical samples;
dense users terminate via an explicit complement draw; a full-catalogue user is
rejected at construction.

**Performance defect found and fixed.** The first implementation tested
membership per row — profiling showed `_isin_sorted` called **1.75 M times per
epoch**, 84% of training time. Encoding `(user, item)` as one `int64` key and
binary-searching the whole batch at once took training from **25 s/epoch to
2.9 s/epoch (8.6×)** with identical loss.

### Hyperparameter search

Convergence measured before choosing a budget (d64, validation):

| Epochs | 5 | 10 | 20 | 30 | 50 | 80 |
|---|---:|---:|---:|---:|---:|---:|
| Final loss | 0.3642 | 0.0855 | 0.0335 | 0.0242 | 0.0184 | 0.0157 |
| NDCG@20 | 0.00105 | 0.00134 | 0.00149 | 0.00156 | 0.00175 | 0.00182 |

Grid at 30 epochs (validation, strict):

| d | lr | reg | neg | NDCG@20 | Recall@20 | Coverage@20 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.005 | 1e-4 | 1 | 0.00145 | 0.00382 | 0.4166 |
| 64 | 0.005 | 1e-4 | 1 | 0.00156 | 0.00398 | 0.4392 |
| 64 | 0.001 | 1e-4 | 1 | 0.00132 | 0.00320 | 0.2739 |
| 64 | 0.005 | 1e-5 | 1 | 0.00155 | 0.00396 | 0.4346 |
| 64 | 0.005 | 1e-4 | 3 | 0.00171 | 0.00412 | 0.4960 |
| 128 | 0.005 | 1e-4 | 1 | 0.00180 | 0.00482 | 0.4271 |

At 30 epochs BPR (0.00180) sat **below** popularity (0.00215). Embedding
dimension, negative count, and epochs were all still improving, so the model was
**under-trained rather than beaten**. The search was extended on validation
before drawing any conclusion:

| Config (80 epochs) | NDCG@20 | Recall@20 | Coverage@20 | Fit |
|---|---:|---:|---:|---:|
| d128 / neg1 | 0.00197 | 0.00530 | 0.4496 | 269 s |
| d128 / neg3 | 0.00227 | 0.00602 | 0.4421 | 415 s |
| **d256 / neg3** | **0.00339** | **0.00848** | 0.4192 | 746 s |

**Selected: d256, lr 0.005, reg 1e-4, neg 3, 80 epochs, batch 8192.**

### Multi-seed (validation, locked config)

| Seed | NDCG@20 | Recall@20 | Fit |
|---:|---:|---:|---:|
| 42 | 0.00339 | 0.00848 | 689 s |
| 43 | 0.00344 | 0.00814 | 720 s |
| 44 | 0.00309 | 0.00764 | 688 s |
| **Mean** | **0.00331 ± 0.00019** | **0.00809 ± 0.00042** | |

Seed 42 (the lowest, a rule fixed before results were seen) is the registered
artifact. Seed 42 reproduced 0.00339 exactly across two independent runs.

---

## 17. Validation metrics (selection stage)

| Model | NDCG@20 | Recall@20 | Coverage@20 |
|---|---:|---:|---:|
| Global popularity | 0.00038 | 0.00088 | 0.00035 |
| Time-decay popularity (45 d) | 0.00215 | 0.00558 | 0.00035 |
| **BPR (d256/neg3/80)** | **0.00331 ± 0.00019** | **0.00809 ± 0.00042** | **0.4192** |

**On validation, BPR beats popularity by 54% on NDCG@20.**

## 18. Final test metrics (evaluated once)

Fit: train+validation. Targets: test. Configuration locked in
`selected_configuration.json` before any test metric was computed.

| Model | Protocol | Recall@20 | NDCG@20 | Coverage@20 | Novelty@20 | Gini@20 | Reachable | Train | Eval |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity (45 d) | strict | **0.01148** | **0.00407** | 0.00035 | 14.398 | 0.9997 | 0.9855 | 0.11 s | 3.5 s |
| Popularity (45 d) | warm | 0.01165 | 0.00413 | 0.00035 | 14.398 | 0.9997 | 0.9855 | 0.11 s | 3.5 s |
| BPR (d256) | strict | 0.00716 | 0.00284 | **0.4181** | 14.320 | **0.8796** | 0.9855 | 721 s | 3.5 s |
| BPR (d256) | warm | 0.00726 | 0.00288 | 0.4181 | 14.320 | 0.8796 | 0.9855 | 721 s | 3.5 s |

All cut-offs, strict:

| Model | K | NDCG | Recall | Precision | MRR | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 5 | 0.00180 | 0.00334 | 0.000668 | 0.00131 | 0.00010 |
| Popularity | 10 | 0.00281 | 0.00642 | 0.000642 | 0.00173 | 0.00019 |
| Popularity | 20 | 0.00407 | 0.01148 | 0.000574 | 0.00207 | 0.00035 |
| Popularity | 50 | 0.00645 | 0.02362 | 0.000472 | 0.00244 | 0.00078 |
| BPR | 5 | 0.00157 | 0.00254 | 0.000508 | 0.00126 | 0.28097 |
| BPR | 10 | 0.00204 | 0.00398 | 0.000398 | 0.00145 | 0.34667 |
| BPR | 20 | 0.00284 | 0.00716 | 0.000358 | 0.00167 | 0.41811 |
| BPR | 50 | 0.00406 | 0.01336 | 0.000267 | 0.00186 | 0.52211 |

## 19. Confidence intervals (user-level bootstrap, 1,000 resamples, 95%, seed 42)

| Model | Metric | Estimate | 95% CI |
|---|---|---:|---|
| Popularity | Recall@20 | 0.01148 | [0.01058, 0.01244] |
| Popularity | NDCG@20 | 0.00407 | [0.00371, 0.00446] |
| BPR | Recall@20 | 0.00716 | [0.00642, 0.00788] |
| BPR | NDCG@20 | 0.00284 | [0.00251, 0.00316] |

## 20. Paired metric deltas (BPR − popularity, same users resampled)

| Metric | Delta | 95% CI | Excludes zero |
|---|---:|---|---|
| ΔRecall@20 | **−0.00432** | [−0.00552, −0.00314] | **yes** |
| ΔNDCG@20 | **−0.00124** | [−0.00175, −0.00076] | **yes** |

Popularity's advantage on test is statistically supported.

## 21. User-activity slices (test, NDCG@20)

| Slice | Users | Popularity | BPR |
|---|---:|---:|---:|
| `users_activity_1-3` | 25 | 0.0 | 0.0 |
| `users_activity_4-10` | 19,901 | 0.00320 | 0.00251 |
| `users_activity_11-30` | 24,028 | 0.00438 | 0.00316 |
| `users_activity_31+` | 6,046 | **0.00571** | 0.00261 |

Popularity's lead **widens with user activity**; BPR peaks in the middle band and
falls back for the most active users. The 25-user slice is flagged
`small_sample`; both models score zero and no conclusion is drawn from it.

## 22. Item-target slices (test, NDCG@20)

| Slice | Users | Popularity | BPR |
|---|---:|---:|---:|
| `items_head` | 28,837 | **0.00647** | 0.00422 |
| `items_long_tail` | 18,893 | 0.00089 | **0.00106** |
| `items_cold_start` | 2,270 | 0.0 | 0.00010 |

**BPR wins on the long tail**, which is the population popularity structurally
cannot serve. Popularity wins decisively on head items, and head targets are the
majority — which is exactly how it wins overall.

## 23. Cold-target analysis — and the validation/test reversal

724 test targets (1.45%) are unreachable against the train+validation fit and are
counted as misses in the strict view. Popularity scores exactly **0.0** on the
Phase 2 `items_cold_start` slice. BPR scores **0.0001** — non-zero because some
of those items appear in *validation*, which is inside the final fit boundary.

### Why the ranking reverses between validation and test

Investigated rather than assumed, following the §36 checklist. Measured
properties of the two target sets:

| Property | Validation targets | Test targets |
|---|---:|---:|
| Distinct items | 23,760 | **20,770** |
| Top-24 items' share of targets | 0.996% | **2.036%** |
| Median age vs fit reference | 172 d | **110 d** |
| Popularity top-20 captures | 0.558% | **1.148%** |

**Test targets are ~2× more concentrated on popular items and considerably more
recent than validation targets.** Each user's *last* interaction is more likely
to be currently-trending content than their second-to-last — a structural
property of the leave-last-N split on a short-video corpus.

That doubling matches popularity's Recall@20 exactly: 0.00558 on validation,
0.01148 on test. **The evaluator is computing precisely what it claims**: recall
equals the measured fraction of targets its top-20 list captures, to the digit.

The evaluation implementation, negative sampling, seen-item masking, id mapping,
loss convergence, score orientation, and validation protocol were each verified
(§36 items 1–7). No defect was found. The reversal is a property of the data.

**Implication:** validation-based selection is only as trustworthy as the
similarity between validation and test target distributions. Here they differ
materially, and the discipline of locking before testing is what made the
discrepancy visible instead of invisible.

## 24. Beyond-accuracy metrics (test, @20)

| Metric | Popularity | BPR |
|---|---:|---:|
| Coverage | 0.00035 (**24 items**) | **0.41811** (28,900 items) |
| Novelty (bits) | **14.398** | 14.320 |
| Gini | 0.9997 | **0.8796** |
| Category diversity | **0.8503** | 0.6800 |

Two of these deserve comment, because the naive reading of each is wrong.

**Novelty is essentially tied** (14.398 vs 14.320) even though popularity shows
24 items and BPR shows 28,900. Novelty is `-log2 p(i)` averaged over the list, and
on a catalogue this sparse even the most popular item is rare in absolute terms —
so the metric barely separates them. It is reported because it was specified, but
**coverage and Gini are the informative pair here**, and they separate the two
models by three orders of magnitude.

**Popularity scores *higher* on category diversity** (0.85 vs 0.68). Its 24 items
happen to span many of the 108 tags, whereas BPR's personalised lists concentrate
within a user's preferred categories — which is what personalisation *should* do.
Category diversity measures within-list tag spread, not personalisation quality,
and a model that gave every user the same 20 tags would score well on it. This is
precisely why it is named `category_diversity` and not `intra_list_diversity`.

**Popularity recommends 24 distinct items to all 50,000 users** — 0.035% of the
catalogue, with a Gini of 0.9997. BPR covers **1,200× more of the catalogue**.

**Embedding-based intra-list diversity is marked unavailable**, not zero:
PixelRec's 1024-d vectors are two ~8.6 GiB JSON files not downloaded in this
phase, so measured coverage is 0.0. Returning 0.0 would be indistinguishable from
a genuinely undiverse recommender. `category_diversity@K` over the 108-value tag
vocabulary is reported in its place, named so the two are never conflated.

## 25–26. Runtime and memory

| Model | Stage | Seconds | Peak Python memory | Throughput |
|---|---|---:|---:|---|
| Popularity | fit | **0.11** | 27.5 MB | 8.3 M interactions/s |
| Popularity | recommend | 0.29 | 28.7 MB | 174,104 users/s · 0.0057 ms/user |
| BPR | fit | **721.5** | 124.9 MB | 1,283 interactions/s |
| BPR | recommend | 17.4 | 30.3 MB | 2,869 users/s · 0.349 ms/user |
| Both | evaluate | ~3.5 | — | 11 passes over 50,000 users |

**BPR costs 6,500× more training time than popularity for a worse test number.**

These are **offline batch throughput**, not serving latency: a vectorised sweep
with no request overhead, network, or per-request model loading. `tracemalloc`
sees Python allocations only — torch tensor memory is not included.

Evaluation was optimised from ~25 s to 3.5 s per experiment by computing each
user's hit positions once instead of rescanning the list 24 times. The fast path
is tested against the reference metric functions on 35 randomised cases, so the
optimisation cannot silently change a reported metric.

Artifact sizes: popularity 3.7 MB, BPR 128 MB (256-d factors over 50,000 users
and 69,347 items).

## 27–28. Artifacts

| Artifact | Version | Size | SHA-256 |
|---|---|---:|---|
| popularity | `phase3-popularity-selection` | 3.6 MB | `400cac6c5b2a29b3…` |
| popularity | `phase3-popularity-final` | 3.7 MB | `492bb91a127b5dbd…` |
| matrix_factorization | `phase3-mf-final` | 128 MB | `be637b95f03368ac…` |

Every manifest records model name/version/type, created_at, training data
version, feature version, configuration hash, random seed, framework versions,
Python version, git commit, metrics (prefixed by the split they were measured
on), supported device, artifact path, artifact checksum, item-mapping checksum,
split version, mapping version, dataset manifest hash, fit splits, and
evaluation protocol.

Experimental trials are **not** registered — only the selection-stage popularity
model and the two final models. All 17 selection trials are recorded in
`validation_runs.jsonl` instead, so a configuration that did badly cannot be
quietly omitted.

## 29–30. Mapping compatibility and save/load verification

All artifacts carry item-mapping checksum `235fcff3343a6511…`, matching the
Phase 2 mapping metadata. `require_mapping()` raises on a mismatch — a model
paired with the wrong mapping would resolve every recommended id to a different
item and fail silently.

Save/load verified three ways:

| Check | Result |
|---|---|
| `train.py --stage final` vs `compare_baselines.py --stage final` | popularity 0.004072 / 0.01148, BPR 0.002836 / 0.00716 — **identical** |
| `evaluate.py` reloading the frozen artifact | **identical** to both |
| BPR trained on **MPS**, reloaded on **CPU** | **identical** — device-neutral persistence confirmed |

Unit tests additionally assert recommendations are identical and scores match to
1e-9 after loading, and that corrupted files, missing files, wrong model types
and unsupported format versions each fail with a clear error. `torch.load` uses
`weights_only=True`; popularity uses NPZ + JSON with no pickle.

## 31. Tests added

**332 new tests** (679 → **1,011**), all offline, CPU-only, no network, no GPU.

| File | Tests | Covers |
|---|---:|---|
| `evaluation/test_metrics.py` | 89 | Every metric against hand-computed values; one-positive identities |
| `evaluation/test_evaluator.py` | 50 | Denominators, strict/warm, non-interference, fast-path vs reference |
| `evaluation/test_bootstrap.py` | 21 | Determinism, paired deltas, invalid inputs |
| `evaluation/test_recommendations.py` | 18 | Order, duplicate rejection, determinism |
| `evaluation/test_ground_truth.py` | 12 | Warm/cold classification, leakage guard |
| `models/test_bpr.py` | 52 | Training, retrieval, unknowns, persistence, device policy |
| `models/test_popularity.py` | 32 | Decay arithmetic, tie-breaking, train-only, persistence |
| `models/test_negative_sampling.py` | 18 | No false negatives, determinism, dense users |
| `models/test_runner_examples.py` | 14 | Deterministic, anonymised, not a highlight reel |
| `integration/test_baseline_pipeline.py` | 8 | Full fixture workflow, save/load/re-evaluate |

Every metric value is hand-computed, not captured from a previous run.

Three defects were found *by* these tests and fixed: BPR's loss history was
rounded on save so it did not round-trip; the examples helper labelled users by
position, so two different samples could render identically and no cross-model
comparison was possible; and the evaluator's fast path needed a randomised
agreement test against the reference metric functions before it could be trusted.

## 32–35. Commands and validation results

```bash
uv pip install -e ".[baseline,dev]"
ruff format src tests scripts && ruff check src tests scripts
mypy
pytest
python scripts/compare_baselines.py --stage selection --epochs 30 --device mps
python scripts/compare_baselines.py --stage final --device mps
python scripts/train.py --model popularity --stage final \
    --version phase3-popularity-final --from-selection
python scripts/train.py --model matrix_factorization --stage final \
    --version phase3-mf-final --from-selection --device mps
python scripts/evaluate.py --model matrix_factorization \
    --version phase3-mf-final --split test --protocol full
```

| Gate | Result |
|---|---|
| `ruff format --check` | **138 files already formatted** |
| `ruff check` | **All checks passed** |
| `mypy` (strict) | **Success — no issues in 138 source files** |
| `pytest` | **1,011 passed** |

## 36. CI changes

Two jobs. **Core** installs `.[data,dev]` — no torch — and runs ruff, mypy, the
full suite, the OpenAPI build, and an explicit assertion that the evaluator and
popularity import **without torch in `sys.modules`** (popularity is the terminal
fallback stage and must work on a lightweight install).

**Baselines** installs `.[baseline,dev]` — torch only, not
sentence-transformers / faiss / lightgbm / mlflow — and runs the metric tests
(which fail if a hand-computed value changes), the popularity and sampler tests,
the BPR tests (which fail if save/load output changes), and the fixture
integration test.

No PixelRec download, no GPU, no model weights, no full-dataset training in CI.

## 37. Known limitations

1. **BPR's NDCG@20 was still rising with embedding dimension** when the search
   stopped (d64 → d128 → d256 gave 0.00156 → 0.00197 → 0.00339). d256 is the
   best configuration **searched**, not a demonstrated optimum.
2. **The validation/test reversal is diagnosed but not solved.** A selection
   protocol robust to it — for example rolling or multi-fold temporal validation
   — is a Phase 4 concern.
3. **Popularity's 45-day half-life was tuned on validation**, whose targets are
   older than test's. A recency parameter tuned on a different recency
   distribution is a known fragility.
4. **Multimodal coverage is 0.0**; embedding-based intra-list diversity is
   unavailable.
5. **MLflow not implemented.** The filesystem registry plus the JSON/CSV reports
   cover the requirement; MLflow remains optional and unbuilt.
6. **Sampled-negative evaluation is configurable but unimplemented** — full
   catalogue is fast enough (3.5 s) that it was never needed. No sampled result
   appears anywhere.
7. **No cross-platform numerical equivalence claim.** Determinism is verified
   within a platform for a fixed seed. MPS and CPU *training* were not asserted
   bitwise-identical because they are not; MPS→CPU *inference* was verified.
8. **`users_activity_1-3` has 25 users** — flagged `small_sample`; both models
   score zero and no conclusion is drawn from it.
9. **BPR multi-seed was run on validation only.** The final test artifact is a
   single seed (42, by a pre-declared rule).

## 38. Technical debt

- `compare_baselines.py` is a long linear script; selection and final would read
  better as separate modules.
- `runner.py` carries `Any` where popularity and BPR are used polymorphically. A
  shared `Baseline` protocol would type them properly.
- The `--from-selection` path re-reads the locked JSON in both `train.py` and
  `compare_baselines.py`; one loader would be better.
- BPR's item factors are dense over the full catalogue even for items outside the
  fit set; masked at retrieval, but memory could be reduced.
- `recommendation_examples` draws successes and failures from per-model
  populations, so pseudonyms only align across models in the neutral `sampled`
  group.

## 39. Honest model conclusion

**On the final test benchmark, time-decayed popularity beats BPR matrix
factorization**, and the paired bootstrap supports it: ΔNDCG@20 = −0.00124
[−0.00175, −0.00076], ΔRecall@20 = −0.00432 [−0.00552, −0.00314], both excluding
zero.

This is not the outcome the phase was hoping for, and it is not adjusted. The
§36 checklist was worked through — evaluation implementation, negative sampling,
seen-item masking, id mapping, loss convergence, score orientation, validation
protocol — and no defect was found. The protocol was not altered to improve the
number: no sampled negatives, no warm-only headline, no reduced catalogue.

The full picture is more interesting than the headline:

- **BPR won on validation** by 54% (0.00331 vs 0.00215) and lost on test. The
  cause is measured: test targets are ~2× more concentrated on trending items.
- **BPR wins on the long tail** (0.00106 vs 0.00089) — the 18,893 users
  popularity structurally cannot serve, where it scores near zero.
- **BPR covers 1,200× more catalogue** (41.8% vs 0.035%) with a far lower Gini
  (0.88 vs 0.9997). Popularity shows **24 items to 50,000 users**.
- **Popularity costs 0.11 s to fit; BPR costs 721 s** — 6,500× more.

So: on this dataset, under this split, at K=20, a recency-weighted popularity
list is a strong and extremely cheap baseline that a first-cut BPR does not beat
on aggregate accuracy. Its aggregate strength comes almost entirely from head
items, and it is useless for the long tail. A production system built on it alone
would be cheap, accurate on trending content, and incapable of personalisation.

ADR-007 is vindicated: had the baseline been skipped, BPR's 0.00284 would have
been reported as a result with nothing to judge it against.

## 40. Recommended Phase 4 scope

1. **Fix the selection protocol first.** The validation/test distribution gap is
   the most important finding here. Rolling-origin or multi-fold temporal
   validation would make selection robust before more models are added.
2. **A popularity + BPR hybrid is the obvious next baseline** — popularity owns
   the head, BPR owns the tail, and the aggregation contract
   (`CandidateAggregator`) already exists. Worth measuring before LightGCN,
   because it may capture most of the available gain.
3. **LightGCN**, measured against both baselines under the identical evaluator,
   with the long-tail and cold-start slices reported from the first run.
4. **SASRec**, using the Phase 2 sequential datasets already built.
5. **Candidate aggregation and the FAISS index**, replacing brute-force
   retrieval.
6. **Extend the BPR sweep** past d256 to establish where it actually plateaus —
   cheap, and it settles whether d256 was near the ceiling.

**Phase 4 exit criterion:** LightGCN and SASRec registered with real metrics from
the same evaluator, reported per slice, against both Phase 3 baselines — and a
selection protocol whose validation ranking survives contact with test.
