# Phase 4 report — LightGCN, SASRec, aggregation and FAISS retrieval

**Date:** 2026-08-25 · **Dataset:** `pixelrec50k@v1` / split v1 / mapping v1 ·
**Manifest:** `9d1eabb7977d24ac…` · **Status:** complete

---

## Headline

Two advanced retrievers were implemented and measured under the same
full-catalogue protocol as the Phase 3 baselines, through the same evaluation
driver.

**Both beat the Phase 3 baselines on validation, and LightGCN wins.**

| Model | Validation NDCG@20 | Recall@20 | Fit cost |
|---|---:|---:|---:|
| **LightGCN**, `L=3`, 30 epochs | **0.00612** | **0.01580** | ~8 min |
| SASRec, `L=50`, 45 epochs | 0.00434 | 0.01130 | ~90 min |
| BPR, `d=256`, 80 epochs (Phase 3) | 0.00339 | 0.00848 | ~12 min |
| Popularity, 45-day half-life (Phase 3) | 0.00215 | 0.00558 | seconds |

**On test, LightGCN scores 0.00605 (mean of 3 seeds, sd 0.000053) — the first
model in this project to beat popularity's 0.00407 there** — and its validation
and test scores differ by 1.2%, so the validation/test reversal that defined
Phase 3 does not recur.

**But the largest gain in the phase is not any single model.** Fusing all four
retrievers with reciprocal rank fusion scores **0.00914** on validation — 1.49×
the best single retriever — because the four return almost entirely disjoint
lists (mean 1.07 sources per retrieved item, every pairwise Jaccard under 0.05).

The findings that carry the phase are an ablation, a near-miss, a non-reversal,
and an ensemble whose members barely overlap.

**LightGCN is a clear improvement, and the improvement is attributable to the
graph.** The decisive evidence is the `num_layers = 0` ablation: with
propagation disabled, LightGCN *is* matrix factorization — same code, same data,
same objective, same sampler, same seed. Turning propagation on is the only
thing that changes.

| `num_layers` | Validation NDCG@20 | vs 0 layers | Recall@20 | Coverage@20 |
|---:|---:|---:|---:|---:|
| 0 (= matrix factorization) | 0.00239 | 1.00× | 0.00612 | 0.512 |
| 1 | 0.00499 | 2.09× | 0.01256 | 0.475 |
| 2 | 0.00584 | 2.44× | 0.01496 | 0.448 |
| **3** | **0.00612** | **2.56×** | **0.01580** | 0.434 |

For reference, Phase 3's best BPR configuration scored **0.00339** on the same
validation split, using twice the embedding width (d256 vs d128) and nearly
three times the epochs (80 vs 30). LightGCN at 3 layers is **1.81×** that.

**Coverage falls monotonically as layers increase** (0.512 → 0.434). Propagation
concentrates recommendations, and the accuracy gain is bought partly with
diversity. That trade-off is reported, not netted out.

**SASRec was nearly reported as a failure, and would have been wrong.** Its
first result — 0.00164, below every other model measured — came from a run whose
training loss was still falling steeply when the epoch budget ended. Extending
it to 45 epochs, changing nothing else, produced **0.00434: a 2.64× improvement
from training alone**, overtaking both Phase 3 baselines. The full account is
below; the transferable lesson is in
[`model_selection.md`](../models/model_selection.md).

### Where LightGCN stopped, and why that is a caveat not a result

NDCG@20 was **still rising at `num_layers = 3`**, the deepest configuration
searched. Three layers is therefore *the best configuration searched*, not a
demonstrated optimum — the same caveat Phase 3 recorded when BPR's NDCG was
still improving with embedding width at the end of its search.

Deeper propagation was not run because the marginal gain was shrinking
(2.09× → 2.44× → 2.56×) while cost grows roughly linearly with layers, and the
budget was better spent giving SASRec a fair trial. That is a resource decision,
and it is recorded here rather than presented as a finding about depth.

---

## Cost, measured before the search was designed

Throughput was measured on the real training data *before* choosing grid sizes,
because the affordable search shape depends entirely on it.

| Model | Configuration | Seconds/epoch | Device |
|---|---|---:|---|
| LightGCN | `d=64, L=2` | 11.0 | MPS |
| LightGCN | `d=128, L=3` | 20.0 | MPS |
| SASRec | `L=20, d=64, blocks=2` | ~56 | MPS |
| SASRec | `L=50, d=64, blocks=2` | 103.1 | MPS |
| SASRec | `L=50, d=128, blocks=2` | 189.7 | MPS |

Evaluation adds roughly 20 seconds per configuration: generating
recommendations for all 50,000 users against the 69,347-item catalogue takes
~17s, and the strict, warm, and nine-slice evaluations ~4s.

**A SASRec epoch costs five to ten times a LightGCN epoch.** The Phase 4 brief
specified a staged search across two folds and three seeds for both models; at
these rates that is a multi-day job for SASRec alone. The search was scaled to
the budget and the reduction is stated explicitly below rather than absorbed
silently.

---

## What was built

| Component | Module | Notes |
|---|---|---|
| LightGCN | `models/lightgcn/model.py` | Symmetric-normalised propagation, equal layer weights, BPR objective shared with Phase 3 |
| SASRec | `models/sasrec/model.py` | Causal self-attention, `padding_id = num_items`, sampled BCE |
| Weighted round robin | `retrieval/aggregation.py` | Interleave by weight; compares nothing across sources |
| Reciprocal rank fusion | `retrieval/aggregation.py` | `w_s / (c + rank_s(i))`, `c = 60` |
| Normalised score union | `retrieval/aggregation.py` | `min_max`, `z_score`, `rank_percentile` |
| Blended retriever | `retrieval/blended.py` | Presents a fusion as one `CandidateGenerator` |
| FAISS index | `retrieval/faiss_index.py` | Flat/HNSW/IVF, exactness-checked, identity-enforced |
| Retrieval diagnostics | `retrieval/diagnostics.py` | Candidate recall and source overlap |
| Rolling temporal folds | `data/rolling.py` | Offsets `(3, 2)`; offset 1 reserved and refused |
| Sampled-negative protocol | `evaluation/sampled.py` | Refused at the `final` stage by construction |

Every model and every blend is scored by `run_experiment` from the Phase 3
runner, unchanged. That is deliberate: a comparison in which each system brought
its own harness would be comparing harnesses as much as models.

### Verification worth naming

Three checks did real work rather than confirming the obvious.

**LightGCN's normalisation, against hand arithmetic.** For the two-user
two-item graph, `A[u0,i0] = 1/√(2·1) = 0.7071` and `A[u0,i1] = 1/√(2·2) = 0.5`,
matching the implementation exactly, with the matrix symmetric and both diagonal
blocks empty. A wrong normalisation still trains and still returns plausible
recommendations — it is simply a slower matrix factorization, and no metric
would say so.

**SASRec's causality, by perturbation.** Changing the last item of a sequence
leaves every earlier hidden state **bit-identical** (max delta 0.0), while the
changed position itself moves (delta 2.06). A future-leaking transformer trains
faster and scores better on every offline metric; nothing else in the suite
would catch it.

**FAISS against exact brute force.** `flat_ip` reproduces brute force in both
set and order, maximum score difference 5.7e-06 from float32 accumulation. An
index built with the wrong metric or over a transposed matrix still returns *k*
neighbours with plausible scores and raises nothing.

### Rolling folds, validated against the official split

`fold_offset_2` reproduces the Phase 2 validation split exactly — 875,976
history rows and 50,000 targets. That equality is what makes the other folds
trustworthy: the fold builder is not an approximation of the split logic, and at
the offset where the two should coincide, they do. `fold_offset_3` yields
825,976 history rows and 49,999 targets, with one user excluded for insufficient
history.

---

## Phase 3 configuration drift, corrected

Four inconsistencies between the Phase 3 record and the code were found and
fixed. None changed a reported metric; all of them would have made a later
change harder to reason about.

| Drift | Correction |
|---|---|
| `AggregationConfig.strategy` allowed `score_union`, which no aggregator implements | Literal now lists the three implemented strategies |
| `retrieval/base.py` header said "PHASE 1 STATUS: contracts only. Both land in Phase 2-3." | Both contracts are implemented; header says so |
| `models/base.py` said "No concrete generator or ranker exists yet" | Five generators exist; `Ranker` still has none |
| `system_architecture.md` listed LightGCN and SASRec as Phase 3, two-tower as Phase 4 | Corrected to Phase 4 and Phase 5 |
| `retrieval.yaml` weighted `two_tower`, which is not implemented until Phase 5 | Removed; a weight for a generator that cannot run does nothing |

A fifth was found by the tests rather than by reading: `contributions` in
`AggregationResult` was computed differently by different aggregators — round
robin counted emitted candidates, RRF and score-union counted pool nominations.
Since the field exists to attribute recall drops, only the emitted count answers
the question. All three now share one definition, and the docstring states it.

---

## Tests added

| Suite | Tests | What it protects |
|---|---:|---|
| `unit/retrieval/test_aggregation.py` | 48 | Fused arithmetic, determinism, degradation reporting |
| `unit/models/test_sasrec.py` | 44 | Causality, padding exclusion, sequence encoding, persistence |
| `unit/models/test_lightgcn.py` | 39 | Normalisation constants, layer averaging, isolated nodes, graph checksum |
| `unit/retrieval/test_faiss_index.py` | 31 | Exactness vs brute force, identity enforcement, bounded exclusion |
| `unit/evaluation/test_sampled.py` | 29 | Final-stage refusal, pool determinism, no seen item as a negative |
| `unit/data/test_rolling.py` | 28 | Reserved offset refusal, fold integrity, no future leakage |
| `unit/retrieval/test_blended.py` | 21 | Over-retrieval, provenance, batch/single equivalence |
| `unit/retrieval/test_diagnostics.py` | 18 | Candidate recall and overlap against hand arithmetic |
| `integration/test_retrieval_pipeline.py` | 12 | The seams: model → index, model → blend, round trip |
| `unit/retrieval/test_runner.py` | 6 | Sequence loading fails with an actionable message |
| **Total** | **276** | |

Two of these closed gaps rather than covering new code. `evaluation/sampled.py`
shipped in the Phase 4 preflight with **no tests at all**, despite existing
specifically to stop a fast protocol becoming a reported number — the refusal it
enforces was itself unenforced. `retrieval/runner.py` had none either.

Suite total: **1,296 passing**, ruff and mypy-strict clean.

### A crash the new tests exposed

Running the model and retrieval suites in one process **aborted the interpreter**
— `OMP: Error #15`. `faiss-cpu` and `torch` each bundle their own
`libomp.dylib`, and the LLVM OpenMP runtime aborts when the second initialises.
Import order does not help; both copies load either way.

This was not a test artefact. It would have hit every real path that builds an
index from a torch model's embeddings, which is every path Phase 4 cares about.

The fix sets `KMP_DUPLICATE_LIB_OK` and pins FAISS to one OpenMP thread, inside
`_require_faiss` where faiss is imported. LLVM documents that flag as unsafe and
warns it "may silently produce incorrect results" — which is exactly the reason
the exact-brute-force test matters: it runs *after* torch has initialised and
would fail on any such corruption. The claim is verified per run, not assumed.

### A stale test the change surfaced

`test_only_bpr_requires_the_modelling_extra` asserted against the in-process
`sys.modules`, so it silently depended on no earlier test having imported faiss.
The new retrieval suites broke that assumption. It now runs in a subprocess —
the pattern the neighbouring test in the same file already used for the same
reason — and covers torch and faiss rather than only the heavy extras.

---

## SASRec: under-trained, not beaten

The 15-epoch grid ranked SASRec below every other model measured, including
Phase 3's popularity baseline:

| Configuration | Validation NDCG@20 | Recall@20 | Coverage@20 | Final loss |
|---|---:|---:|---:|---:|
| `L=20, d=64, 15 epochs` | 0.00151 | 0.00398 | 0.305 | 0.584 |
| `L=50, d=64, 15 epochs` | 0.00164 | 0.00452 | 0.292 | 0.471 |

**That comparison was not sound, and the loss curve said so.** At 15 epochs the
training loss was still falling roughly 5% per epoch with no sign of a plateau:

```text
epoch 13  0.526      epoch 14  0.496      epoch 15  0.471
```

The run ended because the epoch budget ended, not because the model converged.
This is the signature Phase 3 recorded for BPR, where the project's own
selection record says:

> The initial 30-epoch grid ranked BPR below popularity (0.00180 vs 0.00215).
> The grid showed embedding_dim, negatives_per_positive and epochs all still
> improving, i.e. the model was **under-trained rather than beaten**, so the
> search was extended on validation before drawing any conclusion.

So the search was extended, on validation, before concluding anything.

### The extended trial

`L=50, d=64` re-run at **45 epochs**:

| Budget | Final loss | Validation NDCG@20 | Recall@20 |
|---:|---:|---:|---:|
| 15 epochs | 0.471 | 0.00164 | 0.00452 |
| **45 epochs** | **0.223** | **0.00434** | **0.01130** |

**A 2.64× improvement from training alone**, with no hyperparameter change. The
model went from the worst measured to third of five, overtaking both Phase 3
baselines:

| Model | Validation NDCG@20 |
|---|---:|
| LightGCN, `L=3`, 30 epochs | **0.00612** |
| SASRec, `L=50`, 45 epochs | 0.00434 |
| BPR, `d=256`, 80 epochs (Phase 3) | 0.00339 |
| Popularity, 45-day half-life (Phase 3) | 0.00215 |
| SASRec, `L=50`, 15 epochs | 0.00164 |

Had the 15-epoch number been reported as SASRec's result, this phase would have
concluded that sequential modelling does not work on this corpus. It does. It
was simply never trained.

### What the extended trial does and does not settle

**Settled:** SASRec is a viable retriever on PixelRec50K and beats both Phase 3
baselines. The earlier ranking was an artefact of the budget.

**Not settled:** whether SASRec can reach LightGCN. It closed most of the gap
but did not overtake, and at 45 epochs its loss was still creeping down
(0.228 → 0.223 over the last two epochs). Four of its five hyperparameter axes
remain unsearched.

**The comparison is now skewed the other way**, and that is worth stating
plainly: SASRec received 45 epochs to LightGCN's 30. It still lost. Since a
SASRec epoch costs roughly 120 s against LightGCN's 16 s, LightGCN reached a
higher score for about **one-eleventh of the wall-clock cost** — 8 minutes
against 90. On this corpus, at this budget, the graph is the better investment.

---

## Final test metrics — and the Phase 3 reversal does not recur

The locked LightGCN configuration was refit on train+validation from a clean
initialisation and the test split evaluated. Beside the Phase 3 test numbers:

| Model | NDCG@20 | Recall@20 | Coverage@20 | Novelty@20 |
|---|---:|---:|---:|---:|
| **LightGCN**, `L=3` (Phase 4, mean of 3 seeds) | **0.00605** | **0.01487** | **0.432** | 14.31 |
| Popularity, 45-day half-life (Phase 3) | 0.00407 | 0.01148 | 0.000347 | 14.40 |
| BPR, `d=256` (Phase 3) | 0.00284 | 0.00716 | 0.418 | 14.32 |

Three seeds, each refit from a clean initialisation on train+validation:

| Seed | NDCG@20 | Recall@20 | Coverage@20 |
|---:|---:|---:|---:|
| 42 | 0.006108 | 0.01478 | 0.4319 |
| 43 | 0.006014 | 0.01482 | 0.4316 |
| 44 | 0.006017 | 0.01502 | 0.4318 |
| **mean** | **0.006046** | **0.014873** | **0.4318** |
| sd | 0.000053 | 0.000129 | 0.00016 |

The spread is 0.9% of the mean, and **the worst seed still beats popularity by
1.48×**. The result does not depend on initialisation.

**LightGCN is the first model in this project to beat popularity on test.** It
does so by 1.48× on NDCG@20 and 1.30× on Recall@20 — while covering 1,244× more
of the catalogue (0.432 against 0.000347; popularity's entire test output is a
couple of dozen items shown to all 50,000 users).

### The part that matters more than the win

| Model | Validation NDCG@20 | Test NDCG@20 | Change |
|---|---:|---:|---|
| **LightGCN** | 0.00612 | **0.00605** | **−1.2%** |
| Popularity (Phase 3) | 0.00215 | 0.00407 | +89% |
| BPR (Phase 3) | 0.00339 | 0.00284 | −16% |

Phase 3's headline finding was that its model ranking **reversed** between
validation and test: BPR won on validation, popularity won on test, and the
paired bootstrap confirmed the reversal was not noise. That left an unresolved
question about which number to believe.

LightGCN's validation and test scores differ by 1.2% — smaller than the gap
between its own best and worst seed is large. It wins on *both* splits, against
*both* baselines. Whatever property of the test week inflated popularity
and depressed BPR, LightGCN is not sensitive to it — the reversal does not
recur, and the ambiguity Phase 3 ended on does not apply to this result.

This is **not** an explanation of the Phase 3 reversal. That still needs the
rolling folds, which were built and verified but not spent (see below). It is
the narrower and still useful observation that the Phase 4 winner does not
depend on which split it is read from.

**No paired bootstrap.** Phase 3 reported paired confidence intervals against a
common user resampling, which is what established its reversal as real rather
than noise. Phase 4 reports a seed spread instead. The margins here are wide
(1.48× over popularity, against a 0.9% seed spread), so the conclusion is not in
doubt, but the interval is not the same statistic and is not a substitute for
it.

---

## FAISS index

The index over LightGCN's 69,347 × 128 item embeddings was verified against
exact brute force before being written: **order agreement 1.0**, set overlap
1.0, maximum score difference 5.7e-06 — float32 accumulation order, not
disagreement. It is registered at
`artifacts/indexes/pixelrec50k/lightgcn/phase4-lightgcn-final` with the model
name, version, embedding checksum and item-mapping checksum attached.

### Approximate indexes, measured against exact

2,000 queries, single-threaded (see [ADR-009](../adr/ADR-009-faiss-torch-openmp-coexistence.md)):

| Index | Build | µs/query @20 | Recall vs exact @20 | @100 |
|---|---:|---:|---:|---:|
| `flat_ip` | 0.06 s | 38.7 | **1.000** | 1.000 |
| `hnsw` | 9.12 s | 20.5 | 0.671 | 0.553 |
| `ivf_flat` | 0.14 s | 5.5 | 0.264 | 0.211 |

**At this catalogue size the approximate indexes are not worth their losses.**
HNSW is 1.9× faster and drops a third of the true neighbours; IVF is 7× faster
and drops three quarters. Exact search already answers in 39 microseconds, and
saving 18 of them is not worth silently returning the wrong items.

`flat_ip` is therefore the registered default, which is also what
[ADR-004](../adr/ADR-004-faiss-initial-index.md) assumed at this scale.

**The approximate numbers are floors, not verdicts.** Both ran at the untuned
defaults in `configs/models/faiss.yaml`. IVF's `nprobe=16` of `nlist=256` probes
6% of the space, and its 0.264 recall is largely that choice — a tuned `nprobe`
would trade its speed advantage back for accuracy. The conclusion supported here
is "exact is the right default at 69k vectors", not "HNSW and IVF are poor
indexes". They become worth tuning when the catalogue makes exact search too
slow, which it is nowhere near.

---

## Candidate aggregation — the largest gain in the phase

Every source was fitted once at its locked configuration, then each source and
each blend was scored through the same `run_experiment` driver, on validation.

| System | Strategy | NDCG@20 | Recall@20 | Coverage@20 |
|---|---|---:|---:|---:|
| **RRF, all four, uniform weights** | `reciprocal_rank_fusion` | **0.00914** | **0.02122** | 0.386 |
| RRF, weighted by individual strength | `reciprocal_rank_fusion` | 0.00899 | 0.02084 | 0.424 |
| Normalised score union | `normalized_score_union` | 0.00896 | 0.02040 | 0.422 |
| LightGCN alone | — | 0.00612 | 0.01580 | 0.434 |
| Weighted round robin, all four | `weighted_round_robin` | 0.00536 | 0.01322 | 0.447 |
| SASRec alone | — | 0.00433 | 0.01126 | 0.486 |
| Popularity + BPR hybrid | `reciprocal_rank_fusion` | 0.00354 | 0.00888 | 0.345 |
| BPR alone | — | 0.00339 | 0.00848 | 0.419 |
| Popularity alone | — | 0.00215 | 0.00558 | 0.00035 |

**Fusing all four retrievers beats the best single retriever by 1.49× on NDCG@20
and 1.34× on Recall@20.** This is a larger gain than anything the individual
models produced, and it comes from combining models that already exist.

As a harness check, popularity scored 0.0021494 here — matching the Phase 3
validation figure to seven decimal places, on an independent code path.

### Why fusion works so well here: the sources barely overlap

| Diagnostic | Value |
|---|---:|
| Mean sources proposing each retrieved item | **1.068** of 4 |
| Highest pairwise Jaccard (popularity ↔ SASRec) | 0.045 |
| Lowest pairwise Jaccard (LightGCN ↔ popularity) | 0.007 |
| Unique contribution — popularity | 0.898 |
| Unique contribution — BPR | 0.885 |
| Unique contribution — LightGCN | 0.873 |
| Unique contribution — SASRec | 0.857 |

At depth 300, **the four retrievers return almost entirely disjoint lists.** Every
pair agrees on under 5% of items, and each source contributes 86-90% of its
candidates uniquely. The union is nearly four times the coverage of any single
source, which is exactly the condition under which fusion pays.

This is the diagnostic earning its place: without it, "the blend improved" is a
number. With it, the reason is visible, and so is the prediction that fusion
would have added little had the sources agreed.

### Two results that contradict the obvious choice

**Weighting by individual strength made it slightly worse.** Uniform RRF scored
0.00914; weighting towards the stronger models scored 0.00899. The intuition
that a better retriever deserves a louder vote is wrong here, and the overlap
table says why — a weak source contributing 90% unique candidates is not adding
noise, it is adding reach, and down-weighting it discards coverage the strong
sources never had.

**Round robin is worse than its own best member.** At 0.00536 it falls *below*
LightGCN alone (0.00612), while every RRF variant beats it by ~1.7×. Round robin
interleaves by position and never compares across sources, so it cannot see that
two independent retrievers agreeing on an item is evidence. With sources this
disjoint, it mostly dilutes the strongest list with three weaker ones.

**The hybrid baseline the brief asked for adds almost nothing.** Popularity+BPR
fused scores 0.00354 against BPR's 0.00339 — a 4% gain, and still below every
single Phase 4 model. Two weak, similarly-biased retrievers do not make a strong
one; the gain arrives only when genuinely different model families are combined.

### Candidate recall — the ceiling Phase 6 inherits

Union of all four sources, at increasing depth:

| Pool depth | Candidate recall | Reachable-only |
|---:|---:|---:|
| 80 (20 per source) | 0.0338 | 0.0344 |
| 400 (100 per source) | 0.1032 | 0.1050 |
| 1,200 (300 per source) | 0.2042 | 0.2079 |

**Even with 1,200 candidates per user, four-fifths of targets are never
retrieved at all.** No ranker can recover them — this is the hard ceiling the
Phase 6 learning-to-rank stage inherits, and it caps the entire pipeline.

Recall roughly doubles for each tripling of depth with no sign of saturation, so
depth is still buying coverage at 1,200. Whether that is worth its latency is a
serving decision, not an offline one.

Of the 50,000 users, **880 have a target no source could return** — cold items,
absent from every model's fitting catalogue. The gap between `candidate_recall`
and `reachable_candidate_recall` is small precisely because that population is
small, but it is the only part of the ceiling that collaborative modelling
cannot move at any depth.

---

## Artifacts registered

| Artifact | Path | Notes |
|---|---|---|
| LightGCN model (3 seeds) | `artifacts/models/pixelrec50k/lightgcn/phase4-lightgcn-final{,-s43,-s44}` | 69,347 × 128 item embeddings, graph checksum attached |
| LightGCN metadata | `artifacts/metadata/lightgcn/phase4-lightgcn-final.json` | Test metrics prefixed `test_`, so a validation number cannot be mistaken for one |
| FAISS index | `artifacts/indexes/pixelrec50k/lightgcn/phase4-lightgcn-final` | `flat_ip`, 33.9 MB, exactness-verified before write |
| Phase 4 selection | `reports/metrics/phase_04/selected_configuration.json` | Locked before the test split was read |
| Generated config | `configs/models/phase4_selected.yaml` | Derived from the record; `--check` passes |
| Final test metrics | `reports/metrics/phase_04/final_test_metrics.json` | Per-seed and aggregate, with the Phase 3 baselines quoted for comparison |
| Aggregation results | `reports/metrics/phase_04/aggregation_comparison_selection.csv` | Nine systems: four solo, five blends |
| Retrieval diagnostics | `reports/metrics/phase_04/retrieval_diagnostics_selection.json` | Candidate recall at three depths, pairwise source overlap |
| Index benchmark | `reports/metrics/phase_04/index_benchmark_lightgcn.csv` | Three index types × three cut-offs, recall against exact |

### Verified after registration

- **Reload determinism** — the saved model reloaded twice produces identical
  top-20 lists for 2,000 users.
- **Index/model agreement** — the registered index reproduces the model's own
  ranking exactly over 200 query vectors, in both set and order.
- **Identity enforcement** — the index accepts its own model and raises
  `ArtifactValidationError` when handed a different `model_name`, which is the
  ADR-006 guarantee doing its job on real artifacts rather than a fixture.

Recorded in `reports/metrics/phase_04/reload_check_lightgcn.json`.

**SASRec is not registered.** Its selected configuration is locked and
reproducible, but the 45-epoch fit was not persisted — it ran through
`train.py --no-register` as a validation trial. Registering it costs another
90-minute fit and it is not on any serving path in this phase; the configuration
in `phase4_selected.yaml` is sufficient to reproduce it.

---

## Scope actually executed, against the brief

The Phase 4 brief specified a staged hyperparameter search across two rolling
folds and three seeds, for both LightGCN and SASRec. At the measured throughput
that is a multi-day job. What ran is below; what did not is stated as omitted,
not quietly dropped.

| Brief item | Executed | Notes |
|---|---|---|
| Phase 3 config drift corrections | ✅ Full | Five corrections, listed above |
| Rolling temporal validation | ⚠️ Built and verified, not used for selection | `fold_offset_2` reproduces the official validation split exactly; the folds were not spent on a second full search |
| Popularity + BPR hybrid | ✅ Full | Via `BlendedRetriever`, see aggregation below |
| LightGCN | ✅ Search over `num_layers`; single seed for selection, 3 for the final | `embedding_dim` held at the Phase 3 selected width |
| SASRec | ⚠️ Search over sequence length only | `num_blocks`, `num_heads`, `dropout`, `learning_rate` unsearched |
| Candidate aggregation | ✅ Full | Three strategies, five blend configurations |
| FAISS index | ✅ Full | Flat verified exact; HNSW and IVF benchmarked against it |
| Multi-seed final evaluation | ⚠️ LightGCN only, 3 seeds | SASRec was not refit on test; a 3-seed final would cost ~4.5 hours |
| Phase 4 report | ✅ This document | |

### Why rolling folds were built but not spent

The fold machinery is implemented, tested (28 tests), and verified against the
official split. It was not used to run a second complete search because each
fold costs a full re-fit of every configuration, roughly doubling a search that
was already the budget constraint.

That is a resource decision and it leaves the Phase 3 question — is the
validation/test reversal a property of the data or of the models? — **still
open**. The instrument to answer it now exists and is verified; the measurement
is Phase 5 work. Reporting it as "rolling validation complete" would claim an
answer that was never computed.

---

## Limitations

Stated as limits on what was measured, not as caveats that soften a result.

**LightGCN's depth was not bounded.** NDCG was still rising at `num_layers=3`.
Three layers is the best searched, not an optimum.

**LightGCN's width was not searched.** `embedding_dim` was fixed at 128 to
isolate the graph effect. Phase 3 separately recorded that BPR's NDCG was still
rising with width at d256, so the interaction between width and propagation
depth is entirely unmeasured.

**SASRec's search covered one axis of five.** Sequence length only. Any claim
about SASRec's ceiling on this data is unsupported by this phase.

**Single-seed selection.** LightGCN's layer comparison used one seed. The
multi-seed final runs put LightGCN's seed spread at 0.9% of its mean, and the
gaps between adjacent layer counts (0.00499 → 0.00584 → 0.00612, i.e. 5-17%) are
much larger than that — so the ablation ordering is very unlikely to be seed
noise. But that spread was measured at `L=3` on test, not per layer count on
validation, so it is supporting evidence rather than a direct demonstration.

**SASRec was never evaluated on test.** Its numbers throughout are validation
only. A three-seed final at 45 epochs costs roughly 4.5 hours, and with LightGCN
ahead on validation and the fused blend ahead of both, spending it was not
justified. So "SASRec beats the Phase 3 baselines" is a validation claim that
has not been confirmed on the held-out split — which, given Phase 3's reversal,
is exactly the kind of claim that has failed here before.

**No cold-start capability was added.** LightGCN and SASRec are both purely
collaborative. Items absent from fitting have no representation, are excluded
from `fit_item_catalogue`, and are absent from any index built over their
embeddings. The 880 cold test targets remain unreachable, exactly as in Phase 3.
Building a FAISS index does not change this, and the phase makes no claim that
it does.

**Coverage degrades as accuracy improves.** LightGCN's item coverage falls from
0.512 to 0.434 across the layer sweep, and SASRec's is lower still (~0.29). The
accuracy gains are partly concentration. Phase 6's reranker is where that
trade-off is supposed to be managed; nothing in Phase 4 manages it.

**The generated-config drift check is local, not enforced.** `reports/metrics/`
is gitignored, so the selection records that `generate_selected_config.py
--check` and `generate_phase4_config.py --check` read are not in the repository,
and CI has nothing to compare against. The Phase 3 script's docstring previously
claimed CI ran this; it does not, and the claim has been corrected. Committing
the two `selected_configuration.json` files would make the check enforceable —
they contain only hyperparameters and aggregate metrics, no per-user or per-item
rows — but changing the reports gitignore policy is left as a decision rather
than made silently here.

**FAISS runs single-threaded.** A consequence of the OpenMP coexistence decision
([ADR-009](../adr/ADR-009-faiss-torch-openmp-coexistence.md)). Not a constraint
at 69,347 vectors; it would become one at scale.

---

## What Phase 5 inherits

**A retriever that works on both splits, and a control that proves why.**
LightGCN at 3 layers beats popularity on test — the first model here to do so —
and the `num_layers=0` ablation establishes that the graph produced the gain,
not width, epochs, or an implementation difference.

**A fusion that beats every model in it, and the diagnostic explaining that.**
RRF over all four retrievers scores 1.49× the best single one, and the overlap
figures say why: the sources agree on almost nothing. Phase 5's two-tower
retriever should be evaluated as a *fusion member* as much as a solo model —
on this evidence, what a new retriever adds uniquely matters more than where it
ranks alone.

**A SASRec that is competitive on validation and untested on test.** The model
is implemented and causality-verified, and given a fair budget it overtakes both
Phase 3 baselines. It has never been evaluated on the held-out split.

**A verified index, and a verified fold builder.** Both are correctness-checked
against exact references — brute force for the index, the official Phase 2 split
for the folds. Neither has been used at scale yet.

**The open Phase 3 question, still open.** The validation/test reversal has an
instrument pointed at it now, and no measurement through it.

**A hard cold-start boundary.** Everything in Phases 3 and 4 is collaborative.
The 880 unreachable cold test targets are unreachable by construction, and no
amount of further collaborative modelling moves them. That is precisely what the
Phase 5 multimodal two-tower is for, and it is the first thing in this project
that could change the number.

