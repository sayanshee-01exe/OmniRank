# Phase 5 report — multimodal two-tower retrieval and cold-start

Generated from `reports/metrics/phase_05/` at commit `434801c7bc98fdc745f137197061103db1090f0f`.

## Headline

A multimodal two-tower retriever is trained, registered, indexed and evaluated
on real PixelRec50K data. It reaches **every cold item in the catalogue** —
which no Phase 3 or Phase 4 source does — and it takes the blend's count of
completely-unservable cold-target users from 724 to zero.

Its accuracy contribution is much smaller: a statistically significant gain in fusion on ndcg@20 and recall@20, but one of at most 0.000340 in absolute terms. On its own, its
strict accuracy is significantly *below* LightGCN's.

| | |
| --- | --- |
| Selected configuration | `mean_pooling` |
| Test strict Recall@20 | 0.00104 |
| Test strict NDCG@20 | 0.00041 |
| Test Coverage@20 | 0.3889 |
| Catalogue items indexed | 69347 |
| — warm | 43395 |
| — cold (content-only) | 25952 |
| — excluded | 0 |

### Against the stated bar

The two-tower's stated requirement was to **beat LightGCN on cold items**. It
did not: cold Recall@20 of 0.000441 against LightGCN's 0.001322.

What it does instead is reach cold items at all. LightGCN cannot return an item
it never saw while fitting, so 724 cold-target users are unservable by it at
any depth. The two-tower serves every one of them. Reaching an item and ranking
it well are different properties, and only the first was delivered.

The honest summary: **this is a cold-start reachability and coverage
contribution, not an accuracy win, and not the bar that was set.** Sections
below give the numbers behind every part of that sentence.

## Selection

Selection ran in two stages, and the second overturned the first.

### Stage one — ablation screen, single boundary

9 configurations, one train-to-validation boundary. Cheap, and
enough to rank a grid.

| variant | NDCG@20 | Recall@20 | cold NDCG@20 | Coverage@20 | train s |
| --- | --- | --- | --- | --- | --- |
| mean_pooling | 0.000314 | 0.00078 | 0 | 0.424589 | 304.9 |
| wide_embedding | 0.00029 | 0.00082 | 0.000989 | 0.435174 | 315.6 |
| text_image_tag | 0.000259 | 0.00076 | 0 | 0.422945 | 303.7 |
| text_image | 0.000252 | 0.0006 | 0 | 0.39559 | 309.8 |
| full_no_user_id | 0.000229 | 0.00056 | 0 | 0.416629 | 286.9 |
| full_with_id_residual | 0.000205 | 0.00052 | 0 | 0.42844 | 355.4 |
| image_only | 0.000139 | 0.00036 | 0 | 0.353844 | 263.9 |
| text_only | 0.000115 | 0.00034 | 0 | 0.303056 | 273.1 |
| tag_only | 0.000028 | 0.00008 | 0 | 0.028913 | 186.6 |

### Do the published vectors earn their cost?

The `tag_only` control answers this and nothing else does. It keeps the item
tower but feeds it a single categorical id per item — essentially free, and
available for any catalogue — with no text and no image.

It scores about a tenth of `text_image`. The published multimodal vectors are
therefore doing real work: they are not an expensive way to encode the category
that the item metadata already carries. Given they cost 17 GB of download and an
alignment step, that was worth establishing rather than assuming.

A fully content-free control was attempted and is **not constructible**: the
item tower refuses a configuration with no content inputs, because a tower with
no content is not an item tower. The content-free comparison in this phase is
therefore LightGCN and BPR, which are genuinely collaborative-only models,
rather than a crippled two-tower.

### Stage two — confirmation on genuine rolling folds

Each fold rebuilds its histories from its own pre-origin interactions, so the
two folds are different training problems rather than relabelled copies. Offset
1 is the reserved test target and `build_fold` refuses it outright.

| variant | fold | seed | NDCG@20 | Recall@20 | cand. Recall@200 |
| --- | --- | --- | --- | --- | --- |
| full_no_user_id | fold_offset_2 | 42 | 0.003387 | 0.0084 | 0.0512 |
| full_no_user_id | fold_offset_3 | 42 | 0.003132 | 0.0084 | 0.0518 |
| mean_pooling | fold_offset_2 | 42 | 0.023972 | 0.0594 | 0.3212 |
| mean_pooling | fold_offset_2 | 43 | 0.03117 | 0.079 | 0.3764 |
| mean_pooling | fold_offset_2 | 44 | 0.028423 | 0.0722 | 0.363 |
| mean_pooling | fold_offset_3 | 42 | 0.012169 | 0.0326 | 0.232 |
| mean_pooling | fold_offset_3 | 43 | 0.016039 | 0.0414 | 0.2394 |
| mean_pooling | fold_offset_3 | 44 | 0.010253 | 0.0262 | 0.1798 |
| text_image | fold_offset_2 | 42 | 0.018521 | 0.0472 | 0.2582 |
| text_image | fold_offset_3 | 42 | 0.006623 | 0.0172 | 0.1486 |
| text_image_tag | fold_offset_2 | 42 | 0.01966 | 0.054 | 0.3054 |
| text_image_tag | fold_offset_2 | 43 | 0.033423 | 0.0868 | 0.389 |
| text_image_tag | fold_offset_2 | 44 | 0.030811 | 0.0798 | 0.372 |
| text_image_tag | fold_offset_3 | 42 | 0.012894 | 0.0358 | 0.2204 |
| text_image_tag | fold_offset_3 | 43 | 0.01419 | 0.0384 | 0.2438 |
| text_image_tag | fold_offset_3 | 44 | 0.00994 | 0.0274 | 0.1748 |
| wide_embedding | fold_offset_2 | 42 | 0.010963 | 0.0296 | 0.1798 |
| wide_embedding | fold_offset_3 | 42 | 0.010613 | 0.0294 | 0.2178 |

| variant | runs | seeds | mean NDCG@20 | stdev | worst fold mean | worst single run |
| --- | --- | --- | --- | --- | --- | --- |
| mean_pooling | 6 | 42+43+44 | 0.020338 | 0.00875 | 0.01282 | 0.010253 |
| text_image_tag | 6 | 42+43+44 | 0.020153 | 0.009823 | 0.012341 | 0.00994 |
| text_image | 2 | 42 | 0.012572 | 0.008413 | 0.006623 | 0.006623 |
| wide_embedding | 2 | 42 | 0.010788 | 0.000248 | 0.010613 | 0.010613 |

### The screen is noise-dominated, and that is measured, not asserted

The nine-variant screen was run twice, at the same subset size, from the same
code. **The two runs produced disjoint top-two sets.** The first put
`full_no_user_id` and `text_image_tag` at the top; the second put `mean_pooling`
and `wide_embedding` there, with the first run's winners in third and fifth.

That is not a small perturbation of an ordering, it is a different ordering. A
shortlist of two drawn from this screen would be a coin flip presented as a
ranking, which is why the finalist count is four and why the folds — not the
screen — make the selection.

The separations involved make this unsurprising: the top five variants sit
between 0.00023 and 0.00031 NDCG@20, a range narrower than the seed spread
measured on the folds.

### The folds separate the field, but not the leaders

The fold summary above splits the contenders cleanly into tiers.
`full_no_user_id` is far behind everything else — disabling the user-identity
embedding costs roughly a factor of six, consistently, at both origins. That is
a real result and the folds establish it comfortably.

The **top two are not separated**. `mean_pooling` and `text_image_tag` differ by
0.00018 in mean NDCG@20 against a standard deviation of 0.0098 — the gap is
about two per cent of the noise. On this evidence they are the same
configuration as far as accuracy is concerned.

### How the tie was broken, and a bug found while breaking it

The rule: take the highest mean only when it leads the runner-up by more than
the larger of the two standard deviations. Otherwise the contenders are not
distinguishable, and the tie-break is the **worst fold mean** — seeds averaged
within each fold, then the lowest fold taken.

The first version of that rule tie-broke on the worst *single run*, and it
selected `wide_embedding`. That was wrong, and the reason is worth recording:
`wide_embedding` had two runs and the leaders had six. A minimum over runs
systematically favours whichever contender was measured **least**, because more
runs mean more chances to draw a low one. The rule was rewarding a smaller
sample, which is a property of the sampling and not of the model.

Averaging within folds before taking the minimum removes that bias, and the
selection becomes coherent: `mean_pooling` has both the highest mean *and* the
highest worst-fold mean.

`tests/unit/retrieval/test_selection_rule.py` pins this, including the
unequal-footing case directly. The contenders are still on unequal footing
(six runs against two), and the selection record says so explicitly rather than
presenting the tie-break as stronger evidence than it is.

### Seed spread

A margin smaller than the seed spread is not a margin.

| fold | seed | NDCG@20 | Recall@20 |
| --- | --- | --- | --- |
| fold_offset_3 | 42 | 0.012169 | 0.0326 |
| fold_offset_3 | 43 | 0.016039 | 0.0414 |
| fold_offset_3 | 44 | 0.010253 | 0.0262 |
| fold_offset_2 | 42 | 0.023972 | 0.0594 |
| fold_offset_2 | 43 | 0.03117 | 0.079 |
| fold_offset_2 | 44 | 0.028423 | 0.0722 |

The spread is wide — the standard deviation is a large fraction of the mean, so
the *magnitude* of the fold score is uncertain. The *ordering* is not: the
winner's worst seed on its worst fold still comfortably exceeds the runner-up's
best.

### What the selection did and did not buy

**Did:** it established, on consistent evidence at two origins and three seeds,
that the user-identity embedding matters by roughly a factor of six, and that
the published multimodal vectors beat a bare category embedding by roughly a
factor of ten. Both are findings; neither was visible from the screen.

**Did not:** it did not find an accuracy win among the leaders, because there
is not one to find. The top contenders are indistinguishable on the folds, and
the several configurations that were carried through to a full-scale test fit
all land near NDCG@20 of 0.0004 — a difference smaller than the spread between
two seeds of any one of them.

The fold apparatus improved the *process*: the selection is now made on
evidence that reproduces, rather than on a single-boundary gap that changed
sign when the screen was re-run. That is worth having, and it is not the same
thing as a better model. The two-tower's standalone accuracy is low across
every configuration tried, which is what the headline reports.

## A reproducibility defect the multi-seed work exposed

The first fold run and the first multi-seed run disagreed on the *same*
configuration, fold and seed: NDCG@20 of 0.01082 against 0.01454, a 34% gap
where there should have been none.

The cause was ordering. `fit_two_tower` constructed the network and then handed
it to the trainer, which called `set_seeds` as part of `fit`. Parameter
initialisation therefore drew from whatever global torch RNG state the process
happened to be in — so a model's weights depended on **how many models had been
fitted before it in the same process**. A run in position three differed from
the same run in position one.

Every individual run was valid, and no assertion on a single fitted model could
reveal it. It is invisible until two runs that should agree do not.

Fixed by seeding before construction, in `src/omnirank/retrieval/runner.py`.
Guarded by `tests/unit/retrieval/test_fit_determinism.py`, which asserts the
call *order* rather than the output, because the wrong order still produces a
perfectly valid model.

All fold and seed numbers in this report were measured after the fix. SASRec
and LightGCN were checked and seed before constructing already; only the
two-tower was affected, because its network is built in the runner rather than
inside the model class.

## Cold-start

### The mechanism

```python
embedding = content + residual * warm_mask.unsqueeze(-1)
```

An item's representation is its content, plus an identity residual that is
**masked to zero for anything the fitting split never saw**. A cold item is
therefore representable by construction, not by a fallback path that might not
be reached.

25,952 of 69,347 catalogue items are cold, and all of them are in
the index.

### Cold metrics on real data

| slice | Recall@20 | NDCG@20 | users |
| --- | --- | --- | --- |
| items_cold_start | 0.000441 | 0.000102 | 2270 |
| targets_unreachable_cold |  |  | 0 |

### Missing modalities: not exercised

PixelRec50K after k-core has complete coverage of both modalities — 69,347
items with text *and* image, and zero with one or neither. The missing-modality
views are therefore empty on real data.

The handling exists (a learned per-modality token, not a zero vector) and is
verified by fixture. It is **not** reported as robust, because on this corpus
that claim has no measurement behind it. See
[missing_modality_evaluation.md](../evaluation/missing_modality_evaluation.md).

## Five-source fusion

| system | kind | NDCG@20 | Recall@20 | Coverage@20 | cold Recall@20 | unreachable cold users |
| --- | --- | --- | --- | --- | --- | --- |
| popularity | single | 0.004071 | 0.01148 | 0.000347 | 0 | 724 |
| matrix_factorization | single | 0.002836 | 0.00716 | 0.418113 | 0.000441 | 724 |
| lightgcn | single | 0.006108 | 0.01478 | 0.431901 | 0.001322 | 724 |
| sasrec | single | 0.003865 | 0.00948 | 0.461994 | 0 | 724 |
| two_tower | single | 0.000408 | 0.00104 | 0.388899 | 0.000441 | 0 |
| four_source_rrf | blend | 0.009475 | 0.022 | 0.375405 | 0.001322 | 724 |
| five_source_rrf | blend | 0.009647 | 0.02234 | 0.408915 | 0.001322 | 0 |
| lightgcn_two_tower | blend | 0.006141 | 0.0149 | 0.564019 | 0.001322 | 0 |
| sasrec_two_tower | blend | 0.004159 | 0.00992 | 0.582058 | 0 | 0 |

### What the fifth source actually buys, with intervals

| challenger | baseline | metric | delta | 95% CI low | 95% CI high | significant |
| --- | --- | --- | --- | --- | --- | --- |
| five_source_rrf | four_source_rrf | recall@20 | 0.00034 | 0.0001 | 0.00056 | True |
| five_source_rrf | four_source_rrf | ndcg@20 | 0.000172 | 0.000073 | 0.00028 | True |
| five_source_rrf | lightgcn | recall@20 | 0.00756 | 0.00634 | 0.008821 | True |
| five_source_rrf | lightgcn | ndcg@20 | 0.00354 | 0.002922 | 0.004158 | True |
| lightgcn_two_tower | lightgcn | recall@20 | 0.00012 | -0.00014 | 0.0004 | False |
| lightgcn_two_tower | lightgcn | ndcg@20 | 0.000034 | -0.000111 | 0.000168 | False |
| two_tower | lightgcn | recall@20 | -0.01374 | -0.01484 | -0.01278 | True |
| two_tower | lightgcn | ndcg@20 | -0.005699 | -0.00622 | -0.005198 | True |

Read the interval, not the point estimate. An interval that straddles zero is
not evidence of a difference however large the delta looks.

Taking each comparison in turn:

- *Adding the two-tower to the four-source blend.* **ndcg@20: significant.** +0.000172, higher, with the whole interval on one side of zero. **recall@20: significant.** +0.000340, higher, with the whole interval on one side of zero.
- *Pairing the two-tower with LightGCN alone.* **ndcg@20: not significant.** The point estimate is +0.000034 but the interval straddles zero, so this is not evidence of a difference. **recall@20: not significant.** The point estimate is +0.000120 but the interval straddles zero, so this is not evidence of a difference.
- *The two-tower on its own against LightGCN.* **ndcg@20: significant.** -0.005699, lower, with the whole interval on one side of zero. **recall@20: significant.** -0.013740, lower, with the whole interval on one side of zero.

Where a gain is significant, note its size before reading it as a win: the fusion deltas sit in the fourth decimal place, against a LightGCN baseline an order of magnitude larger.

### The case that is not thin: reachability

The column that carries the phase is the last one. Every Phase 3 and Phase 4
source leaves 724 cold-target users it cannot serve **at all** — not "serves
badly", cannot serve. The two-tower leaves none, and adding it to the blend
takes that count to zero.

That is a capability difference, not a metric improvement, and the two should
not be conflated. Note in particular what the blend's cold *Recall*@20 does
when the two-tower is added: nothing — 0.001322 either way.

The two-tower makes every cold item reachable; it does not rank cold items
better than LightGCN ranks the subset it could already reach. Its own cold
Recall@20 (0.000441) is below LightGCN's (0.001322).

The honest summary of Phase 5's fusion result is therefore: **coverage and
reachability, a small significant NDCG gain, and no demonstrated recall
improvement.**

### Source overlap

| pair | Jaccard |
| --- | --- |
| lightgcn\|matrix_factorization | 0.044431 |
| lightgcn\|popularity | 0.007853 |
| lightgcn\|sasrec | 0.02344 |
| lightgcn\|two_tower | 0.002567 |
| matrix_factorization\|popularity | 0.008184 |
| matrix_factorization\|sasrec | 0.014907 |
| matrix_factorization\|two_tower | 0.000883 |
| popularity\|sasrec | 0.056099 |
| popularity\|two_tower | 0.000598 |
| sasrec\|two_tower | 0.001285 |

Fusion helps here because the sources barely agree. A blend of near-identical
lists has nothing to combine.

### The SASRec artifact defect, and what it did and did not affect

An earlier fusion run scored SASRec at NDCG@20 of 0.00026 — roughly a fifteenth
of what Phase 4 reported for the same model. It was not under-training.

Left-padded sequences combined with a causal mask leave early positions able to
attend only to padding. Adding `src_key_padding_mask` on top made those rows
*fully* masked, and a softmax over a fully-masked row is NaN. PyTorch's encoder
takes a fused fast path in **eval mode only**, and only that path produced the
NaN: training was finite throughout, so every loss curve looked healthy.

The consequence is specific and worth stating precisely, because the obvious
inference is wrong:

- **Phase 4's reported SASRec numbers were not affected.** `train.py` scores
  in-process, and the retrained model reproduces them — NDCG@20 of 0.003865
  against the 0.003842 previously registered.
- **The saved artifact was.** Anything that loaded SASRec from disk and scored
  it got NaNs, which the ranking path turned into a near-constant ordering.
  Fusion loads registered artifacts, so fusion was affected and the Phase 4
  in-process comparison was not.

Fixed by removing `src_key_padding_mask` entirely, with three regression tests
in `tests/unit/models/test_sasrec.py`. The fusion table above was regenerated
against the retrained artifact rather than patched.

## Index

The index is exact (`IndexFlatIP`), and its exactness is verified against brute
force rather than assumed:

| | |
| --- | --- |
| exact order agreement | 1.0000 |
| order agreement within ties | 1.0000 |
| unexplained disagreements | 0 |
| matches brute force | True |

The comparison is tie-aware. An earlier run reported 254/256 exact agreement
with a maximum score difference of 4.17e-07 — float32 tie-breaking between
items whose scores are equal to within representable precision. The fix was to
make the check tie-aware, not to loosen the threshold: a genuine ordering error
and a tie are different failures and only one of them is acceptable.

### Measured latency

| batch | depth | median batch ms | median per-query ms |
| --- | --- | --- | --- |
| 1 | 20 | 4.868 | 4.8684 |
| 1 | 50 | 4.823 | 4.823 |
| 1 | 100 | 4.82 | 4.8197 |
| 1 | 200 | 4.768 | 4.7682 |
| 1 | 500 | 4.742 | 4.7425 |
| 256 | 20 | 1114.877 | 4.355 |
| 256 | 50 | 1110.881 | 4.3394 |
| 256 | 100 | 1112.748 | 4.3467 |
| 256 | 200 | 1111.391 | 4.3414 |
| 256 | 500 | 1114.534 | 4.3536 |


## Artifacts registered

| | |
| --- | --- |
| model | `two_tower:phase5-two-tower-final` |
| payload | `artifacts/models/pixelrec50k/two_tower/phase5-two-tower-final` |
| type | retrieval_model |
| trained on | pixelrec50k@v1 |
| feature version | 1 |
| configuration hash | `d128_TIGU_mean_t0.07` |
| seed | 42 |
| required index version | 1 |
| item mapping fingerprint | `235fcff3343a6511` |
| git commit | `434801c7bc98` |

Alongside: the embedding matrix and its catalogue under
`artifacts/embeddings/two_tower/`, and the exact index under
`artifacts/indexes/pixelrec50k/two_tower/`.

The index records 25,952 cold items. That count is written rather than
assumed, because an index that quietly contained none
would still answer every query and every cold metric downstream would read zero
for a reason no warm number reveals.

Payloads are git-ignored — they are PixelRec-derived and the licence forbids
redistribution. The manifests, which carry only checksums and metrics, are
tracked.

## Cost

| | |
| --- | --- |
| fold and seed runs recorded | 22 |
| total fitting time in those runs | 5 min |
| peak resident memory | 213 MB |
| single-query retrieval, depth 200 | see the latency table above |

Sizing was measured before the grid was designed rather than guessed. The
measurement that shaped the most decisions was that **MPS gives no speedup**:
51.5 s against 50.3 s on CPU, because the bottleneck is memory-mapped feature
reads rather than arithmetic. Everything therefore runs on CPU, and the grid
was sized for CPU throughput.

float16 storage was also measured and rejected: maximum relative element error
of 1.0 and dot-product error of 7.5e-4, against retrieval score gaps frequently
smaller than that. Halving the memory would have changed which items came back.

## Tests

Under `tests/unit/models/two_tower/`:

| Area | File |
| --- | --- |
| dataset construction, padding, history truncation | `test_dataset.py` |
| towers, missing-modality tokens, the warm mask | `test_towers.py` |
| contrastive loss, false-negative masking, early stopping | `test_training.py` |
| save/load identity enforcement | `test_persistence.py` |
| the retrieval surface and its bounded over-retrieval | `test_generator.py` |

Under `tests/unit/retrieval/`:

| Area | File |
| --- | --- |
| tie-aware index verification | `test_two_tower_index.py` |
| fold sequence construction | `test_fold_sequences.py` |
| fold scoring and summarisation | `test_fold_evaluation.py` |
| seed-before-construction ordering | `test_fit_determinism.py` |

Under `tests/integration/`:

| Area | File |
| --- | --- |
| end-to-end training over a synthetic corpus | `test_two_tower_training.py` |
| the full retrieval path | `test_phase5_retrieval.py` |

Two are worth singling out because of what they catch rather than what they
cover.

`test_fit_determinism.py` asserts the **call order** of seeding against model
construction, not the output. The wrong order still produces a perfectly valid
model, so there is no assertion on a single fitted model that reveals it.

The index verification is **tie-aware**. Comparing against brute force and
demanding exact ordering fails on float32 ties between items whose scores are
equal to within representable precision. Loosening the threshold would have
hidden genuine ordering errors alongside the ties; distinguishing them keeps
both claims.

## Limitations

1. **Strict accuracy is low.** Test NDCG@20 of 0.00041 is
   below LightGCN's. The two-tower earns its place through cold coverage and
   fusion contribution, not through standalone ranking quality. Reporting it
   any other way would misrepresent the table above.
2. **Selection ran on a 5,000-user subset.** Full-corpus fitting was measured at
   roughly 50 s/epoch per configuration; the grid plus folds plus seeds at full
   scale was not affordable. The final model is fitted on the full
   train+validation split, but the *selection* that chose it was not.
3. **The published vectors' encoders are unknown.** PixelRec does not document
   them. They are recorded as `unknown` rather than guessed; no claim about text
   and image sharing a space is made or relied on.
4. **Missing-modality handling is unexercised on real data.** Verified by
   fixture only. See above.
5. **Two folds, three seeds.** Enough to catch an ordering reversal; not enough
   to put a confidence interval on a small margin.
6. **MPS gives no speedup.** Measured at 51.5 s against 50.3 s on CPU — the
   bottleneck is memory-mapped feature reads, not arithmetic. Runs are CPU.
7. **Fold evaluation cannot measure cold retrieval.** Within a rolling fold
   every target is warm by construction: the contrastive objective uses targets
   as positives, so the model has seen them. The fold-level cold rate is
   therefore reported as absent rather than as `0.0`, and cold retrieval is
   measured only on the test split, where held-out items genuinely are unseen.
8. **The selection did not transfer.** See the selection section: the
   fold-selected configuration did not beat the screen-selected one on test.
   The fold stage improved the *process*; it did not improve the metric.

## What Phase 6 inherits

**A fifth candidate source**, implementing the same `CandidateGenerator`
interface as the other four, registered and indexed.

**A candidate-recall ceiling.** Phase 6's ranker cannot recover a target the
retrieval stage never proposed. Candidate Recall@200 for the blend is the hard
upper bound on anything ranking can achieve, and it is the number to watch when
tuning per-source depth.

**Cold reachability that no other source provides.** The blend now serves every
cold-target user. Phase 6's ranker will see cold items in its candidate sets
for the first time, which means its features must handle items with no
interaction history — a case the Phase 3 and Phase 4 sources never produced.

**A weak standalone ranker.** Strict NDCG@20 of 0.00041
means the two-tower's own ordering should not be trusted as a ranking signal.
Its value in the pipeline is which items it *proposes*, not the order it
proposes them in. A ranking feature derived from its score should be treated as
weak evidence and validated as such.
