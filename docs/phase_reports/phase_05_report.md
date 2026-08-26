# Phase 5 report — multimodal two-tower retrieval and cold-start

Generated from `reports/metrics/phase_05/` at commit
`6b92287ce143e8c8602ed6bbb450b31ddff9f6fd`. Every number below is read from a metric
file at generation time; none is transcribed.

## Headline

A multimodal two-tower retriever is trained, registered, indexed and evaluated
on real PixelRec50K data. It reaches **every cold item in the catalogue** —
which no Phase 3 or Phase 4 source does — taking the blend's count of
completely-unservable cold-target users from 724 to 0.

In fusion, adding it gives a statistically significant gain on ndcg@20 and recall@20 of at most 0.007260, and **no** significant gain on cold_recall@20.

Standalone, its NDCG@20 of 0.00887 is above the best collaborative source (`lightgcn`, 0.00611), and the paired interval excludes zero.

| | |
| --- | --- |
| Selected configuration | `mean_pooling` |
| Test strict Recall@20 | 0.02186 |
| Test strict NDCG@20 | 0.00887 |
| Test Coverage@20 | 0.7943 |
| Catalogue items indexed | 69347 |
| — warm | 69120 |
| — cold (content-only) | 227 |
| — excluded | 0 |

### Against the stated bar

The two-tower's stated requirement was to **beat LightGCN on cold items**.
**It did**: cold Recall@20 of 0.018062 against LightGCN's 0.001322 — a factor of 13.7. The paired interval over the 2270 cold-target users is [+0.011454, +0.022467], which excludes zero: a real advantage.

It also reaches cold items that LightGCN cannot reach at all: a
collaborative source can never return an item it never saw while fitting, so
those users are unservable by it at any depth. The two-tower serves every one
of them. Reaching an item and ranking it well are different properties, and
here both were delivered.

### How to read the result types in this report

| Type | Meaning |
| --- | --- |
| **synthetic** | fixture corpora; proves code paths work, never a Phase 5 result |
| **rolling validation** | pre-test folds at offsets 3 and 2; used for selection only |
| **official final test** | fitted on train+validation, scored once, after the lock |
| **strict** | cold targets counted as misses — the production denominator |
| **warm** | restricted to reachable targets — "how well does it rank what it sees" |
| **cold** | users whose target is a cold item — what this phase exists to move |
| **standalone** | one source alone |
| **fusion** | rank-based blend of several sources |

Sections 24-34 are all **official final test** unless stated otherwise.
Sections 14-23 are **rolling validation** and never touched the test split.

## 1. Repository state before final closure

Phases 1-4 were complete: the data pipeline, the offline evaluation
framework, four candidate generators (popularity, BPR, LightGCN, SASRec),
reciprocal-rank fusion and an exact FAISS index.

Every one of those four is **collaborative**. None can return an item it never
saw during fitting, which meant a block of cold-target users was unservable at
any retrieval depth and no amount of ranking work downstream could recover
them. That gap is what Phase 5 exists to close.

## 2. Completed Phase 5 implementation

| Component | Module |
| --- | --- |
| configuration | `src/omnirank/models/two_tower/config.py` |
| training dataset | `src/omnirank/models/two_tower/dataset.py` |
| towers and network | `src/omnirank/models/two_tower/model.py` |
| contrastive objective | `src/omnirank/models/two_tower/losses.py` |
| trainer | `src/omnirank/models/two_tower/training.py` |
| persistence | `src/omnirank/models/two_tower/persistence.py` |
| cold-inclusive catalogue | `src/omnirank/models/two_tower/catalogue.py` |
| retrieval surface | `src/omnirank/models/two_tower/generator.py` |
| feature store | `src/omnirank/features/multimodal_store.py` |
| embeddings and exact index | `src/omnirank/retrieval/two_tower_index.py` |
| fold evaluation | `src/omnirank/retrieval/fold_evaluation.py` |

The retrieval surface implements the same `CandidateGenerator` interface as the
other four sources, so fusion treats it identically. A source with a bespoke
integration path would be one whose measured contribution is partly an artefact
of its plumbing.

## 3. PixelRec feature source

PixelRec publishes two per-item matrices alongside the interaction log. They
are used as published; nothing is re-encoded.

**The encoders are undocumented.** The release does not say which model produced
these vectors, at which checkpoint, or with what preprocessing. The tracked
config therefore records `encoder_identity: unknown` for both modalities.

Calling them CLIP or BERT embeddings would be a provenance claim the source does
not support, and every downstream comparison would silently inherit it. Two
design consequences follow from not knowing: no text/image alignment is assumed
(each modality is projected separately before fusion), and no input
normalisation is claimed (`input_vectors_normalized: False`).

See [pixelrec_published_vectors.md](../features/pixelrec_published_vectors.md).

## 4. Feature dimensions

| Modality | Dimension | dtype | Storage |
| --- | ---: | --- | --- |
| text | 1024 | float32 | memory_map |
| image | 1024 | float32 | memory_map |

float32, not float16. This was measured rather than assumed: over the real
matrices float16 gave a maximum relative element error of 1.0 and a dot-product
error of 7.5e-4, against retrieval score gaps frequently smaller than that.
Halving the memory would have changed which items came back.

## 5. Feature coverage

| Group | Items |
| --- | ---: |
| both modalities | 69347 |
| text only | 0 |
| image only | 0 |
| neither | 0 |

Coverage is complete: 1.0000 for text and
1.0000 for image. Every catalogue item is
content-representable, which is what makes the cold guarantee reachable at all
— and which also means the missing-modality path is **unexercised** on this
corpus (section 27).

## 6. Feature and mapping checksums

| Identity | Value |
| --- | --- |
| feature manifest | `dc80bc8c54d07ebe9867fa7d2fc442e2` |
| item mapping | `235fcff3343a651150e8f220bfebc9bd` |
| dataset manifest | `586fd66a82e7c56708e0b3547f37a5f1` |
| feature version | 1 |

A model built against a different item mapping resolves every row to the wrong
item **and still returns plausible recommendations**. Nothing about the output
reveals it, so the identity is enforced at load time rather than trusted.

## 7. User-tower architecture

The user tower pools the item embeddings of a user's interaction history,
optionally adding a learned user-identity embedding.

Pooling is recency-weighted or mean; the selected configuration uses mean
(section 15). Padding is excluded from the divisor — including it would shrink
every short history's representation toward zero in proportion to how short it
was, which is a bug that looks like a modelling choice.

Histories are right-aligned and truncated oldest-first at the configured
maximum length.

## 8. Item-tower architecture

The item tower is what distinguishes this from every earlier retriever: it
represents an item by what it *is* rather than by who interacted with it.

```text
content   = fuse(text_projection, image_projection, tag_embedding)
embedding = content + id_residual * warm_mask
```

Each modality is projected separately before fusion, because the two published
matrices are not known to share a space (section 3). Outputs are L2-normalised,
which makes the index's inner product a cosine similarity.

The tower refuses a configuration with no content inputs at all: a tower with no
content is not an item tower, and the refusal is why the content-free ablation
in section 22 is not constructible.

## 9. Missing-modality handling

A missing modality is represented by a **learned per-modality token**, not by
a zero vector:

```python
return torch.where(present.unsqueeze(-1), encoded, self.missing)
```

A zero vector is a specific point in the projected space — one the model did not
choose and cannot move — and it collides with whatever legitimately projects
near the origin. The learned token is trained like any other parameter, so "I
have no image" becomes a representation the model picked.

Verified by fixture (`tests/unit/models/two_tower/test_towers.py`). Its
real-data status is section 27, and it is not what a reader might assume.

## 10. Warm item-ID residual policy

An item the fitting split observed receives a learned identity residual on top
of its content embedding. This is what lets a warm item's representation encode
collaborative signal the content cannot express.

The residual is gated by `warm_mask`, computed from the *fitting* split alone.
An item is warm because training saw it — not because the evaluation split did,
which would be leakage wearing the mask of a feature.

## 11. Cold-item content-only policy

For a cold item the mask is zero, so the residual term vanishes and the
embedding is content only:

```text
embedding_cold = content + id_residual * 0 = content
```

This is the phase's central guarantee, and it holds **by construction** rather
than through a fallback path that might not be reached. An untrained residual
added to a cold item would be an arbitrary vector from the initialiser, moving
that item to a position nothing chose.

227 of 69347 catalogue items
are cold, and all of them are in the index. That count is written to the index
manifest rather than assumed: an index that quietly contained no cold items
would still answer every query, and every cold metric downstream would read zero
for a reason no warm number reveals.

## 12. Contrastive objective

In-batch softmax (InfoNCE). Each row's target is its positive; the other rows'
targets in the same batch are its negatives.

```text
loss = -log( exp(s_ii / T) / sum_j exp(s_ij / T) )
```

Temperature is a tracked hyperparameter (section 15). In-batch negatives are
what make this affordable: sampling explicit negatives over a 69,347-item
catalogue for every example would dominate the step cost.

## 13. False-negative handling

In-batch negatives are drawn from other rows' targets, and some of those are
items the user in question actually liked. Training against them teaches the
model that a correct answer is wrong.

Masked before the softmax:

```python
logits = logits.masked_fill(false_negative_mask, MASKED_LOGIT)  # -1e4
```

`-1e4` rather than `-inf`: a row that ended up fully masked would produce NaN
under `-inf`, and a NaN loss propagates silently into every parameter. The
masked fraction is logged each epoch, so a mask that suddenly covers most of a
batch is visible rather than inferred.

## 14. Rolling-fold selection process

Selection ran in two stages and the folds, not the screen, made the choice.

**Stage one — ablation screen.** Nine variants on the single train-to-validation
boundary. Cheap enough to rank a grid.

**Stage two — rolling-fold confirmation.** The screen's finalists are re-fitted
on each pre-test rolling fold. A fold rebuilds each user's history from that
user's own pre-origin interactions, so the two folds are genuinely different
training problems rather than relabelled copies. Offsets 3 and 2 are used;
**offset 1 is the reserved test target** and `build_fold` refuses it outright,
so a selection run cannot reach it even by mistake.

### The screen is noise-dominated, and that is measured

The nine-variant screen was run twice, at the same subset size, from the same
code. **The two runs produced disjoint top-two sets.** That is not a small
perturbation of an ordering, it is a different ordering — which is why the
finalist count is four rather than two, and why the screen does not select.

| variant | NDCG@20 | Recall@20 | Coverage@20 | cold NDCG@20 | train s |
| --- | --- | --- | --- | --- | --- |
| mean_pooling | 0.000314 | 0.00078 | 0.424589 | 0 | 304.9 |
| wide_embedding | 0.00029 | 0.00082 | 0.435174 | 0.000989 | 315.6 |
| text_image_tag | 0.000259 | 0.00076 | 0.422945 | 0 | 303.7 |
| text_image | 0.000252 | 0.0006 | 0.39559 | 0 | 309.8 |
| full_no_user_id | 0.000229 | 0.00056 | 0.416629 | 0 | 286.9 |
| full_with_id_residual | 0.000205 | 0.00052 | 0.42844 | 0 | 355.4 |
| image_only | 0.000139 | 0.00036 | 0.353844 | 0 | 263.9 |
| text_only | 0.000115 | 0.00034 | 0.303056 | 0 | 273.1 |
| tag_only | 0.000028 | 0.00008 | 0.028913 | 0 | 186.6 |

### Fold results

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

| variant | runs | seeds | mean NDCG@20 | stdev | worst fold mean |
| --- | --- | --- | --- | --- | --- |
| mean_pooling | 6 | 42+43+44 | 0.020338 | 0.00875 | 0.01282 |
| text_image_tag | 6 | 42+43+44 | 0.020153 | 0.009823 | 0.012341 |
| text_image | 2 | 42 | 0.012572 | 0.008413 | 0.006623 |
| wide_embedding | 2 | 42 | 0.010788 | 0.000248 | 0.010613 |

### How the tie was broken, and a bug found while breaking it

The rule: take the highest mean **only** when it leads the runner-up by more
than the larger of the two standard deviations. Otherwise the contenders are not
distinguishable, and the tie-break is the **worst fold mean** — seeds averaged
within each fold, then the lowest fold taken.

The first version of that rule tie-broke on the worst *single run*, and it
selected the contender with the fewest runs. A minimum over runs systematically
favours whichever configuration was measured **least**, because more runs mean
more chances to draw a low one. That is a property of the sampling, not of the
model. Averaging within folds before taking the minimum removes the bias.

`tests/unit/retrieval/test_selection_rule.py` pins this, including the
unequal-footing case. Contenders remain on unequal footing (six runs against
two), and the selection record logs `equal_footing` explicitly rather than
presenting the tie-break as stronger evidence than it is.

## 15. Selected configuration

| Field | Value |
| --- | --- |
| ablation label | `mean_pooling` |
| embedding dimension | 128 |
| text projection | 128 |
| image projection | 128 |
| modality fusion | gated |
| history pooling | mean |
| temperature | 0.07 |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| batch size | 256 |
| max epochs | 10 |
| early-stopping patience | 3 |
| seed | 42 |
| user-ID embedding | True |
| warm item-ID residual | False |
| L2 normalisation | True |

Tracked as `configs/models/phase5_selected.yaml`, generated from
`reports/metrics/phase_05/selected_configuration.json` and never hand-edited.
`python scripts/generate_phase5_configs.py --check` fails on drift.

Selected by: mean_pooling — two stage: ablation screen on the train->validation boundary, then confirmation across rolling folds; chosen on mean fold strict ndcg@20 with the worst fold as tie-breaker

## 16. Multi-seed verification

The selected configuration re-run at seeds 42, 43 and 44 on both folds. A
margin smaller than the seed spread is not a margin.

| variant | fold | seed | NDCG@20 | Recall@20 |
| --- | --- | --- | --- | --- |
| mean_pooling | fold_offset_3 | 42 | 0.012169 | 0.0326 |
| mean_pooling | fold_offset_3 | 43 | 0.016039 | 0.0414 |
| mean_pooling | fold_offset_3 | 44 | 0.010253 | 0.0262 |
| mean_pooling | fold_offset_2 | 42 | 0.023972 | 0.0594 |
| mean_pooling | fold_offset_2 | 43 | 0.03117 | 0.079 |
| mean_pooling | fold_offset_2 | 44 | 0.028423 | 0.0722 |

The spread is wide: the standard deviation is a large fraction of the mean, so
the *magnitude* of any fold score is uncertain. Reproducibility, however, is
exact — the same label, fold and seed reproduce to eight decimal places across
separate processes and run orders (section 40).

## 17. Text-only ablation

NDCG@20 = 0.000115. Text alone is the weakest
content-only variant that still uses a published modality — below image-only,
and roughly half of text-and-image.

## 18. Image-only ablation

NDCG@20 = 0.000139. Image alone outperforms text alone
on this corpus. PixelRec is short-video: the cover image plausibly carries more
of what a viewer responds to than the title does. Stated as an observation, not
an explanation — nothing here isolates *why*.

## 19. Text-plus-image ablation

NDCG@20 = 0.000252, against
0.000115 for text and 0.000139 for
image. The two modalities are complementary rather than redundant: combining
them beats either alone by more than the gap between them.

**Do the published vectors earn their cost?** The `tag_only` control answers
this and nothing else does. It keeps the item tower but feeds it a single
categorical id — essentially free, and available for any catalogue — with no
text and no image. It scores 0.000028, roughly a tenth of
text-and-image. The published vectors are doing real work; they are not an
expensive way to encode a category the metadata already carries.

## 20. Full-model result

`text_image_tag` (text + image + tag + user-ID embedding, no item-ID residual)
scores 0.000259 on the screen. On the folds it is one
of the two indistinguishable leaders (section 14).

## 21. User-ID ablation

`full_no_user_id` disables the user-identity embedding, holding everything
else fixed: 0.000229 on the screen.

The folds are where this shows clearly. `full_no_user_id` is far behind every
other finalist at **both** origins — roughly a factor of six. This is the one
selection finding the folds establish comfortably rather than marginally.

## 22. Item-ID residual ablation

`full_with_id_residual` enables the warm item-ID residual:
0.000205, against
0.000259 without it. Enabling it **lowers** warm
accuracy on this corpus, and the selected configuration therefore does not use
it.

Diagnosed rather than assumed: the residual's norm (~0.13) is small beside the
content embedding's (~1.35), but it applies to every warm item consistently.
That consistent nudge is enough to push warm items above cold ones in the
ranking — with the residual on, the top-20 was 100% warm.

A **fully content-free** control was attempted and is not constructible: the
item tower refuses a configuration with no content inputs, because a tower with
no content is not an item tower. The content-free comparison in this phase is
therefore LightGCN and BPR, which are genuinely collaborative-only models,
rather than a crippled two-tower.

## 23. Pooling ablation

`mean_pooling` replaces recency-weighted history pooling with an unweighted
mean: 0.000314 on the screen, and the highest fold mean
of any finalist.

It is the selected configuration — but by a margin of roughly two per cent of
its own standard deviation, which is a tie broken by rule rather than a
measured advantage. Section 14 says so explicitly; a reader should not take
"selected" to mean "better".

## 24. Final strict metrics (official final test)

Fitted on train+validation, test scored **once**. The strict view counts a
cold target as a miss rather than excluding it — the honest denominator for a
production system, which does not get to skip the users it cannot serve.

| K | strict Recall@K | NDCG@K |
| --- | --- | --- |
| 5 | 0.00778 | 0.004948 |
| 10 | 0.01344 | 0.006763 |
| 20 | 0.02186 | 0.008873 |
| 50 | 0.0403 | 0.012488 |

| | |
| --- | --- |
| Coverage@20 | 0.7943 |
| Novelty@20 | 16.6576 |
| Gini@20 | 0.7583 |

## 25. Final warm metrics (official final test)

The warm view restricts to users whose target the model could reach at all. It
answers a different question from the strict view — "how well does it rank what
it can see?" — and the two are reported together so neither can stand in for
the other.

| K | warm Recall@K | NDCG@K |
| --- | --- | --- |
| 5 | 0.00778 | 0.004948 |
| 10 | 0.01344 | 0.006763 |
| 20 | 0.02186 | 0.008873 |
| 50 | 0.0403 | 0.012488 |


## 26. Final cold metrics (official final test)

Users whose held-out target is a cold item: **2270**
eligible cold targets, all of them content-representable and all present in the
index.

| K | cold Recall@K | NDCG@K |
| --- | --- | --- |
| 5 | 0.008811 | 0.00503 |
| 10 | 0.012775 | 0.006282 |
| 20 | 0.018062 | 0.007655 |
| 50 | 0.029515 | 0.009914 |

| | |
| --- | --- |
| cold NDCG@20 | 0.007655 |
| cold targets in catalogue | 227 |
| cold targets retrieved at 50 | 0.029515 of eligible |

**Cold Recall@K is positive**, which is the phase's completion requirement. It is positive at every measured cutoff (K = 5, 10, 20, 50). Recall@5 of 0.008811 means cold items reach even the shallowest cutoff measured.

Reported as measured; the cold-item definition was not adjusted to improve it.

Per-slice detail:

| slice | Recall@20 | NDCG@20 | users |
| --- | --- | --- | --- |
| items_cold_start | 0.018062 | 0.007655 | 2270 |
| targets_unreachable_cold |  |  | 0 |


## 27. Missing-modality metrics

| view | items | warm | cold | status |
| --- | --- | --- | --- | --- |
| both_modalities | 69347 | 69120 | 227 | present |
| text_only | 0 | 0 | 0 | empty on this corpus; path verified by fixture only |
| image_only | 0 | 0 | 0 | empty on this corpus; path verified by fixture only |
| no_modality | 0 | 0 | 0 | empty on this corpus; path verified by fixture only |

**This path is not exercised on real data.** PixelRec50K after k-core has
complete coverage of both modalities, so three of the four views are empty. The
handling exists and is verified by fixture (section 9), but it is **not**
reported as robust, because on this corpus that claim has no measurement behind
it. Dropping modalities artificially to manufacture a number would report a
property of the ablation labelled as a property of the data.

## 28. Candidate Recall@N

The ceiling Phase 6 inherits. A ranker cannot recover a target that retrieval
never proposed, so this is the hard upper bound on anything ranking can achieve.

| per-source budget | sources | pool depth | candidate Recall | users with target |
| --- | --- | --- | --- | --- |
| 50 | four_source | 200 | 0.06894 | 3447 |
| 50 | five_source | 250 | 0.09728 | 4864 |
| 100 | four_source | 400 | 0.11116 | 5558 |
| 100 | five_source | 500 | 0.15416 | 7708 |
| 200 | four_source | 800 | 0.17236 | 8618 |
| 200 | five_source | 1000 | 0.2312 | 11560 |
| 500 | four_source | 2000 | 0.2189 | 10945 |
| 500 | five_source | 2500 | 0.28874 | 14437 |
| 1200 | four_source | 4800 | 0.2189 | 10945 |
| 1200 | five_source | 6000 | 0.28874 | 14437 |

Two things to read here. Five-source is above four-source at every budget, by a
small margin. And the 1200 budget matches the 500 budget exactly — the sources
saturate, so buying more depth past 500 costs latency and returns nothing.

## 29. Two-tower standalone result

| system | NDCG@20 | Recall@20 | Coverage@20 | cold Recall@20 | unreachable cold |
| --- | --- | --- | --- | --- | --- |
| popularity | 0.004071 | 0.01148 | 0.000347 | 0 | 724 |
| matrix_factorization | 0.002836 | 0.00716 | 0.418113 | 0.000441 | 724 |
| lightgcn | 0.006108 | 0.01478 | 0.431901 | 0.001322 | 724 |
| sasrec | 0.003865 | 0.00948 | 0.461994 | 0 | 724 |
| two_tower | 0.008873 | 0.02186 | 0.79431 | 0.018062 | 0 |

Standalone, the two-tower is the **strongest** of the 5 sources on NDCG@20 — its NDCG@20 of 0.00887 is above the best collaborative source (`lightgcn`, 0.00611), and the paired interval excludes zero. It also has the highest Coverage@20 and the lowest exposure Gini of any source, meaning it spreads recommendations across the catalogue rather than funnelling them to a head.

It also has one column entirely to itself: it is the only source with **zero**
unreachable cold-target users. Every collaborative source leaves a block it
cannot serve at any depth.

## 30. Four-source RRF result

The Phase 4 blend, unchanged: popularity + BPR + LightGCN + SASRec, uniform
reciprocal rank fusion. It leaves the same 724 cold-target users unservable
that its individual members do — fusing four collaborative sources cannot
produce an item none of them can represent.

## 31. Five-source RRF result

| system | kind | NDCG@20 | Recall@20 | Coverage@20 | Novelty@20 | Gini@20 | cold Recall@20 | unreachable cold users |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| popularity | single | 0.004071 | 0.01148 | 0.000347 | 14.397813 | 0.99971 | 0 | 724 |
| matrix_factorization | single | 0.002836 | 0.00716 | 0.418113 | 14.320083 | 0.87956 | 0.000441 | 724 |
| lightgcn | single | 0.006108 | 0.01478 | 0.431901 | 14.307992 | 0.895204 | 0.001322 | 724 |
| sasrec | single | 0.003865 | 0.00948 | 0.461994 | 14.751045 | 0.925089 | 0 | 724 |
| two_tower | single | 0.008873 | 0.02186 | 0.79431 | 16.657637 | 0.758331 | 0.018062 | 0 |
| four_source_rrf | blend | 0.009475 | 0.022 | 0.375405 | 14.276662 | 0.941432 | 0.001322 | 724 |
| five_source_rrf | blend | 0.012884 | 0.02926 | 0.535582 | 14.489005 | 0.912095 | 0.001762 | 0 |
| five_source_weighted_rrf | blend | 0.011489 | 0.0247 | 0.412188 | 14.29343 | 0.913891 | 0.001322 | 0 |
| lightgcn_two_tower | blend | 0.011153 | 0.02538 | 0.752405 | 15.382018 | 0.778029 | 0.004846 | 0 |
| sasrec_two_tower | blend | 0.009548 | 0.02214 | 0.739369 | 15.60018 | 0.820043 | 0.011013 | 0 |

Rank-based fusion throughout. Scores are never summed across models: the
two-tower produces cosine similarities in a learned space and LightGCN produces
graph-propagated dot products, and no calibration puts those on a common scale.
Ranks are comparable by construction.

**Weighted RRF was also run and did not help.** Weights fixed in advance from
the standalone ordering (LightGCN 1.5, SASRec/popularity 1.0, BPR/two-tower
0.75) scored *below* uniform. They were not tuned, because tuning weights
against the test split is the same leak as selecting a model on it. Reported as
a negative result rather than dropped.

## 32. Unique two-tower contribution and per-source accounting

| | |
| --- | --- |
| targets reached **only** by the two-tower | 3492 |
| mean sources per candidate | 1.0864 |
| retrieval depth | 300 |

Pairwise overlap between sources:

| pair | Jaccard |
| --- | --- |
| lightgcn\|matrix_factorization | 0.044431 |
| lightgcn\|popularity | 0.007853 |
| lightgcn\|sasrec | 0.02344 |
| lightgcn\|two_tower | 0.027745 |
| matrix_factorization\|popularity | 0.008184 |
| matrix_factorization\|sasrec | 0.014907 |
| matrix_factorization\|two_tower | 0.009255 |
| popularity\|sasrec | 0.056099 |
| popularity\|two_tower | 0.012388 |
| sasrec\|two_tower | 0.028033 |
| unique_contribution::lightgcn | 0.836275 |
| unique_contribution::matrix_factorization | 0.87579 |

Fusion works here because the sources barely agree — a blend of near-identical
lists has nothing to combine. The two-tower has the least in common with the
others, which is what a content-based ranking should look like beside four
collaborative ones.

### What each source was asked for, and delivered

| source | requested | returned | fill rate | underfilled | failures | targets found | cold targets found |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lightgcn | 15000000 | 15000000 | 1 | 0 | 0 | 3959 | 36 |
| matrix_factorization | 15000000 | 15000000 | 1 | 0 | 0 | 1872 | 13 |
| popularity | 15000000 | 15000000 | 1 | 0 | 0 | 4695 | 52 |
| sasrec | 15000000 | 14999700 | 0.99998 | 1 | 0 | 4017 | 1 |
| two_tower | 15000000 | 14999700 | 0.99998 | 1 | 0 | 6336 | 185 |

Aggregate fusion metrics cannot distinguish "this source contributed nothing"
from "this source silently returned nothing". An underfilled list is a capacity
problem, a failure is a bug, and a full list that hits no target is a quality
problem — three different diagnoses that look identical in a blended NDCG.

### Which source's nominations survive into the blended top 20

| source | slots in top 20 | share |
| --- | --- | --- |
| lightgcn | 387487 | 0.387487 |
| matrix_factorization | 324671 | 0.324671 |
| popularity | 294590 | 0.29459 |
| sasrec | 382952 | 0.382952 |
| two_tower | 287569 | 0.287569 |

Shares sum to more than one: an item can be nominated by several sources, and
RRF has no notion of a single owning source. Read this as "appeared in the final
list having been nominated by S", not as exclusive attribution.

## 33. Paired bootstrap comparisons

Paired at user level, same resampled indices applied to both systems, fixed
seed, 95% intervals.

| challenger | baseline | metric | delta | 95% CI low | 95% CI high | users | resamples | significant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_tower | popularity | ndcg@20 | 0.004801 | 0.00409 | 0.005467 | 50000 | 1000 | True |
| two_tower | popularity | recall@20 | 0.01038 | 0.00876 | 0.01184 | 50000 | 1000 | True |
| two_tower | popularity | cold_recall@20 | 0.018062 | 0.012775 | 0.023789 | 2270 | 1000 | True |
| two_tower | matrix_factorization | ndcg@20 | 0.006037 | 0.005404 | 0.0067 | 50000 | 1000 | True |
| two_tower | matrix_factorization | recall@20 | 0.0147 | 0.01336 | 0.01612 | 50000 | 1000 | True |
| two_tower | matrix_factorization | cold_recall@20 | 0.017621 | 0.012335 | 0.023348 | 2270 | 1000 | True |
| two_tower | lightgcn | ndcg@20 | 0.002765 | 0.002038 | 0.003509 | 50000 | 1000 | True |
| two_tower | lightgcn | recall@20 | 0.00708 | 0.00538 | 0.00864 | 50000 | 1000 | True |
| two_tower | lightgcn | cold_recall@20 | 0.01674 | 0.011454 | 0.022467 | 2270 | 1000 | True |
| five_source_rrf | four_source_rrf | ndcg@20 | 0.003409 | 0.003001 | 0.003802 | 50000 | 1000 | True |
| five_source_rrf | four_source_rrf | recall@20 | 0.00726 | 0.0063 | 0.00826 | 50000 | 1000 | True |
| five_source_rrf | four_source_rrf | cold_recall@20 | 0.000441 | 0 | 0.001322 | 2270 | 1000 | False |
| five_source_rrf | lightgcn | ndcg@20 | 0.006777 | 0.006079 | 0.007464 | 50000 | 1000 | True |
| five_source_rrf | lightgcn | recall@20 | 0.01448 | 0.0131 | 0.015901 | 50000 | 1000 | True |
| five_source_rrf | lightgcn | cold_recall@20 | 0.000441 | 0 | 0.001322 | 2270 | 1000 | False |
| lightgcn_two_tower | lightgcn | ndcg@20 | 0.005046 | 0.004433 | 0.005585 | 50000 | 1000 | True |
| lightgcn_two_tower | lightgcn | recall@20 | 0.0106 | 0.00934 | 0.01172 | 50000 | 1000 | True |
| lightgcn_two_tower | lightgcn | cold_recall@20 | 0.003524 | 0.001322 | 0.005727 | 2270 | 1000 | True |

Taking each comparison in turn:

- `five_source_rrf` vs `four_source_rrf` — **cold_recall@20: not significant** (+0.000441, interval crosses zero). **ndcg@20: significant** (+0.003409, higher). **recall@20: significant** (+0.007260, higher).
- `five_source_rrf` vs `lightgcn` — **cold_recall@20: not significant** (+0.000441, interval crosses zero). **ndcg@20: significant** (+0.006777, higher). **recall@20: significant** (+0.014480, higher).
- `lightgcn_two_tower` vs `lightgcn` — **cold_recall@20: significant** (+0.003524, higher). **ndcg@20: significant** (+0.005046, higher). **recall@20: significant** (+0.010600, higher).
- `two_tower` vs `lightgcn` — **cold_recall@20: significant** (+0.016740, higher). **ndcg@20: significant** (+0.002765, higher). **recall@20: significant** (+0.007080, higher).
- `two_tower` vs `matrix_factorization` — **cold_recall@20: significant** (+0.017621, higher). **ndcg@20: significant** (+0.006037, higher). **recall@20: significant** (+0.014700, higher).
- `two_tower` vs `popularity` — **cold_recall@20: significant** (+0.018062, higher). **ndcg@20: significant** (+0.004801, higher). **recall@20: significant** (+0.010380, higher).

An interval that crosses zero is not evidence of a difference, however large the point estimate looks. Where a gain *is* significant, read its size before reading it as a win.

## 34. FAISS exactness verification

The index is `IndexFlatIP` — exact inner product — and its exactness is
verified against brute force rather than assumed. An index built with the wrong
metric or over a transposed matrix still returns k plausible neighbours for
every query and never raises.

Verification at build time:

| | |
| --- | --- |
| exact order agreement | 1.0000 |
| order agreement within ties | 1.0000 |
| unexplained disagreements | 0 |
| matches brute force | True |

Per-cohort verification against brute force, at depth 200:

| cohort | users | set agreement | order agreement | max score diff | index ms/query | brute ms/query | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_users | 64 | 1 | 1 | 0 | 4.3469 | 4.3417 | 1 |
| active_users | 64 | 1 | 1 | 0 | 4.348 | 4.35 | 1 |
| all_users_sample | 64 | 1 | 1 | 0 | 4.3309 | 4.3608 | 1.01 |
| warm_target_users | 64 | 1 | 1 | 0 | 4.3401 | 4.3495 | 1 |
| cold_target_users | 64 | 1 | 1 | 0 | 4.3303 | 4.3413 | 1 |

Cohorts are chosen to probe what an average would hide: a sparse user queries a
nearly-empty history, and a cold-target user exercises the content-only path no
collaborative source has.

The comparison is **tie-aware**. An earlier run reported 254/256 exact agreement
with a maximum score difference of 4.17e-07 — float32 tie-breaking between items
whose scores are equal to within representable precision. The fix was to make
the check tie-aware, not to loosen the threshold: a genuine ordering error and a
tie are different failures and only one is acceptable.

### Build, size and round trip

| | |
| --- | --- |
| index build time | 0.05 s |
| index size on disk | 35506229 bytes |
| save/load returns identical results | True |
| save/load max score difference | 0.000e+00 |

The round trip matters because the only index anyone queries in serving is a
*loaded* one. An index that answers differently after being written and read
back fails silently — both answers look plausible.

**The speedup over brute force is approximately 1.0x.** That is expected and
worth stating plainly: `IndexFlatIP` *is* an exhaustive scan. At this catalogue
size exactness costs nothing, and it also buys nothing in latency. An
approximate index would be the trade to make if latency mattered, and it would
have to be re-verified against this same brute-force reference.

## 35. Final model artifact

| | |
| --- | --- |
| artifact | `two_tower:phase5-two-tower-final` |
| payload | `artifacts/models/pixelrec50k/two_tower/phase5-two-tower-final` |
| type | retrieval_model |
| fitted on | train + validation |
| training data version | pixelrec50k@v1 |
| seed | 42 |
| device | CPU |
| git commit | `42ad33ecb414` |
| python | 3.11.15 |
| torch | 2.13.0 |

The official final test was scored **once**, after the configuration was locked.
Nothing in selection read it.

## 36. Final embedding artifact

`artifacts/embeddings/two_tower/phase5-two-tower-final/`

| | |
| --- | --- |
| items | 69347 |
| warm | 69120 |
| cold (content only) | 227 |
| excluded | 0 |
| dimension | 128 |
| dtype | float32 |
| normalisation | l2 |

Row order follows the catalogue's stable ordering, so row *i* is catalogue item
*i* in every consumer. Written with `.npy` so it can be memory-mapped back
rather than read whole. Non-finite values are rejected at write time — a NaN
that reached the index would produce confident nonsense at query time rather
than an error.

## 37. Final index artifact

`artifacts/indexes/pixelrec50k/two_tower/phase5-two-tower-final/`

| | |
| --- | --- |
| index type | flat_ip |
| metric | inner_product |
| items indexed | 69347 |
| warm items | 69120 |
| cold items | 227 |
| required index version | 1 |

Inner product is the metric because the towers are L2-normalised, which makes a
dot product a cosine similarity. Building under one convention and querying
under the other returns confident nonsense, so the normalisation policy travels
with the index rather than being assumed.

## 38. Artifact checksums

| Identity | Value |
| --- | --- |
| model checksum | `65c69e0128f0c9d56782548a10914e72` |
| embedding checksum | `e47f77b000ef33901831119473423570` |
| index checksum | `b2459e86d1b8403fce02e3c32b650144` |
| catalogue checksum | `65c69e0128f0c9d56782548a10914e72` |
| feature manifest | `dc80bc8c54d07ebe9867fa7d2fc442e2` |
| item mapping | `235fcff3343a651150e8f220bfebc9bd` |
| configuration hash | `d128_TIGU_mean_t0.07` |

A two-tower index has a way to be wrong that a collaborative one does not: its
embeddings derive from a feature store, so a store with different vectors —
same items, same mapping, different content — produces a different index that
nothing downstream would notice. Feature version and feature-manifest checksum
therefore travel with it.

## 39. Compatibility validation

The gate checks that model, index, feature store and id mapping all describe
the same thing:

- the registered model's `id_mapping_fingerprints.item` against the feature
  manifest's `item_mapping_checksum`;
- the embedding manifest's `model_checksum` against the index manifest's;
- the index manifest's `embedding_checksum` against the written matrix;
- `feature_version` and `required_index_version` against the loaded store.

`MultimodalFeatureStore.require_compatible` refuses a mismatch at load time
rather than reporting one later. A model paired with the wrong mapping resolves
every dense index to the wrong entity and still returns a plausible-looking
list, so this cannot be left to be noticed.

## 40. Save/load smoke test

The gate loads the registered artifact through `TwoTowerRetriever.load` — the
retrieval layer, never the bare `nn.Module`, which can encode but cannot
retrieve — and interrogates one real recommendation:

| Property | Result |
| --- | --- |
| loads without retraining | PASS — returned 10 candidates without retraining |
| contents well-formed | PASS — 10 unique items, finite descending scores, all ids in mapping, source=['two_tower'] |
| seen-item filtering | PASS — none of the user's 33 seen items appear when filtering; disabling the filter surfaces at least one of them |
| deterministic | PASS — identical ordering, max score difference 0.00e+00 |
| cold item retrievable | PASS — 770 of 770 cold items are in the model catalogue |

"Returned something" is far too weak a bar. An artifact can return ten duplicate
items, or NaN scores, or ids absent from the active mapping, and every one of
those looks like success to a length check while being useless downstream. Each
property above is a distinct way the artifact can be broken while loading
cleanly.

The seen-filter check searches for a user the filter actually bites on: for most
users the seen items never reach the top 20 either way, and asserting "no seen
item was returned" for such a user passes without testing anything.

## 41. Runtime measurements

| | |
| --- | --- |
| fold and seed runs recorded | 22 |
| total fitting time in those runs | 5 min |
| final refit (train+validation) | 2682.4 s |
| single-query retrieval @200 | ~4.3 ms (section 34) |

Sizing was measured before the grid was designed. The measurement that shaped
the most decisions: **MPS gives no speedup** — 51.5 s against 50.3 s on CPU,
because the bottleneck is memory-mapped feature reads rather than arithmetic.
Everything runs on CPU and the grid was sized for CPU throughput.

## 42. Memory measurements

| | |
| --- | --- |
| peak resident memory during fitting | 213 MB |
| embedding matrix | 69347 x 128 float32 |
| index size on disk | ~34 MB |

The feature store is memory-mapped rather than loaded, which is what keeps peak
memory at a few hundred MB against a 17 GB feature source.

## 43. Test results

Under `tests/unit/models/two_tower/`: dataset construction and padding
(`test_dataset.py`), towers and missing-modality tokens (`test_towers.py`),
contrastive loss and false-negative masking (`test_training.py`), save/load
identity (`test_persistence.py`), the retrieval surface (`test_generator.py`).

Under `tests/unit/retrieval/`: tie-aware index verification
(`test_two_tower_index.py`), fold construction (`test_fold_sequences.py`), fold
scoring (`test_fold_evaluation.py`), seed-before-construction ordering
(`test_fit_determinism.py`), the selection rule (`test_selection_rule.py`).

Under `tests/unit/scripts/`: config provenance and YAML validity
(`test_phase5_configs.py`), the gate's own behaviour
(`test_validate_phase5.py`).

Under `tests/integration/`: end-to-end training over a synthetic corpus
(`test_two_tower_training.py`) and the full retrieval path
(`test_phase5_retrieval.py`).

Three are worth singling out for what they catch rather than what they cover.
`test_fit_determinism.py` asserts the **call order** of seeding against model
construction, because the wrong order still produces a perfectly valid model.
`test_selection_rule.py` pins the unequal-footing case that made the tie-break
reward being measured less. `test_phase5_configs.py` found that the tracked
`phase5_selected.yaml` was **not valid YAML** — an unquoted colon in a
description turned the rest of the line into a nested mapping, and nothing
noticed because every consumer read the JSON record it was generated from.

## 44. Ruff result

`ruff format --check .` and `ruff check .` both clean.

## 45. MyPy result

`mypy --strict src` clean across all source files.

## 46. CI result

The `multimodal-retrieval` job installs the `retrieval` extra, runs the
two-tower fixture suites, and finishes with the CI-safe gate:

```yaml
- name: Phase 5 completion gate (CI-safe)
  run: |
    set -o pipefail
    python scripts/validate_phase5.py --ci | tee phase5-validation.log
```

`set -o pipefail` is load-bearing. Without it the pipeline reports *tee's* exit
status, so a failing validator produces a passing job — a failure that is
invisible from inside CI because everything is green. The gate checks its own
invocation for exactly this.

CI downloads no PixelRec data, loads no trained artifact and needs no GPU.

## 47. Phase 5 validator result

Two modes, and the difference between them is deliberate.

**CI-safe** (`--ci`) runs the deterministic fixture tests and records every
real-data check as **SKIP**. A skip is not a pass: it is "not looked at", it is
counted separately in the JSON report, and the mode is stamped as `ci` so no
consumer can mistake a green badge for real completion.

**Full local** (no flag) additionally verifies the registered artifacts, loads
them, interrogates a real recommendation, reads the real cold-recall number,
and checks the README.

Latest full-local run: **30/30
checks passed**, 0 critical failures,
0 warnings, 0 skipped.

All critical checks passed.

## 48. Known limitations

1. **Absolute accuracy is low, even where it leads.** Test NDCG@20 of 0.00887 is the highest of the five sources — above `lightgcn` at 0.00611 — but roughly nine users in a thousand get a hit in their top 20. Every comparison in this report is internal to this repository and this corpus; none of it says the system is good in any absolute sense.
2. **Its cold-start advantage rests on one corpus.** It beat LightGCN on cold items — cold Recall@20 of 0.018062 against LightGCN's 0.001322 — a factor of 13.7. The paired interval over the 2270 cold-target users is [+0.011454, +0.022467], which excludes zero: a real advantage. PixelRec50K has complete modality coverage, so every cold item is content-representable; a corpus with real gaps would not be so kind.
3. **Selection ran on a 5,000-user subset.** The final model is fitted on the
   full train+validation split, but the selection that chose it was not.
4. **The published vectors' encoders are unknown.** Recorded as `unknown`
   rather than guessed; no claim about a shared text/image space is relied on.
5. **Missing-modality handling is unexercised on real data.** Fixture-verified
   only (section 27).
6. **Two folds, three seeds.** Enough to catch an ordering reversal; not enough
   to put a confidence interval on a small margin.
7. **Fold evaluation cannot measure cold retrieval.** Within a fold every target
   is warm by construction, so the fold-level cold rate is reported as absent
   rather than as `0.0`.
8. **The selection did not transfer.** The fold-selected configuration did not
   beat its runner-up on test; both land near NDCG@20 of 0.0004.
9. **The exact index gives no speedup.** `IndexFlatIP` is an exhaustive scan
   (section 34).

## 49. Technical debt

- **Two independent scorers exist.** Split evaluation goes through
  `run_experiment`; fold evaluation has its own scorer in
  `fold_evaluation.py`, because a fold target is not a split. The metric
  definitions match and a test pins them, but they are two code paths that must
  be kept in agreement by hand.
- **`--reuse-screen` / `--reuse-folds` trust the CSV.** They verify the
  finalists are present but not that the rows came from the current code. A
  stale CSV from an older commit would be reused silently.
- **`KMP_DUPLICATE_LIB_OK` is set for FAISS/torch coexistence** (ADR-009).
  Justified because the exact brute-force test verifies no corruption, but it is
  a workaround, not a fix.
- **Weighted RRF weights are hand-set.** Fixed in advance rather than tuned,
  which is correct discipline but means the weighted variant is a single point
  rather than a search.
- **Fusion diagnostics re-run retrieval.** `source_diagnostics.csv` and
  `final_list_contribution.csv` each issue their own `recommend_batch` sweep
  rather than reusing the scoring pass, costing a few minutes per run.

## 50. Honest conclusion

Phase 5 delivered a working multimodal two-tower retriever, registered,
indexed, and evaluated on real PixelRec50K data. The engineering is sound:
exactness is verified rather than assumed, identity travels with every
artifact, selection is reproducible to eight decimal places, and the gate
distinguishes what it verified from what it skipped.

**The two-tower met its stated bar and exceeded it.** It was required to beat LightGCN on cold items: cold Recall@20 of 0.018062 against LightGCN's 0.001322 — a factor of 13.7. The paired interval over the 2270 cold-target users is [+0.011454, +0.022467], which excludes zero: a real advantage.

It is also the strongest single source on this corpus by NDCG@20 and Recall@20, with the widest catalogue coverage and the least concentrated exposure of the five.

Adding it to the blend moves NDCG@20 from 0.00948 to 0.01288 (+0.00341), and takes the count of completely unservable cold-target users from 724 to zero.

**Absolute numbers stay small.** An NDCG@20 in the hundredths is not a good
recommender by any external standard; it is the best this repository has
produced on a hard corpus with one implicit signal and a 69,347-item catalogue.
Every comparison in this report is internal, and none of it says the system is
ready for anyone.

**The largest single finding was a defect, not a model.** The first registered
final model was fitted with the development `--subset-users` default and could
answer for 5,000 of 50,000 users. It loaded cleanly, its checksums matched, and
every metric it produced was depressed by roughly an order of magnitude — which
looked exactly like a weak model. The gap was found by a per-source fill-rate
diagnostic showing the two-tower returning 30 candidates where it was asked for
300. Refitting on the full population changed every number in this report and
reversed its conclusion. A guard now refuses to register a final model that
cannot answer for the population it will be asked about.

The phase produced several other corrections worth as much as the metric: a
reproducibility defect that made results depend on process history, a selection
rule that rewarded being measured less, a tracked config that was not valid
YAML, and a CI gate that could have reported success for a failing validator.
Each was found by building the check, not by inspection.

## 51. Recommended Phase 6 scope

**Start here:** build the ranking dataset from the five-source candidate pool,
using the registered artifacts as frozen inputs. The exact starting command is
in the README's reproducibility section; the candidate pool it consumes is the
five-source RRF blend at the budget section 28 shows saturating (500).

What Phase 6 inherits, and must handle:

1. **A candidate-recall ceiling.** No ranker can recover a target retrieval
   never proposed. Section 28 is the hard upper bound.
2. **Cold items in the candidate set for the first time.** Ranking features must
   handle items with no interaction history — a case the Phase 3 and Phase 4
   sources never produced. Any feature that divides by an interaction count
   will fail on them.
3. **A two-tower whose ordering is now worth using.** On this corpus it is the
   strongest single source, so a ranking feature derived from its score is
   worth building — but validate it rather than assuming it, because that
   status is one refit old and rests on a single corpus.
4. **A saturating depth budget.** Retrieving past 500 per source costs latency
   and returns nothing.

Not in scope for Phase 6 and deliberately deferred: re-encoding the published
vectors, approximate indexing, and any attempt to improve the two-tower's
absolute accuracy — that is a modelling project, not a pipeline stage.
