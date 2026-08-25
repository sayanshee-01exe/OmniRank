# Five-source fusion

Adding the multimodal two-tower retriever to the Phase 4 blend, and what it
changes.

Numbers live in `reports/metrics/phase_05/five_source_fusion_metrics.csv` and
are quoted in [phase_05_report.md](../phase_reports/phase_05_report.md). They
are not repeated here: a document that restates numbers it does not generate
goes stale the first time anything is re-run.

## The sources

| Source | Phase | Signal |
| --- | --- | --- |
| popularity | 3 | global frequency |
| matrix_factorization | 3 | BPR user/item factors |
| lightgcn | 4 | graph-propagated collaborative signal |
| sasrec | 4 | sequential, self-attentive |
| two_tower | 5 | multimodal content, warm-masked id residual |

All five implement the same `CandidateGenerator` interface, so fusion treats
them identically. That is deliberate: a source with a bespoke integration path
would be a source whose contribution is partly an artefact of its plumbing.

## How they are combined

Reciprocal rank fusion, unchanged from Phase 4 — see
[reciprocal_rank_fusion.md](reciprocal_rank_fusion.md):

```text
RRF(i) = sum over sources s that returned i of  w_s / (c + rank_s(i))
```

Equal weights, `c = 60`. Fusion consumes **ranks, not scores**, which matters
here more than anywhere else in the system: the two-tower produces cosine
similarities in a learned space and LightGCN produces graph-propagated dot
products. Those numbers are not on a common scale and no calibration makes them
one. Ranks are comparable by construction.

## What the fifth source contributes

Three distinct things, and they should not be conflated:

**1. Cold reachability.** Every Phase 3 and Phase 4 source is collaborative:
an item with no interactions has no representation, so it cannot be returned
at any depth. This leaves a block of cold-target users that the four-source
blend cannot serve at all — not "serves badly", *cannot serve*. The two-tower
represents cold items from content, so this count goes to zero.

This is the phase's actual result, and it is a **capability** difference rather
than a metric improvement. A recall number would show only the fraction of
those users who then got a hit; the count of users who had no chance shows the
thing that changed.

**2. Coverage.** The two-tower ranks over a catalogue partition the
collaborative sources concentrate away from, so the blend's Coverage@20 rises.

**3. A small accuracy gain — on one metric only.** Paired bootstrap over the
shared users puts the five-source-versus-four-source NDCG@20 delta at roughly
+0.0001 with a 95% interval that excludes zero: real, and entirely in the
fourth decimal place. The same comparison on Recall@20 has an interval that
**straddles zero** — adding the two-tower does not reliably put more correct
items in the top 20, it reorders the ones already there.

Read the interval, not the point estimate. `bootstrap_deltas.csv` carries both,
plus the comparison that matters most for calibration: the two-tower alone is
significantly *worse* than LightGCN alone, on both metrics, by a wide margin.

Note also what does **not** move: the blend's cold Recall@20 is unchanged with
and without the two-tower. It makes every cold item reachable; it does not rank
cold items better than LightGCN ranks the subset it could already reach.

## Why fusion works at all here

Because the sources barely agree. Pairwise Jaccard overlap between the returned
lists is low (`source_overlap.csv`), and RRF over near-identical lists has
nothing to combine. The Phase 4 report established this; the two-tower extends
it, since a content-based ranking has the least in common with the others by
construction.

## Method note: artifacts, not refits

The comparison **loads registered artifacts** rather than refitting each source.
Two reasons, in order of importance:

1. Every source is then the exact model that was registered and evaluated
   elsewhere, so a fusion number and a single-source number in the same report
   describe the same model. A refit would make them two different models with
   one name.
2. It avoids roughly two hours of redundant training.

The consequence is that fusion results are only as current as the registered
artifacts. When a source is retrained, fusion must be re-run.

That consequence was not hypothetical. An earlier fusion run scored SASRec at
roughly a fifteenth of its Phase 4 number, because the saved artifact returned
NaNs when loaded and scored in eval mode — a fully-masked attention row, on a
fast path PyTorch takes only in eval. Training was finite throughout, so the
in-process Phase 4 evaluation was unaffected and only artifact-loading
consumers saw it. Fusion is an artifact-loading consumer. The table was
regenerated against the retrained artifact rather than patched.

## Budgets

Per-source retrieval depth is swept over 50, 100, 200, 500 and 1200. Deeper
retrieval raises the candidate-recall ceiling that Phase 6's ranker inherits,
at proportional cost. The sweep is reported so that ceiling is a chosen
trade-off rather than an accident of a default.

See also [candidate_aggregation.md](candidate_aggregation.md) and
[../evaluation/candidate_recall.md](../evaluation/candidate_recall.md).
