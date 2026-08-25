# Candidate aggregation

[`src/omnirank/retrieval/aggregation.py`](../../src/omnirank/retrieval/aggregation.py)

## The problem

Four generators each return a ranked list. The ranker downstream needs one list.
Merging them is not obvious, because the lists are not comparable:

| Generator | What its score is |
| --- | --- |
| `popularity` | a time-decayed interaction count, in the hundreds |
| `matrix_factorization` | a dot product of embeddings, order 1 |
| `lightgcn` | a dot product of *propagated* embeddings, order 1 |
| `sasrec` | an attention-derived logit, unbounded |

Sorting the union by raw score means popularity wins everything. Not because it
is better — because its numbers are bigger.

## The three strategies

### Weighted round robin

Interleave the sources, giving each a number of turns proportional to its weight,
deduplicating as it goes.

Compares nothing across sources; it only ever compares a source with itself. That
makes it the safest default and the least clever: it cannot recognise that two
sources agreeing on an item is evidence.

### Reciprocal rank fusion — the default for cross-model blending

```text
RRF(i) = sum over sources s of  w_s / (c + rank_s(i))
```

Uses **rank**, the one quantity that means the same thing in every list. An item
ranked 2nd by BPR and 3rd by popularity scores `1/62 + 1/63 = 0.0323`, beating an
item ranked 1st by one source alone at `1/61 = 0.0164`.

That property — agreement between independent sources outranks a single strong
vote — is the reason to fuse at all. A fusion that cannot express it is just a
concatenation.

The constant `c` (default 60) damps the top ranks. Without it, `1/rank` makes
every source's first result overwhelm everything else, and fusion degenerates
into round robin over the top items. Smaller `c` sharpens the advantage of high
ranks; larger `c` flattens the sources towards equal contribution.

### Normalised score union

Rescale each source's scores independently, then sum with weights. Three
normalisations are available: `min_max`, `z_score`, `rank_percentile`.

This is the only strategy that uses score *magnitude*, so it is the only one that
can express "BPR was very confident about this one". It is also the only one that
can be wrong about it, since a rescaled dot product and a rescaled decayed count
are still not measuring the same thing. `rank_percentile` is the default because
it discards magnitude, which makes it RRF-like but with a different rank curve.

## Over-retrieval

Each source is asked for `over_retrieval_factor * k` candidates, not `k`.

Fusing four top-20 lists and truncating back to 20 returns fewer than 20 distinct
items whenever the sources agree — and sources that agree are exactly the case
fusion is meant to reward. Without over-retrieval, the better the sources agree,
the shorter the output gets.

## Determinism

Ties break on item id in every strategy, and the output never depends on the
order sources appear in the input dictionary.

This matters more than it looks. Retrieval output becomes ranker training data in
Phase 6. An ordering that depended on dict iteration order would make every
downstream experiment irreproducible, and the cause would be invisible — the
metrics would just wobble between runs for no stated reason.

## The audit trail

`AggregationResult` carries two diagnostics beside the candidates.

**`contributions`** — per source, how many of the *emitted* candidates it
nominated. Counted over what was emitted, not what was offered: a source that
supplied 200 candidates which all lost their tie-breaks contributed nothing to
the list the ranker actually saw, and the number has to say so. A shared item
credits every source that nominated it, so the total can exceed the number of
candidates — that excess measures source overlap, which is worth reading on its
own, since sources that agree completely add no coverage.

**`degraded_sources`** — sources that returned nothing. This is a legitimate
runtime state, distinct from a wiring error: aggregating with *no sources at all*
raises, because returning an empty result there would disguise a misconfigured
pipeline as an ordinary empty response.

Together they answer "why did recall drop", which is otherwise one of the harder
questions to answer about a multi-stage retriever after the fact.

## Blending as a retriever

[`BlendedRetriever`](../../src/omnirank/retrieval/blended.py) wraps a set of
fitted generators and an aggregator, and presents the result as a single
`CandidateGenerator`.

This is deliberate: a blend is then measured by exactly the same
`run_experiment` driver that measures a single model. A hybrid evaluated through
its own bespoke path would reintroduce the comparability problem the shared
harness exists to prevent.

It refuses `fit`, `save` and `load`. There is no coherent meaning for "training a
fusion" of four independently-trained objectives, and persisting a blend would
copy every source's weights into a second artifact that can drift from the
registered originals. A blend is reconstructed from the registry instead.

## Related

- [Reciprocal rank fusion](reciprocal_rank_fusion.md) — the arithmetic in detail
- [FAISS index](faiss_index.md)
- [ADR-001](../adr/ADR-001-modular-monolith.md) — why the stages are separate
