# Candidate recall

[`src/omnirank/retrieval/diagnostics.py`](../../src/omnirank/retrieval/diagnostics.py)

## The question Recall@20 cannot answer

`Recall@20 = 0.015` conflates two entirely different failures:

1. the target **was never retrieved** — it was not in the candidate pool at all;
2. the target **was retrieved but ranked below 20**.

Only the first is retrieval's fault, and it is the one that matters most,
because **no amount of ranker work in Phase 6 can fix it.** A target absent from
the candidate pool is unreachable by every downstream stage. Candidate recall is
the ceiling the whole pipeline inherits.

The second failure is what a ranker exists to fix, and improving it is Phase 6's
job.

Reporting only Recall@20 makes these indistinguishable, which is how teams spend
a quarter tuning a ranker against a ceiling that was set months earlier by the
retrieval stage.

## Definition

```text
candidate_recall@d = users whose target appears in the top-d candidate pool
                     -------------------------------------------------------
                                users with a held-out target
```

Depth `d` is normally much larger than the reporting cut — the pool handed to a
ranker is hundreds of candidates, not 20. Measuring at several depths shows how
much headroom buying a deeper pool would actually purchase.

## Reachable candidate recall

A target that no generator *could* return — a cold item, absent from every
model's fitting catalogue — is not a retrieval failure. It is a coverage gap, and
it is fixed by content features, not by better retrieval.

So two numbers are reported:

- **`candidate_recall`** — over all evaluated users. The honest, absolute figure.
- **`reachable_candidate_recall`** — over users whose target was reachable in
  principle. The ceiling retrieval could actually have hit.

The gap between them is the cold-item problem, sized. The gap between
`reachable_candidate_recall` and 1.0 is the retrieval problem, sized. They are
closed by different work, which is why they are not averaged together.

This mirrors the [strict vs warm](strict_vs_warm_evaluation.md) distinction
already used for accuracy metrics: strict never hides a cold target, warm asks
what was achievable.

## Source overlap

The second diagnostic answers whether the ensemble is an ensemble.

Fusing four generators that return nearly the same list costs four times the
compute for nearly one generator's coverage. Three figures make that visible:

- **`pairwise_jaccard`** — mean per-user Jaccard index for each source pair.
  Two sources at 0.9 are one source with extra steps.
- **`mean_sources_per_item`** — how many sources proposed each retrieved item, on
  average. Near 1.0 means the sources are disjoint; near the source count means
  they are redundant.
- **`unique_contribution`** — per source, the mean fraction of its items no other
  source proposed. A source at 0.0 is contributing nothing the others do not
  already supply, and could be dropped without changing the pool.

Two empty sources score 0.0 together, not 1.0. Agreement by vacancy is not
similarity, and scoring it as identity would make two silent generators look
exactly like two that returned the same list — which is the failure this
diagnostic exists to catch.

## Reading the two together

| Candidate recall | Overlap | What it means |
| --- | --- | --- |
| Low | Low | Sources are diverse but individually weak. Improve the models. |
| Low | High | Sources are redundant *and* weak. Add a different model family. |
| High | Low | Healthy ensemble. Remaining loss is a ranking problem. |
| High | High | One source may be carrying the rest; test dropping the others. |

## Related

- [Offline evaluation protocol](offline_evaluation_protocol.md)
- [Strict vs warm evaluation](strict_vs_warm_evaluation.md)
- [Candidate aggregation](../retrieval/candidate_aggregation.md)
