# Reciprocal rank fusion

[`src/omnirank/retrieval/aggregation.py`](../../src/omnirank/retrieval/aggregation.py)

## Formula

```text
RRF(i) = sum over sources s that returned i of   w_s / (c + rank_s(i))
```

- `rank_s(i)` is 1-based, in the order source `s` returned its list
- `w_s` is the source weight, default 1.0
- `c` is `rrf_constant`, default 60.0

An item absent from a source contributes nothing from that source — not zero
score, no term at all.

## Worked example

Two sources, `k = 3`, default weights and `c = 60`:

| Source | Rank 1 | Rank 2 | Rank 3 |
| --- | --- | --- | --- |
| `bpr` | `a` | `b` | `c` |
| `popularity` | `d` | `e` | `b` |

| Item | Terms | Fused score |
| --- | --- | --- |
| `b` | `1/(60+2) + 1/(60+3)` | **0.032266** |
| `a` | `1/(60+1)` | 0.016393 |
| `d` | `1/(60+1)` | 0.016393 |
| `e` | `1/(60+2)` | 0.016129 |
| `c` | `1/(60+3)` | 0.015873 |

`b` wins. It is not first in either list, and it beats both items that are.

That is the entire value of the method: **two independent sources agreeing is
stronger evidence than one source being confident.** `a` and `d` tie exactly, and
the tie breaks on item id so the ordering is reproducible.

## Choosing `c`

`c` controls how much a top rank is worth relative to a lower one.

| `c` | rank 1 vs rank 10 | Behaviour |
| --- | --- | --- |
| 0 | 10× | Each source's first item dominates; fusion ≈ round robin over heads |
| 1 | 5.5× | Sharp; strongly favours the top of each list |
| 60 | 1.15× | Nearly flat; agreement across sources dominates |
| 1000 | 1.01× | Effectively counts how many sources returned the item |

The default of 60 comes from the original RRF paper and is deliberately flat: at
that setting, an item's number of *supporting sources* matters far more than
where in each list it appeared. That suits blending genuinely different model
families, which is what Phase 4 does.

A smaller `c` is the right choice when one source is known to be much better than
the others and its top ranks should carry more weight than a second opinion.

## Why rank and not score

Because scores are not comparable and no normalisation makes them so.

Min-max rescaling a popularity count and a SASRec logit puts both in `[0, 1]`, and
the result *looks* comparable. But the mapping is determined by each source's
observed spread within that one request — an outlier in either list silently
rescales everything else. The numbers become comparable in type, not in meaning.

Rank sidesteps this. "This source put it second" means the same thing regardless
of what the source's scores look like.

The cost is real and worth stating: **RRF discards confidence.** A source that
was near-certain about its top item and indifferent between ranks 2 and 3
contributes the same rank gaps as a source that was equally unsure about all
three. `normalized_score_union` exists for cases where that information is worth
the risk of using it.

## Weights

`w_s` scales a source's whole contribution. A weight of 0 keeps the source's
items in the pool but contributes nothing to their scores — which is a genuinely
useful setting: the items remain reachable through other sources' votes, and
provenance still records that the zero-weighted source proposed them.

Negative weights are rejected. A source that actively demotes items is a
different mechanism than a fusion weight, and expressing it here would make the
score no longer a sum of evidence.

## Related

- [Candidate aggregation](candidate_aggregation.md) — the other two strategies
