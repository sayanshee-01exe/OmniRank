# FAISS vector index

[`src/omnirank/retrieval/faiss_index.py`](../../src/omnirank/retrieval/faiss_index.py)
· [ADR-004](../adr/ADR-004-faiss-initial-index.md) · [ADR-006](../adr/ADR-006-versioned-artifacts.md)

## What it is for

LightGCN and SASRec both end in a dot product between a query vector and every
item embedding. At 69,347 items that is a fast matmul, and the index buys little.
At ten million it is not, and the retrieval stage needs an approximate
nearest-neighbour structure.

The index is built now, at a size where **exact** search is still affordable,
precisely so its correctness can be established before it is needed.

## The failure mode this is built around

An index built with the wrong metric, over a transposed matrix, or from a
different model's embeddings **still returns `k` neighbours with plausible
scores for every query.** Nothing raises. Latency looks right. The results are
simply wrong.

Two defences follow from that.

### Checked against brute force, not against reasonableness

`brute_force_top_k()` is a deliberately naive exact reference — a direct matmul
(or squared-euclidean) and an argsort. Its only job is to be obviously correct.
Every flat index must agree with it **in both set and order**, for both metrics.

Measured on 500 × 32 random vectors, `flat_ip` matches brute force exactly, with
a maximum score difference of 5.7e-06 attributable to float32 accumulation order.

Both metrics are covered because building under one and querying as though it
were the other is the classic silent index bug, and an L2 path with no exact
reference is an untested path.

### Identity is checked at load

Index metadata records the model name, model version, embedding checksum, item
mapping checksum, dimension, vector count, and FAISS version.
`require_compatible()` refuses any mismatch.

Pairing an index with the wrong model is not a degraded result, it is a wrong
one: every returned id resolves through a mapping the vectors were not built
against, so every recommendation is a different item than intended. This is
[ADR-006](../adr/ADR-006-versioned-artifacts.md) applied to indexes.

## Filtering seen items

FAISS does not know a user's history, so returning `k` *unseen* items means
retrieving more than `k` and filtering.

Naively, "retrieve more until enough survive" is an unbounded loop, and the user
who triggers it is the heaviest user in the corpus — the one who has seen most of
the catalogue. In production that is a request that never returns.

`search_excluding()` grows the search buffer geometrically from
`oversampling_factor * k`, capped at `maximum_search_multiplier * k`. A user who
has seen almost everything gets a short list, padded with `EMPTY_SLOT` (`-1`),
rather than a hang. Verified against a user with 495 of 500 items seen.

`EMPTY_SLOT` is `-1` rather than `0`, because `0` is a valid item id and padding
that collides with a real recommendation is worse than no padding at all.

## Index types

| Type | Exact | Use |
| --- | --- | --- |
| `flat_ip` | yes | Default. The correctness reference. |
| `flat_l2` | yes | Euclidean equivalent. |
| `hnsw` | no | Graph-based; fast queries, slow builds, high memory. |
| `ivf_flat` | no | Cluster-based; needs training and a tuned `nprobe`. |

Approximate types are benchmarked **against `flat_ip`**, never against each
other. Recall relative to another approximation says nothing about whether either
is right.

## Input validation

The index rejects, before building:

- `NaN` or infinity — FAISS accepts both and then returns silent nonsense
- 1-D input — a common shape mistake that would otherwise be interpreted as a
  single high-dimensional vector
- empty matrices, and dimension mismatches between query and index
- `k < 1`

Every one of these is a case where FAISS's own behaviour is to proceed.

## What an index does not add

An index built from a model's item embeddings inherits **exactly** that model's
catalogue. LightGCN and SASRec are both collaborative, so items absent from
fitting are absent from their embeddings and therefore absent from the index.

Adding a vector index does not confer cold-start capability. It makes an existing
catalogue searchable faster.

## Related

- [Candidate aggregation](candidate_aggregation.md)
- [LightGCN](../models/lightgcn.md) · [SASRec](../models/sasrec.md)
