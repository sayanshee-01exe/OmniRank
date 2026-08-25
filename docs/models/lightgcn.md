# LightGCN

[`src/omnirank/models/lightgcn/model.py`](../../src/omnirank/models/lightgcn/model.py)

## What it changes

BPR and LightGCN share an objective, a loss, a negative sampler, and a data
split. They differ in one thing: what a user embedding *is*.

In matrix factorization, a user's embedding is a free parameter, learned only
from that user's own interactions. In LightGCN it is that parameter *plus a
smoothed average of the items the user touched, plus the other users who touched
those items*, and so on outward for `num_layers` hops. Information reaches a user
from users they have never overlapped with, along paths through shared items.

That is the entire idea. There is no feature transformation and no non-linearity
— the two components the LightGCN authors removed from GCN after finding they
made recommendation accuracy worse, not better.

## The propagation

Users and items become nodes of one bipartite graph, laid out `[users | items]`,
with a symmetrically normalised adjacency matrix:

```text
A[u, i] = A[i, u] = 1 / sqrt(deg(u) * deg(i))     if u interacted with i
        = 0                                        otherwise
```

Propagation is repeated multiplication, and the output is the **equally weighted
mean** of every layer:

```text
E_0 = base embeddings
E_k = A @ E_{k-1}
E   = (E_0 + E_1 + ... + E_K) / (K + 1)
```

The normalisation is the part worth understanding. Without `1/sqrt(deg)`, a node
with a thousand edges would contribute a thousand times more signal than a node
with one, and popular items would dominate every propagated embedding — the model
would rediscover popularity through the graph. Dividing by the square root of
*both* endpoint degrees damps that on both sides while keeping the matrix
symmetric.

The layer weights are fixed at `1/(K+1)` rather than learned. This is the
paper's choice and it is kept: learned layer weights add parameters that, in the
original ablation, did not improve accuracy.

### Why `num_layers = 0` is in the search space

With zero layers, `E = E_0`. Propagation does nothing, and LightGCN **is** matrix
factorization — trained by the same code, on the same data, with the same
objective, the same sampler, and the same seed.

That makes it a controlled ablation rather than a comparison across two
codebases. If LightGCN at 2 layers beats LightGCN at 0 layers, the difference is
attributable to the graph and nothing else. Comparing against the Phase 3 BPR
implementation instead would confound propagation with every incidental
difference between two implementations.

The measured result is in the
[Phase 4 report](../phase_reports/phase_04_report.md).

## Isolated nodes

An item with no interactions in the fitting split has degree zero. `1/sqrt(0)` is
infinity, and a single infinity propagates to `NaN` across the whole embedding
matrix within one layer.

Isolated nodes are therefore given a zero row rather than an infinite one. They
receive nothing and contribute nothing, which is the correct behaviour: there is
no evidence about them to propagate. The graph build reports how many it found
(`isolated_nodes` in `lightgcn.graph_built`) rather than silently absorbing them,
because a sudden jump in that count means the upstream split changed.

## The graph checksum

A LightGCN model is only meaningful next to the adjacency it was trained on. Its
propagated embeddings *encode* that adjacency; served against a different graph
they are not merely stale, they are wrong in a way that produces confident,
plausible recommendations.

So the edge set is checksummed at build time, stored in the artifact, and
`require_graph()` refuses a mismatch. This is the same argument as the item
mapping checksum in [ADR-006](../adr/ADR-006-versioned-artifacts.md), applied to
the structure rather than to the ids.

## Device handling

Training runs on MPS where available and falls back to CPU otherwise. The
fallback is not assumed from the device name: sparse matrix multiplication is
probed on the target device at fit time, and a failure logs
`lightgcn.mps_sparse_unsupported` and falls back explicitly. A silent CPU
fallback would show up only as a confusing tenfold slowdown.

CUDA is never selected implicitly.

## What it cannot do

LightGCN is purely collaborative. An item that appears in no fitting interaction
has no node, no embedding, and no path by which one could be inferred — it is
unreachable, and `fit_item_catalogue` excludes it.

**This is not a cold-start model, and adding a vector index over its embeddings
does not make it one.** The index inherits exactly the catalogue the model was
fitted on. Cold-start coverage requires content features, which arrive in Phase 5.

## Cost

Measured on this hardware (Apple silicon, MPS), on the 875,976-edge PixelRec50K
training graph:

| Configuration | Seconds per epoch |
| --- | --- |
| `d=64, L=2` | 11.0 |
| `d=128, L=3` | 20.0 |

Propagation is recomputed each step over the full graph, so cost grows with
`num_layers` roughly linearly and is dominated by the sparse matmul rather than
by the embedding width.

## Related

- [BPR matrix factorization](bpr_matrix_factorization.md) — the `num_layers=0` case
- [Model selection](model_selection.md) — how a configuration is chosen and locked
- [ADR-007](../adr/ADR-007-baselines-before-advanced-models.md) — why baselines come first
