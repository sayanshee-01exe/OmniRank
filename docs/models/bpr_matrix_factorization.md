# BPR matrix factorization

[`src/omnirank/models/baselines/bpr.py`](../../src/omnirank/models/baselines/bpr.py)

## Why BPR for implicit feedback

PixelRec records that a user engaged with an item and **nothing about how much**.
There is no rating, no watch duration, no explicit signal — so there is nothing
to reconstruct, and a squared-error objective would be fitting a number the data
does not contain.

Bayesian Personalized Ranking optimises a *ranking* objective instead. It asks
only that an observed item outrank an unobserved one, which is exactly the claim
implicit feedback supports:

```text
L = -log sigmoid(y_ui - y_uj) + lambda * ||Theta||^2      y_ui = p_u . q_i
```

for a user `u`, an observed item `i`, and a sampled unobserved `j`.

## Implementation

PyTorch, sparse embeddings with `SparseAdam`, `softplus(-x)` for the log-sigmoid
(numerically stable where `log(sigmoid(x))` overflows).

The L2 term covers **only the embeddings the batch touched**. Penalising the
whole matrix would densify the gradient and undo the sparse update.

### Repeated interactions

Collapsed to **unique binary positives**. Measured on PixelRec50K's training
split: `interaction_count` maxes at 1 and there are **zero** duplicate
`(user, item)` pairs, so this changes nothing here. Making it explicit means a
dataset that *does* repeat cannot let a handful of heavily-repeated pairs
dominate the sampler. The policy is recorded in the artifact metadata.

### Device policy

CPU and Apple MPS. **Never CUDA implicitly** — `auto` selects MPS when available
and CPU otherwise; an explicit CUDA request without permission logs and falls
back rather than failing, because a slower run beats no run.

An unsatisfiable MPS request falls back to CPU with a logged reason, and the
device actually used is recorded in the artifact.

**No claim is made that MPS and CPU produce bitwise-identical results.** They do
not, and asserting it without testing it would be a reproducibility claim this
project has not earned. Determinism is guaranteed *within* a platform for a fixed
seed, and that is what the tests assert.

### Safety checks

A non-finite loss or gradient aborts training with the epoch and tensor named,
rather than producing a NaN-filled model that fails much later and much less
informatively.

## Retrieval

Full-catalogue top-K, memory-bounded:

```text
user batch -> matmul with item factors -> mask non-catalogue items
           -> mask seen items -> topk(max_k) -> internal ids -> external ids
```

A 50,000 × 69,347 matrix is never built. Items outside the fit catalogue are
masked to `-inf` — their factors are still at initialisation and mean nothing.
Positions that survive masking as `-inf` are padded with a `-1` sentinel and
filtered, so a padding id can never be emitted as a recommendation.

Tests assert: batched output equals the naive per-user output; the batch size
does not change results; seen items never appear; nothing outside the fit
catalogue appears; no sentinel leaks.

## Cold and unknown behaviour

| Case | Behaviour |
|---|---|
| Unknown user, `recommend` | **empty list** |
| Unknown user, `score` | 0.0 per item |
| Unknown item, `score` | 0.0, no exception |
| Cold held-out target | miss in strict, excluded in warm |

An unknown user gets nothing rather than a random embedding. A collaborative
model has no representation for them, and inventing one produces
confident-looking nonsense. Answering cold users is the serving fallback chain's
job (Phase 6), not this model's — and the offline evaluator counts the miss
honestly.

## Persistence

`torch.save` for tensors, JSON for configuration and mappings. Tensors are saved
**on CPU**, so an artifact trained on MPS loads on any host. Loading uses
`weights_only=True`: an artifact read from disk must not be able to execute code.

Round-trip tests assert recommendations are identical and scores match to
1e-9 after loading, and that corrupted files, missing files, wrong model types
and unsupported format versions each fail clearly.

## Reproducibility

Seeded: Python `random`, NumPy, torch CPU, torch MPS, batch shuffling, and
negative sampling. The artifact records the config, the seed, the loss history at
full precision, the sampler configuration, the device, the dataset manifest
identity, and the item-mapping checksum.
