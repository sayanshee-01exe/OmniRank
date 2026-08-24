# Negative sampling

[`src/omnirank/models/baselines/negative_sampling.py`](../../src/omnirank/models/baselines/negative_sampling.py)

BPR learns from triples `(user, positive, negative)`. The sampler's only job is
to produce an item the user has **not** interacted with in the fit data.

## Why this is worth its own module and its own tests

Getting it wrong is a **silent** failure. Training against false negatives
teaches the model to rank genuine preferences downward, and nothing in the loss
curve reveals it — the loss goes down either way. It is only visible as
"the model is mysteriously worse than popularity", by which point the cause is
several layers away.

## Guarantees, each tested

**No sampled negative is a known positive**, for any user, verified over
thousands of draws including a 200-user / 500-item randomised case.

**Same seed, same samples.** `reset()` replays the stream exactly, so an epoch is
reproducible.

**Bounded work for dense users.** Naive rejection sampling degenerates as a user
approaches the full catalogue: expected retries are `1 / (1 - density)`, which
diverges. After a bounded number of rejection rounds the sampler switches to an
explicit complement draw, so a user who has seen 9 of 10 items terminates in the
same time as anyone else — and provably returns the one item they have not seen.

**A user who has interacted with the entire catalogue is rejected at
construction**, with a message saying to filter them before training, rather than
looping forever looking for a negative that does not exist.

## Implementation note: vectorised collision detection

The first implementation tested membership per row, which profiling showed was
**84% of training time** — `_isin_sorted` called 1.75 M times per epoch.

It now encodes each `(user, item)` pair as a single `int64` key
(`user * catalogue_size + item`), keeps one sorted array of positive keys, and
tests an entire `(batch, negatives)` block with one `searchsorted`. Training went
from **25 s/epoch to 2.9 s/epoch**, an 8.6× speedup, with identical loss.

The encoding is safe at this scale — 50,000 × 69,347 peaks near 3.5 × 10⁹, far
inside `int64`.

## Configuration

Uniform sampling is the baseline and the right first choice: it makes no
assumption about which negatives are informative, so a BPR result obtained with
it is not confounded by a sampling heuristic.

The sampler configuration — strategy, seed, catalogue size — is recorded in the
model artifact metadata, so a result can be traced to the sampler that produced
it.

`NegativeSampler` is a Protocol, so Phase 4 can add popularity-biased or
hard-negative samplers without touching the trainer.
