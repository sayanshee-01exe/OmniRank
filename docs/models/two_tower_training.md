# Two-tower training

[`src/omnirank/models/two_tower/training.py`](../../src/omnirank/models/two_tower/training.py)
· [`losses.py`](../../src/omnirank/models/two_tower/losses.py)

## The objective

In-batch softmax (InfoNCE). For a batch of user vectors `u` and their positive
item vectors `v`:

```text
S[i][j] = (u_i · v_j) / temperature
```

Row *i*'s label is column *i*. Every other column is a negative.

This is what makes two-tower training affordable: one forward pass over B items
yields B × B comparisons, with no explicit negative sampler and no full softmax
over 69,347 items.

## In-batch negatives are not all negative

If user *i* has already interacted with the item that happens to be user *j*'s
target, the loss actively pushes user *i* away from an item they demonstrably
like.

At batch 512 over 69,347 items the collision rate is low — measured around 10–15%
of off-diagonal pairs on the fixture — but it is **not** zero, and it is
systematically biased towards popular items, precisely because those are the ones
most likely to be someone else's target. Left unmasked it trains the model to
under-rank popular relevant items, and nothing in a falling loss curve says so.

Two sources of false negatives are masked:

| Source | Detection |
|---|---|
| Duplicate targets in the batch | Two rows share a target item |
| Known user positives | The row's user history contains another row's target |

Masked entries are set to a large negative logit (`-1e4`) rather than removed,
because the softmax denominator must keep its shape for the batched
implementation to stay vectorised. A true `-inf` would produce `NaN` for a fully
masked row.

**The diagonal is never masked.** It is the label; masking it would remove the
only term the numerator has.

`masked_fraction` is reported per epoch, so a mask that has silently become
inert is visible rather than assumed.

## What the trainer guards against

Three failures that a contrastive run produces without announcing them:

**A non-finite loss trains a dead model.** One `NaN` propagates through every
parameter in a single backward pass; from then on the loss prints as `nan` and
the run completes normally. Checked per batch and raised, along with a
per-parameter finite-gradient check.

**A device that silently degrades.** MPS is used when available and falls back to
CPU explicitly. The resolved device is logged and stored in the artifact. An
unlogged fallback shows up only as an unexplained tenfold slowdown. CUDA is
never selected.

**Early stopping on the wrong quantity.** Stopping on *training* loss selects the
most overfit checkpoint available. Validation loss is the monitored signal, and
the **best** checkpoint is restored at the end — without that, early stopping
decides when to stop but not which weights survive.

Without a validation set the trainer falls back to training loss and records
that weaker guarantee in the history rather than implying the stronger one.

## The validation proxy

Early stopping monitors validation loss. Alongside it the trainer reports a
`proxy_recall@20`: each user's target ranked against **the other items in a
small validation subset**, not against the catalogue.

That makes it far easier than real retrieval and **not comparable to a reported
Recall@20**. It is a direction signal — a falling contrastive loss and a rising
retrieval quality are not the same claim, and this is the cheap way to watch
both. It is named `proxy` in the code, the config and the history for that
reason.

Full-catalogue retrieval costs orders of magnitude more and belongs to the next
milestone.

## Device and memory

The dataset is deliberately device-agnostic and torch-free: it produces numpy
arrays, and `_to_tensors` performs every transfer in one place. That is also
what makes a partially-moved model impossible — the model is moved whole, once.

The feature store is memory-mapped and read per batch. The 18.5 GB source
collection is never in RAM; the aligned matrices are 271 MB per modality on
disk, and a batch holds only the rows it asked for. Measured peak on the
1,000-user smoke run: **425 MB**.

## Measured behaviour

Fixture (40 items, 30 users, CPU): loss 2.67 → 0.59, in-batch accuracy
0.15 → 0.87 over 25 epochs.

Real PixelRec subset (1,000 users, 8,558 examples, 12.6M parameters, CPU,
3 epochs): loss 5.530 → 4.997, in-batch accuracy 0.004 → 0.022, proxy recall
0.041 → 0.076, roughly 3.7 s/epoch.

Three epochs is a smoke run, not a trained model. **No claim about retrieval
quality follows from it.**

## Related

- [Model core](multimodal_two_tower_core.md) · [Persistence](two_tower_persistence.md)
