# Multimodal two-tower: model core

[`src/omnirank/models/two_tower/`](../../src/omnirank/models/two_tower/)

## What it changes

Every retriever before this one represents an item by *who interacted with it*.
Popularity counts interactions. BPR factorises the interaction matrix. LightGCN
propagates along interaction edges. SASRec reads interaction sequences.

All four share a consequence: **an item nobody has interacted with has no
representation at all.** It is not badly represented, it is absent — excluded
from `fit_item_catalogue`, unreachable at any retrieval depth, and counted as a
guaranteed miss for the 880 PixelRec50K users whose held-out target is one of
the 770 cold items.

The two-tower item encoder represents an item by *what it is*: its published
text vector, its image vector, its category. All three exist the moment the item
is published, before anyone has seen it.

## The two towers

```text
history items ──► item content path ──► pooling ──┐
user id ──────────────────────────────────────────┼──► user MLP ──► u
history length ───────────────────────────────────┘

text 1024-d ──► projection ──┐
image 1024-d ─► projection ──┼──► gated fusion ──► item MLP ──► content
tag ──────────► embedding ───┘                                     │
                                                                   ▼
                              final = content + warm_mask × id_residual ──► v

score(u, i) = u · v
```

Both towers output the same configurable width because retrieval is a dot
product between them; the model refuses a mismatch rather than broadcasting.

## The cold-item guarantee

This is the one design decision the phase turns on.

```python
final_item_embedding = content_embedding + warm_item_mask * item_id_residual
```

An item id that never appeared in training has an embedding still at its random
initialisation. Adding it to a cold item's vector would inject noise of the same
magnitude as the learned signal — and the result would be neither content nor
identity, just a randomly perturbed content vector.

`warm_item_mask` is 0 for any item with no fitting interaction, so a cold item's
representation is **exactly** its content embedding. Not approximately, not
mostly: the tests assert `allclose(gated, content_only, atol=1e-6)` and a second
test asserts the residual is non-trivial for warm items, so the first cannot
pass by the residual simply being absent.

Verified against the real corpus: all 64 sampled PixelRec cold items encode
identically with and without the residual, are finite, and are unit-norm.

**Why this matters more than it looks.** If the gate leaks, nothing fails. The
model trains, the loss falls, warm metrics look normal, and cold recall reads
zero for a reason no warm number reveals. The failure is only visible if you
assert the mechanism directly.

## Missing modalities

An item may have text, image, both, or neither. Each modality is projected by
its own encoder carrying a **learned missing-token**:

```python
projected * mask + missing_token * (1 - mask)
```

A zero vector would not mean "no text" — it is one specific point in text space
that every text-less item would share, so the model would learn spurious
similarity between items whose only common property is an absent feature. The
missing-token makes absence a state the model can represent.

An item with neither modality falls back to its tag. An item with neither
modality *and* no tag cannot be represented from content, and
`MultimodalFeatureStore.content_representable` exists to say so rather than let
it be counted as a retrieval failure.

**On the real corpus this path is not exercised.** PixelRec50K has 100% coverage
of both modalities across all 69,347 items, so there are no text-only,
image-only, or no-modality items. The handling is implemented and fixture-tested
because the code must be correct for other corpora; the real-data result is
reported as *not exercised* rather than filled in.

## The user tower

History items are encoded through the item tower's **content path only** —
`encode_content`, not `forward`. Using the identity residual here would make a
user's query depend on item ids, and an unknown-user request built from a
supplied history would then require those ids to be warm.

Pooling is configurable:

| Strategy | Behaviour |
|---|---|
| `mean` | Every history position weighted equally. The ablation control. |
| `recency_weighted_mean` | Geometric decay by `recency_decay`; at 1.0 identical to mean. |

Histories are right-aligned, so the newest item is the last column and recency
weighting indexes from the end without needing the length.

Padded positions contribute nothing: zeroed before summing *and* excluded from
the divisor. A mean that divided by the padded width would shrink short
histories towards the origin, making history length a hidden magnitude signal.

An empty history pools to exactly zero rather than to whatever the
division guard leaves behind.

### Unknown users

| Case | Behaviour |
|---|---|
| Known user, with history | Identity embedding plus pooled history |
| Unknown user, history supplied | Query built from history alone; the id slot contributes zeros |
| Unknown user, no history | No personalised query — the caller falls back to popularity |

Phase 5 targets **new-item** cold start. A user with no history and no supplied
context has nothing to encode, and manufacturing a query for them would produce
a confident ranking derived from no evidence.

## Normalisation

With `l2_normalize` on, a dot product is a cosine similarity and FAISS's inner
product means the same thing. The rule is recorded in model metadata because an
index built under one convention and queried under the other returns confident
nonsense — the same class of failure as a mismatched item mapping.

## Related

- [Training](two_tower_training.md) · [Persistence](two_tower_persistence.md)
- [Feature store](../features/multimodal_feature_store.md)
