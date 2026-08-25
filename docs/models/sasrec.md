# SASRec

[`src/omnirank/models/sasrec/model.py`](../../src/omnirank/models/sasrec/model.py)

## What it changes

Every other retrieval model in OmniRank sees a user as an unordered set. BPR and
LightGCN would produce identical embeddings if a user's history were shuffled.

SASRec does not. It reads the history in order and predicts what comes next. On a
short-video corpus that ordering is real signal — sessions are long, interest
drifts within them, and what someone watched three items ago predicts the next
item better than what they watched three months ago.

## The architecture

```text
item embedding + positional embedding
  -> N x [causal self-attention -> feed-forward], residual + layer-norm
  -> final non-padding hidden state
  -> dot product against item embeddings
```

Sequences are **left-padded and right-aligned**, so the most recent item is
always in the final column. Inference reads that column and nothing else needs to
know how long the sequence was. Truncation to `maximum_sequence_length` keeps the
**newest** items and drops the oldest, which is the opposite of the natural
slicing mistake and the whole point of a sequential model.

## Causality is the correctness property

Position *t* may attend to positions ≤ *t* and never beyond.

If it could see ahead, the model would learn to read the answer out of its own
input. Training loss would fall *faster*. Offline metrics would *improve*. Recall
and NDCG would look excellent, and every number would be meaningless.

There is no symptom. No exception, no `NaN`, no failing metric, nothing in the
loss curve to notice. A future-leaking transformer looks exactly like a very good
one right up until it is deployed and predicts nothing.

So causality is asserted directly rather than inferred:

- perturb the last position of a sequence, and require earlier hidden states to
  be **bit-identical** — not close, identical;
- perturb a middle position, and require everything before it to be unchanged and
  everything from it onward to move;
- check the mask blocks exactly the strict upper triangle.

The mask is boolean rather than additive `-inf`, because torch deprecates mixing
a float attention mask with the boolean padding mask and both are needed here.

## Padding is `num_items`

The padding token is one past the last valid internal item id.

Reusing `0` would collide with a real item — under the Phase 2 global mappings,
id 0 is a genuine video — and every padded position would silently train the
model to predict it. The alternative, shifting all item ids by one to free up
zero, would mean the model's ids no longer matched the mappings every other
component uses.

So the embedding table has `num_items + 1` rows, the extra one is padding, and
it is:

- masked in attention, so no real position attends to it;
- excluded from the loss, so it never becomes a training target;
- sliced off the scoring matrix, so it can never be recommended.

`item_embeddings()` excludes it too. An index built over the padding row could
return it as a nearest neighbour, and it would resolve to no item at all.

## The objective

Sampled binary cross-entropy over next-item positions: one positive and
`negatives_per_positive` sampled negatives at every valid position.

Not a full softmax. A 69,347-way softmax at each of up to 50 positions across
776k training sequences is not practical, and no profiling was needed to
establish that. Sampled BCE is the standard SASRec formulation.

Only positions whose next item is a real item contribute; padded positions are
masked out of the loss entirely.

## Unknown users

`recommend()` returns an empty list for a user with no history, rather than
falling back to popular items.

A sequential model's input *is* the sequence. With nothing to encode there is no
query vector, and manufacturing one — a zero vector, an average user — produces a
confident ranking derived from no evidence about that user. An empty list is
honest and lets the aggregator's fallback chain do its job, where the choice of
fallback is explicit and logged.

## Cost, and what that meant for the search

Measured on this hardware (Apple silicon, MPS), on 775,977 training sequences:

| Configuration | Seconds per epoch |
| --- | --- |
| `L=50, d=64, blocks=2` | 103.1 |
| `L=50, d=128, blocks=2` | 189.7 |

A LightGCN epoch on the same machine costs 11–20 seconds. **SASRec is roughly ten
times more expensive per epoch**, and the staged search across folds and seeds
that LightGCN receives is not runnable in the same budget.

The grid was therefore cut to sequence length alone, at fixed width and depth.
`num_blocks`, `num_heads`, `dropout` and `learning_rate` were **not** searched.
That is a limitation of what was run, not a finding about which axes matter, and
the [Phase 4 report](../phase_reports/phase_04_report.md) records it as such.

## What it cannot do

SASRec is collaborative. Item representations are learned from co-occurrence in
sequences, so an item never seen during fitting has no embedding and cannot be
recommended. It makes no cold-start claim.

## Related

- [LightGCN](lightgcn.md) — the set-based counterpart
- [Model selection](model_selection.md) — how a configuration is chosen and locked
- [Rolling temporal validation](../evaluation/rolling_temporal_validation.md)
