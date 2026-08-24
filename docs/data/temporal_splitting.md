# Temporal splitting

Implemented in [`src/omnirank/data/splitters.py`](../../src/omnirank/data/splitters.py).
Rationale: [ADR-002](../adr/ADR-002-temporal-splitting.md).

## Strategy: per-user leave-last-N

```yaml
splitting:
  strategy: per_user_leave_last_n
  validation_interactions: 1
  test_interactions: 1
  minimum_history_before_validation: 1
```

For each **eligible** user, ordered chronologically:

```text
… earlier events …    →  train
second-to-last event  →  validation target
last event            →  test target
```

A user is eligible when they have at least
`validation_interactions + test_interactions + minimum_history_before_validation`
= **3** interactions.

### Worked example

| Position | Timestamp order | Split |
|---:|---|---|
| 0 | oldest | train |
| 1 | | train |
| 2 | | train |
| 3 | | **validation** |
| 4 | newest | **test** |

## Ineligible users are kept, not discarded

A user with fewer than 3 interactions contributes **all** of their events as
training history and appears in no evaluation set.

Discarding them would shrink the item catalogue and the collaborative signal for
no benefit. Evaluating them would mean scoring a user whose entire history is
the target — which measures nothing but popularity.

On PixelRec50K this case does not arise: all 50,000 users are eligible. The
policy is implemented and tested regardless, because a subset run or a future
dataset will produce it.

## Why leave-last-N rather than a global time cutoff

A global cutoff is more faithful to production, where one model serves everyone
from a fixed training moment. It is also the wrong instrument here: PixelRec's
interactions span 3,793 days with users active in very different periods, so any
single cutoff puts most users entirely on one side of it. The evaluation set
would be dominated by whoever happened to be active in the final window.

Leave-last-N evaluates *every* user on their own most recent behaviour, which is
the question a recommender actually answers.

A global-cutoff strategy remains available in the config
(`temporal_global`) and is a genuine option because PixelRec has real
timestamps — but it is not the default, and the trade-off above is why.

## Ordering

By `(external_user_id, timestamp, source_row_id)`. Details, including why
`interaction_order` is a per-user rank rather than a raw timestamp, are in
[`interaction_ordering.md`](interaction_ordering.md).

## Determinism

The split is a pure function of the input rows and the configuration:

- No RNG. The seed is recorded in `split_metadata.json` for the pipeline as a
  whole, but the splitter itself consumes no randomness.
- No dependence on input row order — sorting is explicit, and a test shuffles
  the input and asserts an identical result.
- `split_version` is recorded so two splits built by different protocol versions
  are never mistaken for each other.

## Result on PixelRec50K

| Split | Rows | Users | Items |
|---|---:|---:|---:|
| train | **875,976** | 50,000 | 68,577 |
| validation | **50,000** | 50,000 | 23,760 |
| test | **50,000** | 50,000 | 20,770 |

Eligible users 50,000 · ineligible 0 · minimum history 3 · ordering field
`timestamp`.

Exactly one validation and one test target per user, as the configuration
specifies.

**770 items appear only in held-out splits** and never in training. That is
genuine new-item cold start produced by the protocol, reported as a warning by
leakage check L11 and enumerated in the `items_cold_start` evaluation slice. It
bounds what any purely collaborative model can achieve, and is the population
Phase 4's content features exist to serve.

## Split metadata

`data/processed/pixelrec50k/split_metadata.json` records the strategy, version,
dataset version, ordering field, per-split row/user/item counts, eligible and
ineligible user counts, minimum history, configuration hash, and random seed —
everything needed to determine whether another build produced the same split.

## Verification

Six leakage checks assert the split's integrity on every run, and all pass with
zero violations on the full dataset. See
[`leakage_prevention.md`](leakage_prevention.md).
