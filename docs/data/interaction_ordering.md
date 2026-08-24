# Interaction ordering

## PixelRec50K has genuine timestamps

Confirmed by inspecting the file, not by assumption:

| Property | Value |
|---|---|
| Column | `timestamp` |
| Type | int64 |
| Unit | **Unix epoch seconds** |
| Range | 1328292931 → 1656087360 |
| As UTC | 2012-02-03T18:15:31Z → 2022-06-24T16:16:00Z |
| Span | 3,793 days |
| Non-positive values | 0 |

The values are plausible wall-clock instants spread over a decade, not a
sequence counter and not a relative offset. **No timestamps are invented**, and
the fallback described below is never exercised for this dataset.

## The ordering key

```text
sort by (external_user_id, timestamp, source_row_id)
```

`interaction_order` is then the **0-based position within that user's own
history** — not a global counter.

### Why a per-user rank rather than the raw timestamp

Three consumers need "the *n*th event of this user": the leave-last-N splitter,
the sequence builder, and every leakage check that asks "did this precede that?".
Expressing the ordering once, as an integer column, means all three share one
definition. If each re-derived an ordering from timestamps, a subtle
disagreement between them would be invisible and would silently corrupt the
split.

### Tie-breaking

`source_row_id` — the row's position in the original CSV — breaks ties.

PixelRec50K has **zero per-user timestamp ties** (measured: 0 duplicate
`(user_id, timestamp)` pairs), so the tiebreak never fires here. It is applied
unconditionally anyway, because behaviour that depends on a property of one
dataset is behaviour that breaks on the next one.

The tiebreak is deterministic and checksummed: `source_row_id` comes from a file
whose SHA-256 is recorded in the manifest, so the same file always produces the
same order.

## What is explicitly not done

**File order is never treated as chronological.** PixelRec50K's
`interaction.csv` is not sorted by timestamp — the first three rows are
2020-11-11, 2021-08-14, 2020-09-15. Nothing in the official documentation claims
row order carries meaning, so it is used only as a deterministic tiebreak, never
as a signal.

## If a future dataset has no timestamps

The contract supports it, and it is a documented degradation rather than an
invention:

1. If the source provides an explicit sequence position, use it as
   `interaction_order` directly.
2. If the source documents that file order is chronological, use it — and record
   `ordering_field` accordingly in the split metadata.
3. If neither exists, **the dataset cannot be split temporally**. Say so; do not
   manufacture an order.

`split_metadata.json` and `dataset_manifest.json` both record `ordering_field`,
so any consumer can see which of these applied. For PixelRec50K it reads
`"timestamp"`.

## Verification

`docs/data/leakage_prevention.md` checks L02–L04 assert the ordering is
respected across every split boundary, for every user. On the full dataset all
three pass with zero violations.
