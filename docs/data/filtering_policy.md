# Filtering policy

Implemented in [`src/omnirank/data/filtering.py`](../../src/omnirank/data/filtering.py).

## Configuration

```yaml
filtering:
  enabled: true
  min_interactions_per_user: 3
  min_interactions_per_item: 2
  iterative: true
```

## Why it iterates

Removing users with too few interactions can push items below their threshold,
and removing those items can push further users below theirs. A single pass
leaves the k-core invariant unsatisfied. The loop therefore runs to a fixed
point — the round where nothing more is removed — and every round is recorded.

Both removals apply within the same round. Dropping users first and recomputing
item counts afterwards would double the number of rounds without changing the
fixed point.

A `MAX_ITERATIONS = 100` bound turns mutually unsatisfiable thresholds into a
loud failure rather than a spin.

## Applied once, before splitting

Filtering runs on the **whole interaction log**, before the split. Filtering the
three splits separately would give them different item vocabularies and
different user populations, and every comparison between them would silently be
a comparison of different datasets.

## Cold-start information is captured first

The population filtering removes — single-interaction items, short-history users
— is exactly the population cold-start analysis is about, and it is
unrecoverable afterwards. `snapshot_before_filtering` records it before the loop
runs, and the snapshot appears in the filtering report under `before`.

## Result on PixelRec50K

**Converged in 1 iteration.**

| | Users | Items | Interactions |
|---|---:|---:|---:|
| Before | 50,000 | 82,865 | 989,494 |
| After | **50,000** | **69,347** | **975,976** |
| Removed | 0 | 13,518 | 13,518 |

Pre-filtering snapshot:

| Metric | Value |
|---|---:|
| Singleton items | 13,518 |
| Items below the item threshold | 13,518 |
| Users below the user threshold | **0** |

The user threshold is inert on the full dataset — every user already has at
least 6 interactions. The item threshold removes exactly the 13,518 singletons,
and because each contributed one interaction, exactly 13,518 interactions go
with them. No user falls below 3 as a result, so the loop converges immediately.

## The `--subset-users` trap

A user subset destroys item density: sampling 300 of 50,000 users leaves almost
every item with a single interaction, and the item threshold then cascades to
zero across seven rounds.

This is the filter working correctly, not a bug, and the pipeline fails loudly
with an actionable message rather than emitting an empty dataset. For a
development subset, either raise the subset size or lower
`min_interactions_per_item` to 1.

## Disabling it

`enabled: false` returns the input untouched with an empty audit trail. Kept as
a real option so a run can *measure* filtering's effect rather than assume it.

## Reports

- `reports/data_quality/pixelrec50k/filtering/filtering_report.json`
- `reports/data_quality/pixelrec50k/filtering/filtering_report.md`

Both record the configuration, the pre-filtering snapshot, every iteration, and
the final population.
