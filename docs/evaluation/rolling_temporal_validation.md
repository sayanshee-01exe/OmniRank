# Rolling temporal validation

[`src/omnirank/data/rolling.py`](../../src/omnirank/data/rolling.py)

## The problem it exists to solve

Phase 3 ended with a result that could not be explained by model quality: the
ordering of the baselines **reversed** between validation and test.

Diagnosis showed the two splits are not exchangeable. Test targets are roughly
twice as concentrated on popular items as validation targets (the top-24 items
cover 2.036% of test targets against 0.996% of validation targets) and are more
recent (median age 110 days against 172).

Both splits are correct. They are simply different weeks, and a single held-out
origin cannot distinguish **"this model is better"** from **"this model suits
this particular week"**.

Rolling-origin validation answers that by measuring the same model at several
temporal origins. A model that wins at every origin has a property; a model that
wins at one has a coincidence.

## How a fold is built

Each user's interactions are ordered. A fold at `offset = n` holds out the
user's *n*-th-from-last interaction as the target, and everything strictly before
it becomes history:

```text
user history:   e1  e2  e3  e4  e5  e6
offset = 2:     [------history------] target=e5, excluded=e6
offset = 3:     [--history--] target=e4, excluded=e5,e6
```

Interactions after the target are **excluded**, not used as history. Including
them would be exactly the future leakage the temporal split exists to prevent.

Every row is labelled `history`, `target`, or `excluded`, and folds are
checksummed over the `(user, order, role)` triples so two builds of the same fold
hash identically.

## Offset 1 is reserved and refused

`RESERVED_TEST_OFFSET = 1` is the official Phase 2 test target. `build_fold`
**raises** if asked for it, and `check_no_reserved_offset_used()` asserts no
selection fold touched it.

This is a hard error rather than a warning because the failure it prevents is
invisible: a selection run that quietly included the test target would produce
better-looking numbers and no symptom whatsoever. The default offsets are
`(3, 2)`.

## Verification against the official split

`fold_offset_2` reproduces the Phase 2 validation split **exactly** —
875,976 history rows and 50,000 targets, matching the official split row for row.

That is the check that makes the rest trustworthy. The fold builder is not an
approximation of the split logic; at the offset where they should coincide, they
do. `fold_offset_3` yields 825,976 history rows and 49,999 targets, with one user
excluded for insufficient history.

## Eligibility

A user needs at least `minimum_history` interactions before the target to be
eligible. Users below the threshold contribute history but no target — evaluating
a user whose entire history *is* the target measures nothing but popularity.

Excluded users are counted per fold rather than dropped silently, because a jump
in that count means the fold reached back further than the data supports.

## What it does not do

Rolling validation does not make the test split unnecessary, and folds are never
used to report final numbers. It is a **selection-time** instrument: it says
whether a configuration's advantage is stable across origins before that
configuration is locked.

`assert_not_final()` enforces this for the sampled-negative protocol on the same
principle — a selection-time shortcut must not become a reported result.

## Related

- [Offline evaluation protocol](offline_evaluation_protocol.md)
- [Strict vs warm evaluation](strict_vs_warm_evaluation.md)
- [ADR-002](../adr/ADR-002-temporal-splitting.md)
