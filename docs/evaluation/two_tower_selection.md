# Selecting the Phase 5 two-tower configuration

Two stages, and the second overturned the first. That reversal is the reason
this document exists.

## Stage one — ablation screen

Nine configuration variants, evaluated on the **train → validation** boundary.
One origin, one fit each. Cheap enough to rank a whole grid.

The variants are deltas against `configs/models/two_tower.yaml`, never full
configurations. A variant that restated every field would silently stop
tracking changes to the baseline, and the difference between two such variants
would no longer be attributable to the input under test.

The grid lives in one place —
[`ABLATION_OVERRIDES`](../../src/omnirank/retrieval/fold_evaluation.py) — used
by both the screen and the fold confirmation, so the two cannot end up running
different models under the same labels.

## Stage two — rolling-fold confirmation

The top four variants are re-fitted on each pre-test rolling fold. Four,
not two, because the screen's ordering is measurably unstable — see below.

A fold is not a relabelled copy of the training split. Each fold picks an origin
at a per-user offset from the end of the log, uses only that user's pre-origin
interactions as history, and holds out the event at the origin as the target.
Offsets 3 and 2 are used; **offset 1 is the official test target** and
`build_fold` refuses it outright, so a selection run cannot reach it even by
mistake.

Selection is on the **mean fold NDCG@20**, and only when that mean leads by
more than the measured spread. The exact rule, and the reason it needs the
"only when" clause, is in stage three.

## What the two stages showed

**The screen is noise-dominated, and that is measured.** The nine-variant screen
was run twice, at the same subset size, from the same code, and produced
**disjoint top-two sets**. The top five variants sit inside a range narrower
than the seed spread later measured on the folds. A shortlist of two drawn from
it would be a coin flip presented as a ranking — which is why the finalist count
is four, and why the folds make the selection.

**The folds separate the field but not the leaders.** They establish
comfortably that disabling the user-identity embedding costs about a factor of
six, and that the published multimodal vectors beat a bare category embedding
by about a factor of ten. They do **not** separate the top two: those differ by
about two per cent of their own standard deviation.

## Stage three — seed verification, and the tie-break

The leading configurations are re-run at seeds 42, 43 and 44 on both folds. A
margin smaller than the seed spread is not a margin.

The rule, stated before the numbers were read:

1. Take the highest mean **only** if it leads the runner-up by more than the
   larger of the two standard deviations.
2. Otherwise the contenders are not distinguishable, and the tie-break is the
   **worst fold mean** — seeds averaged within each fold, then the lowest fold
   taken.

### A bug in the tie-break, found by applying it

The first version of rule 2 used the worst *single run*, and it selected the
configuration with the fewest runs. That was wrong: a minimum over runs
systematically favours whichever contender was measured **least**, because more
runs mean more chances to draw a low one. The rule was rewarding a smaller
sample — a property of the sampling, not of the model.

Averaging within folds before taking the minimum removes the bias, and the
selection becomes coherent: the winner has both the highest mean and the
highest worst-fold mean.

`tests/unit/retrieval/test_selection_rule.py` pins this, including the
unequal-footing case directly. Contenders may still be on unequal footing, and
the selection record logs `equal_footing` explicitly rather than presenting the
tie-break as stronger evidence than it is.

## What selection bought, and what it did not

**Bought:** two findings that reproduce at two origins and three seeds — the
size of the user-identity effect, and that the published vectors are not an
expensive way to encode a category the metadata already carries.

**Did not buy:** an accuracy win. Every configuration carried through to a
full-scale test fit lands near NDCG@20 of 0.0004. The fold apparatus improved
the *process* — the selection now rests on evidence that reproduces rather than
on a single-boundary gap that changed sign when re-run — which is worth having
and is not the same thing as a better model.

## Reproducibility

Runs are reproducible given a seed, and this was verified rather than assumed:
the same label, fold and seed reproduce to eight decimal places across separate
processes and different run orders.

It was not always so — see the reproducibility section of
[phase_05_report.md](../phase_reports/phase_05_report.md) for the defect that
made a run's result depend on how many runs preceded it, and
`tests/unit/retrieval/test_fit_determinism.py` for the guard.

## Cost

Selection ran on a 5,000-user subset. Full-corpus fitting was measured at
roughly 50 s/epoch per configuration, and the grid plus folds plus seeds at
full scale was not affordable. The **final** model is fitted on the full
train+validation split; the selection that chose it was not. That is a stated
limitation, not a hidden one.

MPS was measured and gives no speedup (51.5 s against 50.3 s on CPU) — the
bottleneck is memory-mapped feature reads. Everything runs on CPU.
