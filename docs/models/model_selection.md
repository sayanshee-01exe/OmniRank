# Model selection protocol

## The rule

**Hyperparameters are chosen on validation. Test is read once, afterwards.**

Everything below exists to make that structurally difficult to violate rather
than merely intended.

## Two boundaries

```text
selection   fit: train              targets: validation   history: train
final       fit: train+validation   targets: test         history: train+validation
```

`boundary_for_stage()` is the only place these are defined. `scripts/evaluate.py`
derives the fit boundary from `--split` rather than accepting it as a flag, so an
optimistic combination cannot be requested by accident.

## The lock

1. Every candidate is fitted on train and scored on validation.
2. **Every trial** is appended to `validation_runs.jsonl` as it happens — so a
   later selection cannot quietly omit a configuration that did badly.
3. The winners are written to `selected_configuration.json`.
4. Only then may test be touched. `compare_baselines.py --stage final` **refuses
   to run** without that file.

The selection file is the commitment. It records the selection rule, the
boundary, the dataset identity, and the chosen configuration with its validation
metrics.

## Selection criteria

Primary **validation NDCG@20**; ties broken by Recall@20, then runtime, then
memory.

NDCG@20 is primary because it is rank-sensitive — Recall@20 cannot distinguish a
hit at position 1 from one at position 20. Under one-positive ground truth
Recall@20 is identical to HitRate@20, so quoting both adds nothing
([`../evaluation/metric_definitions.md`](../evaluation/metric_definitions.md)).

## Refit from a clean initialisation

The final model is **retrained from scratch** on train+validation, not
warm-started from the selection run. Warm-starting would carry state fitted
without validation data into a model that claims to be fitted with it.

## Multi-seed

The selected configuration is retrained with several seeds and the mean and
standard deviation are reported, so a single lucky initialisation is not
presented as the model's performance. The registered artifact uses a documented
seed rather than the best one.

## Grids: small and justified

Not a full cartesian sweep. The BPR grid covers both embedding sizes, both
learning rates, both regularisation strengths, and both negative counts in six
runs rather than sixteen — sixteen would take an hour to say the same thing.

The epoch budget was chosen from **measured** convergence, not guessed:

| Epochs | Final loss | Validation NDCG@20 | Fit time |
|---:|---:|---:|---:|
| 5 | 0.3642 | 0.00105 | 19 s |
| 10 | 0.0855 | 0.00134 | 30 s |
| 20 | 0.0335 | 0.00149 | 64 s |
| 30 | 0.0242 | 0.00156 | 97 s |
| 50 | 0.0184 | 0.00175 | 168 s |
| 80 | 0.0157 | 0.00182 | 269 s |

NDCG is still creeping up at 80 epochs while training loss has nearly bottomed
out — the gain is small and the model is well into fitting the training data.
The grid runs at 30 epochs to **rank** configurations, since relative ordering is
what selection needs, and the selected configuration is then trained longer.

## If the personalised model loses

Then that is the result, and it gets reported
([ADR-007](../adr/ADR-007-baselines-before-advanced-models.md), §36 of the phase
brief). Before accepting it, the checklist is: verify the evaluation
implementation, the negative sampling, the seen-item masking, the id mapping,
loss convergence, score orientation, and the validation protocol.

What is **not** permitted is adjusting the protocol until the number improves.
A sampled-negative protocol, a warm-only headline, or a smaller catalogue would
all raise the number and all measure something other than what was asked.
