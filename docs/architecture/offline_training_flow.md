# Offline training flow

```text
Raw dataset
    │
    ▼
Schema validation
    │
    ▼
Data cleaning and ID mapping
    │
    ▼
Temporal train-validation-test split
    │
    ▼
Feature and sequence generation
    │
    ▼
Model training
    │
    ▼
Offline evaluation
    │
    ▼
Model and embedding export
    │
    ▼
Artifact registration
```

**Phase 1 implements stages 1–3 partially and stage 9 fully.** Validation and ID
mapping are real, tested code; artifact registration is real, tested code. The
stages between them are contracts. The table below marks each.

---

## 1. Raw dataset → canonical records

| | |
|---|---|
| **Component** | 1 — Data ingestion |
| **Contract** | `omnirank.data.loaders.DatasetLoader` |
| **Status** | ✅ Implemented (Phase 2) |

A loader turns one source format into `DatasetBundle(users, items, interactions,
provenance)`. It does not clean, filter, deduplicate, or reject — that is the
next stage's decision, made against the domain profile.

`provenance` (source paths, row counts, export date) flows into the artifact
manifest, so a model can always be traced back to the files it came from.

## 2. Schema validation

| | |
|---|---|
| **Component** | 2 — Schema validation |
| **Contract** | `omnirank.data.validation.validate_batch` |
| **Status** | ✅ **Implemented and tested** |

Record-level rules come from the Pydantic contracts; batch-level rules
(duplicates, dangling references, clock skew) come from the batch validator.
Output is a `ValidationReport` per entity, carrying counts by rule.

**Non-throwing by default.** Real catalogues contain bad rows; failing the job on
the first one is useless. The caller inspects `rejection_rate` and decides.
`strict=True` restores fail-fast for CI fixtures.

Full rule list: [`../data/data_contracts.md`](../data/data_contracts.md).

## 3. Cleaning and ID mapping

| | |
|---|---|
| **Components** | 3 — Preprocessing · 4 — ID mapping |
| **Contracts** | `omnirank.data.preprocessing.Preprocessor` · `omnirank.data.id_mapping.IdMapping` |
| **Status** | 📋 Preprocessor contract only (Phase 2) · ✅ **IdMapping implemented and tested** |

k-core filtering must be **iterative**: removing users with fewer than *n*
interactions can push items below their threshold, and vice versa. A single pass
leaves the invariant unsatisfied.

Mappings are built from post-filtering survivors only, so no dense index is
allocated to an entity the models will never see. They are **append-only** and
**fingerprinted** — an embedding matrix trained against version *n* stays valid
against *n + k* for every id it already knew, and a tampered mapping file fails
to load rather than silently resolving ids to the wrong entities.

## 4. Temporal split

| | |
|---|---|
| **Component** | 5 — Temporal splitting |
| **Contract** | `omnirank.data.splitting.Splitter` · `check_split_integrity` |
| **Status** | 📋 Splitter contract only (Phase 2) · ✅ **Integrity checker implemented** |

Strictly by time. Random splitting lets a model observe event *t+1* while
predicting event *t* for the same user, which inflates every offline metric and
produces a model that is worse online than offline. See
[ADR-002](../adr/ADR-002-temporal-splitting.md).

`check_split_integrity` is runnable today and asserts: no interaction in two
splits, every event inside its window, and the configured embargo respected at
each boundary. Phase 2's splitter will be tested against it.

## 5. Feature and sequence generation

| | |
|---|---|
| **Components** | 6 — Feature generation · 7 — Sequence generation |
| **Contracts** | `omnirank.features.FeatureStore` · `SequenceBuilder` |
| **Status** | 📋 Contracts only (Phase 2) |

**The rule that matters: a feature may only use information that existed at the
timestamp it is attached to.** Computing "user's lifetime purchase count" over
the whole log and attaching it to a row from six months ago leaks the future.
`as_of` is threaded through every signature in these contracts so that
respecting it is the default and violating it requires deliberately ignoring a
parameter.

Sequence builders receive **training-window interactions only**. Passing the full
log is precisely the leak the contract exists to prevent, and it is documented on
the method.

## 6. Model training

| | |
|---|---|
| **Component** | 8 — Model training |
| **Contract** | `omnirank.models.base.CandidateGenerator` · `Ranker` |
| **Status** | ✅ Popularity + BPR (Phase 3); LightGCN + SASRec + aggregation + FAISS (Phase 4) |

Training is model-specific; the *contract* is not. Every generator implements
`fit` / `recommend` / `score` / `save` / `load`, which is what lets the aggregator
treat five very different models as a list.

Order of delivery is fixed by [ADR-007](../adr/ADR-007-baselines-before-advanced-models.md):
popularity, then matrix factorization, then the neural models — each measured
against its predecessor.

## 7. Offline evaluation

| | |
|---|---|
| **Component** | 9 — Model evaluation |
| **Contract** | `omnirank.evaluation.Evaluator` · `GroundTruth` |
| **Status** | ✅ Implemented (Phase 2) |

An evaluator receives already-made recommendations and never calls a model, so
the same evaluator scores a popularity baseline and a full pipeline, and no
metric can accidentally re-rank what it is measuring.

Two denominator rules are written into the contract because both are common ways
an offline number ends up flattering a model:

- Users with **no held-out items are excluded** at `GroundTruth` construction —
  including them dilutes recall unboundedly.
- Users who **received no recommendations score zero**, they are not dropped.
  Dropping them lets a model that cannot serve half its traffic look excellent.

`evaluation.protocol: full` ranks against the whole catalogue. `sampled` is
faster and biased, and is for development loops only — never for a reported
number.

## 8. Export

| | |
|---|---|
| **Component** | 14 — Artifact management |
| **Status** | 📋 Per-model exporters (Phase 2+) |

Models export a payload plus, for embedding-based retrievers, an embedding
matrix and the index built over it. Both record `required_index_version`.

## 9. Artifact registration

| | |
|---|---|
| **Component** | 14 — Artifact management |
| **Contract** | `omnirank.artifacts.ArtifactRegistry` · `build_metadata` |
| **Status** | ✅ **Implemented and tested** |

`build_metadata` fills in the environment fields (python version, framework
versions, git commit) itself, so they cannot be forgotten or faked. Training code
supplies only what it knows: data version, feature version, config hash, seed,
metrics, device, index version.

Writes are atomic (temp file then `replace`), so a reader never observes a
half-written manifest. Re-registering an existing version requires
`overwrite=True` — silently rewriting a version another process may have loaded
is how "the metrics changed but the model didn't" bugs happen.

---

## Reproducibility

A training run is reproducible from three recorded values:

| Value | Recorded in | Covers |
|---|---|---|
| `configuration_hash` | manifest | seed, device, data, models, evaluation config |
| `training_data_version` | manifest | which dataset snapshot |
| `git_commit` | manifest | which code |

`AppConfig.training_config_hash` deliberately excludes serving-only sections, so
changing an API port does not invalidate a trained model — while
`AppConfig.config_hash` covers everything. Both exclude secrets: `SecretStr`
renders as a mask, so two deployments differing only in password hash identically.
