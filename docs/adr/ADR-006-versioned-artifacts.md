# ADR-006: Versioned model and index compatibility

## Status

Accepted — 2026-08-24.

## Context

A retrieval model is not one file. It is a *set* of artifacts that are only
meaningful together:

- the trained model,
- the item embedding matrix it produced,
- the vector index built over that matrix,
- the `IdMapping` that gives every row its identity.

Every failure mode from mixing versions of these is **silent**. Load an
embedding matrix with a mapping built from a different item set, and every dense
index resolves to the wrong item. Retrieval returns plausible-looking results.
No exception is raised, no metric is obviously wrong, and the only symptom is
that recommendations are subtly nonsensical. The same is true of an index built
by a different procedure, and of a checkpoint saved on one device and loaded on
another with different numerics.

An unversioned `model.pt` on disk cannot answer: what data trained this, with
what config, on what device, at what commit — and what else must it be paired
with?

## Decision

**Every artifact carries a metadata manifest, and incompatible combinations fail
at load time rather than at inference time.**

Fourteen mandated fields in `ArtifactMetadata`:

| Group | Fields |
|---|---|
| Identity | `model_name`, `model_version`, `model_type` |
| Provenance | `created_at`, `training_data_version`, `feature_version`, `configuration_hash`, `random_seed`, `framework_version`, `python_version`, `git_commit` |
| Quality | `metrics` |
| Compatibility | `supported_device`, `required_index_version` |

Enforcement:

1. **An artifact without a manifest does not exist.** The registry will not load
   it, `/v1/models` will not list it, `/ready` will not count it.
2. **Retrieval-participating types must declare `required_index_version`.**
   `retrieval_model`, `embedding`, and `index` fail validation without it — the
   pairing cannot be forgotten.
3. **`ArtifactRegistry.require_compatible` raises `ArtifactCompatibilityError`**
   on a device or index-version mismatch, at load, with both values named.
4. **`IdMapping` files carry a content fingerprint**, verified on load. A mapping
   edited after it was written fails to load rather than silently mis-resolving.
   Artifacts record the fingerprints they were built against.
5. **`build_metadata` fills the environment fields itself** — python version,
   framework versions, git commit — so they cannot be forgotten or faked.
   Absent frameworks are *omitted*, never recorded as "not installed".
6. **Registration is atomic and non-clobbering.** Manifests are written
   temp-then-`replace`, so no reader sees a half-written file; re-registering an
   existing version requires `overwrite=True`.
7. **`configuration_hash` covers training-relevant config only** — seed, device,
   data, models, evaluation. Changing an API port does not invalidate a model,
   while changing a split fraction does. Secrets are excluded: `SecretStr`
   renders as a mask, so two deployments differing only in password hash alike.

## Alternatives considered

**MLflow Model Registry as the Phase 1 system of record.** Rich UI, stage
transitions, experiment linkage. Rejected as the *primary*: it needs a server or
a tracking directory, it is not readable with `cat`, and it does not natively
express "this model requires index version 3". MLflow is added in Phase 2 as an
*additional* sink for experiment tracking, not a replacement.

**Filename conventions** (`lightgcn_v3_idx2.pt`). Zero infrastructure. Rejected:
unparseable in practice, silently truncated, and cannot carry metrics or a config
hash.

**A database table only.** Queryable and shared. Rejected as the sole store: it
requires PostgreSQL running for any training work, including offline
experimentation. The table exists in `schema.sql` for when multiple processes
write; the filesystem stays authoritative single-machine.

**Trusting semantic versioning by hand.** Rejected: it is exactly the discipline
that lapses under time pressure, and the failure is silent.

## Consequences

**Positive.** Mismatches fail loudly, at startup, with both values named. Every
artifact is traceable to a commit, a config, a dataset, and a seed. `/v1/models`
and `/ready` are honest because they read real manifests. Manifests are small
JSON and are committed, so history is auditable in git.

**Negative.** Training code must supply metadata — friction, and deliberately so.
A rebuilt index requires re-registering the models that depend on it, which is
correct but tedious. Version bookkeeping is manual (`model_version` is chosen by
the training run), so a careless overwrite is still possible — which is why
`overwrite=True` must be explicit.
