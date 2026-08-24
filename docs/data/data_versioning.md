# Data versioning and reproducibility

## The goal

Another developer, given this repository and the dataset manifest, can determine
whether their rebuild is **the same dataset**. That requires recording what went
in, how it was processed, and what came out — all three, with checksums.

## The manifest

`data/processed/pixelrec50k/dataset_manifest.json`, written on every successful run.

### Inputs

| Field | Purpose |
|---|---|
| `dataset_name`, `dataset_version` | Which snapshot |
| `source_repository`, `licence` | Provenance and terms |
| `source_files` | Per-file path, bytes, rows, SHA-256 |
| `source_checksums`, `source_file_sizes` | Flattened for quick comparison |

Verified checksums for the current snapshot:

```text
interaction.csv  638b53ec100f760cb9bd540c361f6d6e3617c81b1c054ced63fffa41da909e4d
item_info.csv    a073c2c65900f215a8137929b27dc57cf6f4f8fa11453a5c74fa8ff3a730a04e
```

### Process

| Field | Purpose |
|---|---|
| `pipeline_version` | Bumped when the pipeline changes its outputs for identical inputs |
| `schema_version` | Bumped when processed table schemas change |
| `git_commit` | Code version, or `null` outside a checkout |
| `python_version` | |
| `configuration_hash` | SHA-256 of the training-relevant config sections |
| `random_seed` | |
| `mapping_version`, `split_version` | |
| `split_strategy`, `ordering_field` | |
| `filtering_configuration` | |
| `subset_users` | **Non-null flags a development run** |
| `processing_timestamp` | |

### Outputs

Row counts per entity and split, feature dimensions and coverage, and
`output_files` / `output_checksums` — every written file with its bytes, rows,
and SHA-256. The current run records **34 outputs**.

### `known_limitations`

A **required** field, not a courtesy. A dataset whose gaps are undocumented gets
used as though it had none. It currently records: the single implicit event
type, the absence of user metadata, the absent item fields, why engagement
counters are excluded from features, and the 0.0 multimodal coverage.

## Determinism

Byte-identical outputs across runs are what make checksums meaningful. Achieved by:

- **Fixed column order** — `write_parquet` takes an explicit column list and
  treats a missing column as an error.
- **Fixed row order** — every write sorts by a declared key.
- **No index column**, fixed zstd compression.
- **Sorted id assignment** — mappings sort external ids before assigning dense
  indices, so source row order cannot change them.
- **No RNG in any stage.** The seed is recorded for completeness; nothing in
  Phase 2 consumes randomness.
- **Injectable clock** — validators take `now=`, so tests never read the wall clock.

An integration test runs the pipeline twice and asserts every Parquet checksum
matches. Only the manifest differs, because it embeds a timestamp.

## ID mappings

`artifacts/mappings/pixelrec50k/`:

| File | Contents |
|---|---|
| `user_id_mapping.parquet` | `external_user_id` → `internal_user_id` |
| `item_id_mapping.parquet` | `external_item_id` → `internal_item_id` |
| `mapping_metadata.json` | Versions, counts, checksums, fingerprints, policies |

**Properties.** Internal ids start at 0 and are contiguous, so they index an
embedding matrix directly. Assignment follows sorted external ids, so the same
population always yields the same mapping. Reverse lookups are exact.

**Unknown-entity policy**, recorded in the metadata: an unmapped id resolves to
`-1`. Phase 2 produces none — the mapping is fitted on the full post-filtering
population. At serving time an unmapped user is a cold user routed to the
fallback chain, never to an embedding lookup.

**Reassignment policy.** Internal ids are stable for a given `dataset_version`.
Changing the version rebuilds the mapping and invalidates every embedding
trained against the previous one. The fingerprint recorded in artifact metadata
is what makes that mismatch detectable rather than silent
([ADR-006](../adr/ADR-006-versioned-artifacts.md)).

Each mapping also exposes the Phase 1 `IdMapping` fingerprint, so a Phase 3
model can record it and the registry can refuse an incompatible pairing.

## What is not committed

`data/` and the binary contents of `artifacts/` are git-ignored. The dataset
licence prohibits redistribution, and processed outputs are derivatives.

## DVC

**Not configured.** Phase 1 did not set DVC up, and the Phase 2 brief specifies
integrating with it *if it was already configured*. Adding it now would mean
choosing a remote, and there is no second machine or collaborator to share with yet.

The manifest's source and output checksums provide the property DVC would be
adopted for — knowing whether two copies of the data are the same. The adoption
trigger is explicit: **when a second machine or person needs the processed
dataset**, DVC (or an object-store equivalent) is introduced, with the manifest
checksums as the migration check.

## Rebuilding

```bash
python scripts/download_pixelrec50k.py
python scripts/prepare_data.py --config configs/data/pixelrec50k.yaml --overwrite
```

Compare `output_checksums` between the old and new manifests. Any difference is
explained by a change in `source_checksums`, `configuration_hash`,
`pipeline_version`, or `git_commit` — and if none of those changed, the pipeline
has a non-determinism bug worth finding.
