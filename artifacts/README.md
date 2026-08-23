# Artifacts

Outputs of training runs. Binary payloads are git-ignored; **metadata JSON is
not**, because it is small and is the auditable record of what was built.

| Directory | Holds |
|---|---|
| `mappings/` | `IdMapping` JSON files — string ids ↔ dense indices, with integrity fingerprints |
| `models/` | Trained model payloads (checkpoints, boosters) |
| `embeddings/` | Exported user/item embedding matrices |
| `indexes/` | Built vector indexes (FAISS) |
| `metadata/` | One `ArtifactMetadata` manifest per artifact version: `<model_name>/<model_version>.json` |

## The rule

**An artifact without a manifest does not exist.** `ArtifactRegistry` will not
load one, `/v1/models` will not list one, and `/ready` will not count one toward
readiness. The manifest records the config hash, seed, data version, feature
version, framework versions, git commit, supported device, and required index
version — everything needed to answer "can I trust this, and can I reproduce
it?". See [ADR-006](../docs/adr/ADR-006-versioned-artifacts.md).

## Phase 1 status

Empty. No model has been trained, so `/ready` correctly returns **503**.
