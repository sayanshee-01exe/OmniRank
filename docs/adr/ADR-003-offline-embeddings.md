# ADR-003: Offline precomputation of text and image embeddings

## Status

Accepted — 2026-08-24.

## Context

The multimodal two-tower retriever needs text embeddings (SentenceTransformers)
and image embeddings (CLIP) for every item. These can be computed at request
time or precomputed offline.

The numbers decide it. A SentenceTransformer forward pass on CPU is roughly
10–50 ms per item; CLIP image encoding is more. The serving latency budget for
the *entire* pipeline is 300 ms. Embedding even a handful of candidates
synchronously consumes the whole budget, and embedding a few hundred is not
arithmetically possible.

There is a second reason, independent of latency: an embedding model loaded in
the serving process is a **model version that nobody is tracking**. Upgrading
the SentenceTransformer version silently changes every vector, and thus every
retrieval result, with no artifact, no metadata, and no way to attribute the
change.

## Decision

**All text and image embeddings are computed offline, exported as artifacts, and
loaded read-only at serving time.**

- Embedding jobs run in the offline pipeline and write to
  `artifacts/embeddings/`.
- Each export is registered with full `ArtifactMetadata`, including the encoder's
  version in `framework_version` and a `required_index_version`.
- Serving loads matrices and the FAISS index; it never runs an encoder.
- **A missing modality degrades, it does not fail.** `Item.description` and
  `Item.image_id` are both optional, because real catalogues are incomplete. The
  fusion layer must handle text-only, image-only, and neither.
- New items are embedded by the next batch job. Until then they are served by the
  collaborative and popularity paths.

## Alternatives considered

**Online embedding at request time.** Handles new items instantly. Rejected on
latency arithmetic alone; it also puts an untracked model version in the serving
path.

**Online embedding with a Redis cache.** Amortises cost for popular items.
Rejected: the *first* request for any item still pays full latency, and cold
items are exactly the ones content embeddings are supposed to help — so the
cache misses precisely when it matters most. It also keeps the encoder in the
serving process.

**A dedicated embedding microservice.** Isolates the encoder. Rejected for
Phase 1 as premature ([ADR-001](ADR-001-modular-monolith.md)); it also does not
solve the latency problem, only relocates it.

## Consequences

**Positive.** Serving stays inside its latency budget with no encoder loaded, and
memory stays low enough for a 16 GB laptop. Embeddings are versioned, comparable,
and attributable. Encoder upgrades become an explicit re-export with a new
artifact version, not an invisible behaviour change.

**Negative.** New items are not content-retrievable until the next embedding
batch — a real cold-start gap, bounded by batch frequency, and one that must be
stated when cold-start performance is measured. Storage grows linearly with the
catalogue. A re-export is required whenever an encoder or preprocessing step
changes, which is a deliberate cost: it makes the change visible.
