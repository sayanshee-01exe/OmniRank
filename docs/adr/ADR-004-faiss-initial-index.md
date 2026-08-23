# ADR-004: FAISS as the initial vector index

## Status

Accepted — 2026-08-24. Revisit when the catalogue exceeds what fits comfortably
in one process's memory, or when index updates must be incremental.

## Context

Embedding-based retrieval (two-tower, LightGCN, SASRec) needs approximate
nearest-neighbour search over item vectors. The realistic options are an
in-process library (FAISS, hnswlib, Annoy), a PostgreSQL extension (pgvector),
or a dedicated service (Qdrant, Milvus, Weaviate).

The constraint that dominates Phase 1–4: development happens on one Apple
Silicon laptop with 16 GB of RAM and a deliberately subset-sized dataset. At that
scale, an index of a few hundred thousand vectors is tens to hundreds of
megabytes — it fits in memory with room to spare, and a network round trip to a
separate service would cost more than the search itself.

## Decision

**FAISS (`faiss-cpu`), in-process, behind the `VectorIndex` protocol.**

- `omnirank.retrieval.base.VectorIndex` defines `build` / `search` / `save` /
  `load`. Nothing outside `retrieval` imports FAISS.
- Start with `IndexFlatIP` — exact search, no training step, no recall/latency
  tradeoff to tune. Move to `IVFFlat` or `HNSW` when measurements say exact
  search is too slow, not before.
- **Row order is the dense item index** from `IdMapping`. The index deliberately
  does not know about string ids, so it cannot drift out of sync with the mapping
  in a way that silently resolves to the wrong item.
- `models.index.index_version` is recorded in every artifact that participates in
  retrieval, and enforced at load time ([ADR-006](ADR-006-versioned-artifacts.md)).
- `search` pads with `-1` so output is rectangular and callers need no
  special-casing.

## Alternatives considered

**pgvector.** One less system, transactional consistency with the catalogue, and
the vectors live next to the items. Genuinely attractive, and the most likely
successor. Rejected for now: it forces PostgreSQL to be running for any retrieval
work, including local experimentation and CI, and its ANN indexes are slower than
FAISS at the scales where FAISS fits in memory. The `VectorIndex` protocol keeps
the door open — `backend: pgvector` is already a valid config value.

**Qdrant / Milvus / Weaviate.** Purpose-built, with filtering, incremental
updates, and horizontal scaling. Rejected as premature: another service to run,
another client dependency, and another failure mode on the request path, in
exchange for capabilities a single-machine subset-scale system does not need.

**hnswlib.** Excellent recall/latency, simpler than FAISS. Reasonable
alternative; FAISS chosen for breadth of index types, so moving from exact to
IVF/HNSW/PQ later does not change library.

**Brute-force NumPy.** Zero dependencies, and honestly fine at a few thousand
items. Rejected: it does not survive the first realistic catalogue, and switching
later means writing the `VectorIndex` implementation anyway.

## Consequences

**Positive.** No extra service; retrieval works offline and in CI. Exact search
removes an entire class of "is this a recall problem or a model problem?"
ambiguity while models are being developed. Indexes are files, so they version
and ship like any other artifact.

**Negative.** The index lives in process memory, bounding catalogue size by RAM.
FAISS has **no incremental update** for flat indexes — adding items means a
rebuild, which is acceptable at batch cadence and would not be under streaming.
`faiss-cpu` wheels on Apple Silicon have historically been patchy; it stays in
the optional `ml` extra so Phase 1 never depends on it. Multi-process serving
would duplicate the index in each worker.
