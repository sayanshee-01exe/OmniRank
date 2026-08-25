# System architecture

## 1. What OmniRank is

A multi-stage recommendation system: a cheap retrieval stage proposes a few
hundred candidates from a catalogue of any size, an expensive ranking stage
orders them, and a reranking stage trades a little relevance for diversity. The
whole thing is one deployable process ([ADR-001](../adr/ADR-001-modular-monolith.md)).

The staged design is not architectural taste — it is arithmetic. Scoring a
million items per request with a gradient-boosted model over fifty features is
not possible inside a 300 ms budget. Scoring two hundred is comfortable. So
retrieval optimises **recall** cheaply, and ranking optimises **precision**
expensively over what retrieval survived.

## 2. Domain independence

OmniRank is not an e-commerce system that could be adapted. Nothing under
`src/omnirank/` names a vertical. A domain enters through exactly two places:

1. **A configuration profile** — `configs/data/<domain>.yaml`, declaring the
   event vocabulary, their implicit-feedback weights, validation bounds, and
   split parameters.
2. **A loader** — one implementation of `DatasetLoader` that turns that domain's
   source format into `User` / `Item` / `Interaction` records.

Everything downstream is identical for products, courses, jobs, books, movies,
music, news, and videos. The generic `attributes` mapping on `User` and `Item`
carries vertical-specific fields without schema changes.

The design cost of this is real and worth naming: the core schema is *thin*.
A jobs vertical wanting salary-band matching, or news wanting recency decay,
implements it as a feature over `attributes`, not as a first-class column.

## 3. Component map

```text
                     ┌───────────────────────────────────────────┐
                     │             omnirank.core                 │
                     │  config · logging · exceptions · device   │
                     │  (imports nothing else in the project)    │
                     └───────────────────────────────────────────┘
                                       ▲
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        │              │               │               │              │
   ┌────┴────┐   ┌─────┴─────┐   ┌─────┴─────┐   ┌─────┴─────┐  ┌─────┴─────┐
   │  data   │   │ features  │   │  models   │   │ artifacts │  │ database  │
   │ schemas │   │  store    │   │  base     │   │ metadata  │  │  cache    │
   │validate │   │ sequences │   │ baselines │   │ registry  │  │monitoring │
   │id_map   │   │           │   │ lightgcn  │   │           │  │           │
   │split    │   │           │   │ sasrec    │   │           │  │           │
   └────┬────┘   └─────┬─────┘   │ two_tower │   └─────┬─────┘  └─────┬─────┘
        │              │         └─────┬─────┘         │              │
        └──────────────┴───────────────┼───────────────┴──────────────┘
                                       │
                     ┌─────────────────┴──────────────────┐
                     │  retrieval → ranking → reranking   │
                     └─────────────────┬──────────────────┘
                                       │
                              ┌────────┴────────┐
                              │       api       │
                              │ routes/schemas  │
                              └─────────────────┘
```

**The layering rule** is enforced by a test
(`tests/integration/test_repository_smoke.py::TestLayering`): `core` imports
nothing from any other subpackage, and `data` imports nothing from `models`,
`retrieval`, or `api`. Dependencies point one way. A cycle here would mean the
"modular" in modular monolith had quietly stopped being true.

## 4. Target pipeline

```text
User and item data
        │
        ▼
Candidate generation  ── each implements CandidateGenerator
├── Popularity fallback              Phase 3
├── Matrix factorization baseline    Phase 3
├── LightGCN collaborative retrieval Phase 4
├── SASRec sequential retrieval      Phase 4
└── Multimodal two-tower retrieval   Phase 5
        │
        ▼
Candidate aggregation and deduplication    Phase 4 — CandidateAggregator
        │
        ▼
Ranking-feature generation                 Phase 6 — FeatureBuilder
        │
        ▼
LightGBM ranker                            Phase 6 — Ranker
        │
        ▼
MMR diversity-aware reranking              Phase 6 — Reranker
        │
        ▼
Final Top-K recommendations
```

Every box is an interface that exists today. None has an implementation.

## 5. The three flows

| Flow | Document |
|---|---|
| Offline training | [`offline_training_flow.md`](offline_training_flow.md) |
| Online serving + fallback | [`online_serving_flow.md`](online_serving_flow.md) |
| Component responsibilities and interfaces | [`component_contracts.md`](component_contracts.md) |

## 6. Storage responsibilities

| Store | Holds | Loses on wipe |
|---|---|---|
| **PostgreSQL** | Users, items, the interaction event log, served-recommendation audit, feedback events, model registry | Everything. This is the only durable store. |
| **Redis** | Rendered responses, hot features, session histories | Nothing — all regenerable. Eviction under memory pressure is correct behaviour. |
| **Filesystem** | Model payloads, embeddings, indexes, id mappings, artifact manifests | Retrainable, but expensively. |

Rationale in [ADR-005](../adr/ADR-005-postgres-redis-responsibilities.md).

## 7. Compute

Development target is Apple Silicon: CPU and MPS, **no CUDA**.
`omnirank.core.device.resolve_device` is the only place a device is chosen;
`auto` resolves to MPS when available and CPU otherwise, and *never* selects
CUDA. An explicit CUDA request requires `device.allow_cuda`, so a config copied
back from a cloud GPU host fails loudly on a laptop instead of silently changing
numerics.

Artifacts record the device they were built on
(`ArtifactMetadata.supported_device`), and the registry refuses to load one that
does not match — see [ADR-006](../adr/ADR-006-versioned-artifacts.md).

## 8. What is deliberately absent

| Not present | Why |
|---|---|
| Microservices | One process, explicit module boundaries. Split when a component's scaling profile genuinely diverges, not before. [ADR-001](../adr/ADR-001-modular-monolith.md) |
| Kubernetes | Nothing to orchestrate. `docker compose` runs the two backing services. |
| Real-time streaming | Batch retraining is sufficient until measured staleness says otherwise. |
| Online / reinforcement learning | Requires a trustworthy online evaluation loop, which requires the recommendation audit log, which is defined but not yet written to. |
| A feature store product | `FeatureStore` is a protocol over parquet in Phase 2. Adopting a product before the feature set is known would fix decisions we cannot yet inform. |
