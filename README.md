# OmniRank

A production-oriented, multi-stage, multimodal recommendation system.

> **Status: Phase 2 (data engineering) complete.**
> The repository contains the Phase 1 foundation plus a complete, reproducible,
> leakage-checked data pipeline that has been run end to end on the **full real
> PixelRec50K dataset** (989,494 interactions, 50,000 users, 82,865 items).
> **No recommendation model has been trained or implemented yet.** Endpoints that
> would need one return HTTP 501 naming the phase that will deliver them, and no
> benchmark number appears anywhere in this repository.

---

## Dataset

**[PixelRec50K](docs/data/pixelrec50k_overview.md)** — a 50,000-user sample of
[PixelRec](https://github.com/westlake-repl/PixelRec), a short-video
recommendation dataset.

Chosen because it is genuinely multimodal (cover image + title + category +
description per item, which the Phase 4 two-tower retriever needs), has **real
Unix timestamps** spanning 2012–2022 rather than only an implied ordering, fits
a laptop at 51 MB of CSV, and is honest about what it measures — one implicit
engagement signal, not a synthetic multi-event taxonomy.

> **Licence.** Provided by the Westlake Representation Learning Lab for
> **non-commercial research and education only**, with no rights to copy,
> modify, publish, distribute, or commercialise. `data/` is git-ignored and no
> PixelRec data — raw or processed — is ever committed. Test fixtures are
> generated, never sampled from the download.

```bash
make download-data    # 51 MB from the official Google Drive folder
make prepare-data     # full pipeline, ~15 s on an M-series Mac
```

## Objective

Build a recommendation system whose architecture is reusable across verticals.
The initial demonstration is e-commerce, but nothing in `src/` names a vertical:
the domain lives entirely in a configuration profile (`configs/data/*.yaml`), so
the same pipeline serves **products, courses, jobs, books, movies, music, news,
and videos** by swapping one file and writing one loader.

## Target architecture

```text
User and item data
        │
        ▼
Candidate generation
├── Popularity fallback              (Phase 2)
├── Matrix factorization baseline    (Phase 2)
├── LightGCN collaborative retrieval (Phase 3)
├── Multimodal two-tower retrieval   (Phase 4)
└── SASRec sequential retrieval      (Phase 3)
        │
        ▼
Candidate aggregation and deduplication   (Phase 3)
        │
        ▼
Ranking-feature generation                (Phase 5)
        │
        ▼
LightGBM ranker                           (Phase 5)
        │
        ▼
MMR diversity-aware reranking             (Phase 5)
        │
        ▼
Final Top-K recommendations
```

Every stage above is an **interface that exists today** (`src/omnirank/models/base.py`,
`retrieval/base.py`, `ranking/base.py`, `reranking/base.py`). What does not exist
yet is any concrete implementation behind them.

Detailed diagrams: [`docs/architecture/system_architecture.md`](docs/architecture/system_architecture.md),
[`offline_training_flow.md`](docs/architecture/offline_training_flow.md),
[`online_serving_flow.md`](docs/architecture/online_serving_flow.md).

## Implementation status

| Area | Status | Where |
|---|---|---|
| Configuration (YAML + env overrides + validation) | ✅ Implemented | `src/omnirank/core/config.py` |
| Structured logging, redaction, run correlation | ✅ Implemented | `src/omnirank/core/logging.py` |
| Exception hierarchy | ✅ Implemented | `src/omnirank/core/exceptions.py` |
| Device resolution (CPU / MPS, never assumes CUDA) | ✅ Implemented | `src/omnirank/core/device.py` |
| Data contracts (User / Item / Interaction) | ✅ Implemented | `src/omnirank/data/schemas.py` |
| Batch validation (12 rule families) | ✅ Implemented | `src/omnirank/data/validation.py` |
| ID mapping (append-only, fingerprinted) | ✅ Implemented | `src/omnirank/data/id_mapping.py` |
| **PixelRec50K loaders (chunked, schema-asserting)** | ✅ Implemented | `src/omnirank/data/pixelrec/` |
| **Cleaning + rejected-record trail** | ✅ Implemented | `src/omnirank/data/cleaning.py` |
| **Iterative k-core filtering** | ✅ Implemented | `src/omnirank/data/filtering.py` |
| **Per-user leave-last-N splitting** | ✅ Implemented | `src/omnirank/data/splitters.py` |
| **Leakage validation (13 checks, build-failing)** | ✅ Implemented | `src/omnirank/data/leakage.py` |
| **Sequential / graph / collaborative datasets** | ✅ Implemented | `src/omnirank/data/{sequences,pipeline}.py` |
| **Training-only user & item statistics** | ✅ Implemented | `src/omnirank/data/statistics.py` |
| **Evaluation slices (12)** | ✅ Implemented | `src/omnirank/data/slices.py` |
| **Multimodal feature alignment (streaming)** | ✅ Implemented | `src/omnirank/data/pixelrec/features.py` |
| **Dataset manifest + checksums** | ✅ Implemented | `src/omnirank/data/manifest.py` |
| Artifact metadata + registry | ✅ Implemented | `src/omnirank/artifacts/` |
| PostgreSQL schema (DDL, partitioning, retention) | ✅ Implemented | `src/omnirank/database/schema.sql` |
| API: `/health`, `/ready`, `/v1/models` | ✅ Implemented | `src/omnirank/api/routes/` |
| API: all other endpoints | 📋 Contract only → **501** | `src/omnirank/api/schemas/` |
| Metrics emission seam (logging sink) | ✅ Implemented | `src/omnirank/monitoring/` |
| Candidate generators, ranker, reranker | 📋 Interface only | `src/omnirank/models/base.py` |
| Vector index (FAISS) | 📋 Interface only | `src/omnirank/retrieval/base.py` |
| Database and cache clients | 📋 Protocol only | `src/omnirank/{database,cache}/` |
| Prometheus / Grafana, Kubernetes, streaming | ❌ Deferred | — |

✅ implemented and tested · 📋 contract defined, no implementation · ❌ not started

## Planned recommendation models

None of these are implemented. They are listed with the phase that delivers them
and the baseline each must beat ([ADR-007](docs/adr/ADR-007-baselines-before-advanced-models.md)).

| Model | Kind | Phase | Must beat |
|---|---|---|---|
| Time-decayed popularity | non-personalised | 2 | — (it is the floor) |
| Implicit matrix factorization | collaborative | 2 | popularity |
| LightGCN | graph collaborative | 3 | matrix factorization |
| SASRec | sequential | 3 | matrix factorization |
| Two-tower multimodal | content + collaborative | 4 | LightGCN on cold items |
| LightGBM LambdaRank | learning-to-rank | 5 | best single retriever |
| MMR | diversity reranking | 5 | ranker, on diversity at equal NDCG |

## Repository structure

```text
omnirank/
├── configs/            # base.yaml + data/models/evaluation/serving overlays
├── data/               # raw, interim, processed, external (git-ignored)
├── artifacts/          # mappings, models, embeddings, indexes, metadata
├── src/omnirank/
│   ├── api/            # FastAPI app, routes, schemas, dependencies
│   ├── core/           # config, logging, exceptions, device  (imports nothing else)
│   ├── data/           # contracts + the full PixelRec50K pipeline
│   │   ├── pixelrec/   #   source adapter: loaders, canonical mapping, features
│   │   └── ...         #   cleaning, filtering, mapping, splitters, leakage,
│   │                   #   sequences, statistics, slices, profiling, manifest, pipeline
│   ├── features/       # feature store + sequence builder contracts
│   ├── models/         # CandidateGenerator / Ranker interfaces + reserved model packages
│   ├── retrieval/      # aggregation + vector index contracts
│   ├── ranking/        # ranking-feature contract
│   ├── reranking/      # post-ranking filters + reranker contracts
│   ├── evaluation/     # evaluator + ground-truth contracts
│   ├── artifacts/      # metadata contract + filesystem registry
│   ├── database/       # repository protocols + schema.sql
│   ├── cache/          # cache backend protocol + key builder
│   └── monitoring/     # metrics sink protocol + logging sink
├── scripts/            # prepare_data, train, evaluate, serve
├── tests/              # unit, integration, fixtures
├── docs/               # architecture, adr, api, data, phase_reports
└── reports/            # figures, metrics, phase reports
```

## Environment

Developed and tested on **macOS / Apple Silicon (arm64), Python 3.11, CPU + MPS,
no CUDA**. Nothing in the codebase assumes a GPU: `core/device.py` resolves
`auto` to MPS when available and CPU otherwise, and *never* selects CUDA
implicitly.

Phase 1 has **no heavy dependencies**. PyTorch, FAISS, LightGBM,
SentenceTransformers, MLflow, and DVC are declared in the `ml` extra and are not
installed by default, which is why the test suite runs in seconds and offline.

## Installation

```bash
# Recommended: uv (fetches Python 3.11 automatically)
uv venv --python 3.11
uv pip install -e ".[dev]"

# Or with pip, from an existing Python 3.11
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Configuration
cp .env.example .env      # then edit; .env is git-ignored
```

Or simply:

```bash
make setup
```

## Running

```bash
# Start the API (reads configs/base.yaml, honours .env)
make serve
# equivalently: .venv/bin/python scripts/serve.py

# Then:
curl localhost:8000/health          # 200 - process is alive
curl -i localhost:8000/ready        # 503 - no artifacts registered yet (expected)
curl localhost:8000/v1/models       # 200 - {"models": [], "count": 0, ...}
open http://localhost:8000/docs     # full OpenAPI contract for every endpoint
```

`/ready` returning **503 on a fresh checkout is correct**: no model has been
trained, so the service cannot serve recommendations and says so.

Optional backing services (not required by Phase 1):

```bash
make up      # PostgreSQL + Redis via docker compose; applies schema.sql on first start
make down
```

## Data pipeline

```bash
make download-data                  # 51 MB; --with-features adds 17.3 GB of vectors
make prepare-data                   # full pipeline
make validate-data                  # check source files only
make profile-data                   # raw profiling reports only
```

Or directly:

```bash
python scripts/prepare_data.py --config configs/data/pixelrec50k.yaml [--overwrite] \
    [--validate-only] [--profile-only] [--subset-users N]
```

### Stages

```text
inspect → validate → profile raw → canonicalize → clean → filter →
map ids → split → graph/sequences/statistics/slices →
LEAKAGE CHECKS → write outputs → reports → manifest
```

A critical leakage check **aborts the build** with a non-zero exit code.

### Split strategy

**Per-user leave-last-N** ordered by the source's real Unix timestamps: each
eligible user's last interaction is the test target, the second-to-last is the
validation target, everything earlier is training history. Users with fewer than
3 interactions contribute training history and appear in no evaluation set —
they are not discarded. See [`docs/data/temporal_splitting.md`](docs/data/temporal_splitting.md).

### Leakage controls

13 checks run on every build: no interaction in two splits; train precedes
validation precedes test per user; sequence histories strictly past and never
containing their own target; graph edges training-only; item popularity and user
statistics verified against an independent training-only recount; one mapping
resolving every split; no split/target column in any feature table. Cold-start
items are reported as a **warning**, not a failure.

Every check has a test that injects the specific leak it is meant to catch.
See [`docs/data/leakage_prevention.md`](docs/data/leakage_prevention.md).

### Verified result on the full dataset

| | Raw | Processed |
|---|---:|---:|
| Users | 50,000 | **50,000** |
| Items | 82,865 | **69,347** |
| Interactions | 989,494 | **975,976** |

Splits: 875,976 train / 50,000 validation / 50,000 test · sparsity 0.99972 ·
filtering converged in 1 iteration removing 13,518 singleton items ·
**leakage: 13 checks, 0 critical failures, 1 expected warning** (770 cold-start
items) · runtime ~15 s.

Multimodal feature coverage is **0.0** — the 17.3 GB of published vectors is not
downloaded by default, and the pipeline reports their absence rather than
assuming it.

### Output structure

```text
data/interim/pixelrec50k/     canonical_{users,items,interactions}, rejected_records
data/processed/pixelrec50k/
├── {train,validation,test}_interactions.parquet
├── collaborative/  graph/  sequential/  metadata/  features/  evaluation_slices/
├── split_metadata.json
└── dataset_manifest.json
artifacts/mappings/pixelrec50k/   user/item id mappings + metadata
reports/data_quality/pixelrec50k/ raw/ processed/ leakage/ filtering/
```

Full schemas: [`docs/data/processed_schemas.md`](docs/data/processed_schemas.md).

## Testing

```bash
make test          # pytest
make lint          # ruff check
make format        # ruff format
make typecheck     # mypy --strict
make check         # lint + typecheck + test
```

Tests require no GPU, no network, no database, and no downloaded model weights.

## Phase roadmap

| Phase | Scope | Status |
|---|---|---|
| **1** | Foundation: structure, config, contracts, artifact registry, API skeleton, docs, tests | ✅ **Complete** |
| **2** | Data pipeline on PixelRec50K: loaders, cleaning, filtering, ordered splitting, leakage validation, collaborative/graph/sequential datasets, feature alignment, slices, manifest | ✅ **Complete** |
| **3** | Baselines + offline evaluation metrics: popularity, matrix factorization, recall/NDCG/MAP, per-slice reporting | Next |
| **4** | Collaborative and sequential retrieval: LightGCN, SASRec, FAISS index, candidate aggregation | Planned |
| **5** | Multimodal: PixelRec's published 1024-d vectors, two-tower retrieval | Planned |
| **6** | Ranking and serving: feature builder, LightGBM ranker, MMR reranking, serving pipeline, fallback chain, hot reload | Planned |
| **7** | Operations: Prometheus + Grafana, online evaluation, A/B framework | Planned |
| **8+** | Explicitly out of scope for now: Kubernetes, real-time streaming, online learning, reinforcement learning | Deferred |

Phase reports: [Phase 1](docs/phase_reports/phase_01_report.md) · [Phase 2](docs/phase_reports/phase_02_report.md).

## Architectural decisions

| ADR | Decision |
|---|---|
| [001](docs/adr/ADR-001-modular-monolith.md) | Modular monolith before microservices |
| [002](docs/adr/ADR-002-temporal-splitting.md) | Temporal splitting for interaction data |
| [003](docs/adr/ADR-003-offline-embeddings.md) | Offline precomputation of text and image embeddings |
| [004](docs/adr/ADR-004-faiss-initial-index.md) | FAISS as the initial vector index |
| [005](docs/adr/ADR-005-postgres-redis-responsibilities.md) | PostgreSQL and Redis responsibilities |
| [006](docs/adr/ADR-006-versioned-artifacts.md) | Versioned model and index compatibility |
| [007](docs/adr/ADR-007-baselines-before-advanced-models.md) | Baselines before advanced models |
| [008](docs/adr/ADR-008-lightgbm-over-catboost.md) | LightGBM over CatBoost for ranking |

## License

MIT.
