# OmniRank

A production-oriented, multi-stage, multimodal recommendation system.

> **Status: Phase 5 (multimodal two-tower retrieval and cold-start) complete.**
> Five candidate generators are trained, registered and evaluated on the **real
> PixelRec50K dataset** under a full-catalogue protocol: time-decayed
> popularity, BPR matrix factorization, LightGCN, SASRec, and a multimodal
> two-tower retriever that represents items from content and so can return
> items with **no training interactions at all**.
>
> The blend's count of cold-target users it cannot serve at any depth goes from
> 724 to **zero**. The two-tower's accuracy contribution is much smaller: a
> statistically significant fusion gain on both Recall@20 and NDCG@20, but one
> of at most 0.00034 in absolute terms. Standalone, it is significantly *below*
> LightGCN.
> See [`docs/phase_reports/phase_05_report.md`](docs/phase_reports/phase_05_report.md).
>
> **No ranker, reranker, or serving model exists.** Recommendation endpoints
> still return HTTP 501, and every reported number comes from a real run
> recorded under `reports/metrics/`.

---

## Dataset

**[PixelRec50K](docs/data/pixelrec50k_overview.md)** — a 50,000-user sample of
[PixelRec](https://github.com/westlake-repl/PixelRec), a short-video
recommendation dataset.

Chosen because it is genuinely multimodal (cover image + title + category +
description per item, which the Phase 5 two-tower retriever needs), has **real
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
├── Popularity fallback              ✅ Phase 3
├── Matrix factorization baseline    ✅ Phase 3
├── LightGCN collaborative retrieval ✅ Phase 4
├── SASRec sequential retrieval      ✅ Phase 4
└── Multimodal two-tower retrieval ✅ Phase 5
        │
        ▼
Candidate aggregation and deduplication   ✅ Phase 4
        │
        ▼
Ranking-feature generation                (Phase 6)
        │
        ▼
LightGBM ranker                           (Phase 6)
        │
        ▼
MMR diversity-aware reranking             (Phase 6)
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
| **Offline evaluation framework** | ✅ Implemented | `src/omnirank/evaluation/` |
| **Popularity baseline (global + time-decay)** | ✅ Implemented | `src/omnirank/models/baselines/popularity.py` |
| **BPR matrix factorization** | ✅ Implemented | `src/omnirank/models/baselines/bpr.py` |
| **Negative sampling** | ✅ Implemented | `src/omnirank/models/baselines/negative_sampling.py` |
| **LightGCN graph retrieval** | ✅ Implemented | `src/omnirank/models/lightgcn/` |
| **SASRec sequential retrieval** | ✅ Implemented | `src/omnirank/models/sasrec/` |
| **Candidate aggregation (3 strategies)** | ✅ Implemented | `src/omnirank/retrieval/aggregation.py` |
| **Vector index (FAISS)** | ✅ Implemented | `src/omnirank/retrieval/faiss_index.py` |
| **Rolling temporal validation** | ✅ Implemented | `src/omnirank/data/rolling.py` |
| **Multimodal two-tower retrieval** | ✅ Implemented | `src/omnirank/models/two_tower/` |
| **Multimodal feature store** | ✅ Implemented | `src/omnirank/features/multimodal_store.py` |
| **Cold-item retrieval and evaluation** | ✅ Implemented | `src/omnirank/models/two_tower/catalogue.py` |
| Ranker and reranker | 📋 Interface only | `src/omnirank/models/base.py` |
| Database and cache clients | 📋 Protocol only | `src/omnirank/{database,cache}/` |
| Prometheus / Grafana, Kubernetes, streaming | ❌ Deferred | — |

✅ implemented and tested · 📋 contract defined, no implementation · ❌ not started

## Planned recommendation models

Popularity and BPR (Phase 3), LightGCN and SASRec (Phase 4), and the multimodal two-tower
(Phase 5) are implemented. The rest are listed
with the phase that delivers them and the baseline each must beat ([ADR-007](docs/adr/ADR-007-baselines-before-advanced-models.md)).

| Model | Kind | Phase | Must beat |
|---|---|---|---|
| Time-decayed popularity | non-personalised | 3 | — (it is the floor) |
| BPR matrix factorization | collaborative | 3 | popularity |
| LightGCN | graph collaborative | 4 | matrix factorization |
| SASRec | sequential | 4 | matrix factorization |
| Blended retriever (RRF and friends) | fusion | 4 | its own best single source |
| Two-tower multimodal | content + collaborative | 5 | LightGCN on cold items — **not met, see below** |
| LightGBM LambdaRank | learning-to-rank | 6 | best single retriever |
| MMR | diversity reranking | 6 | ranker, on diversity at equal NDCG |

> **The two-tower did not meet its stated bar.** It was required to beat
> LightGCN on cold items. Its cold Recall@20 is 0.000441 against LightGCN's
> 0.001322 — it did not.
>
> What it does instead is *reach* cold items at all. LightGCN cannot return an
> item it never saw during fitting, so 724 cold-target users are unservable by
> it at any depth; the two-tower serves all of them. That is a different
> property from ranking the reachable ones better, and it is recorded here as
> what actually happened rather than as the bar being met. The full accounting,
> with bootstrap intervals, is in
> [the Phase 5 report](docs/phase_reports/phase_05_report.md).

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
│   ├── models/         # CandidateGenerator / Ranker interfaces, baselines, lightgcn, sasrec
│   ├── retrieval/      # aggregation strategies, blended retriever, FAISS index, diagnostics
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

## Baselines and offline evaluation

```bash
make install-baseline     # adds PyTorch (the only extra Phase 3 needs)
make compare-baselines    # selection -> lock -> final -> reports
```

Or step by step:

```bash
python scripts/train.py --model popularity \
    --data-config configs/data/pixelrec50k.yaml \
    --stage selection --version phase3-popularity-selection

python scripts/evaluate.py --model popularity \
    --version phase3-popularity-selection --split validation --protocol full

python scripts/compare_baselines.py --stage selection   # search on validation
python scripts/compare_baselines.py --stage final       # lock, refit, test once
```

### Protocol

**Full-catalogue** evaluation is the only protocol used for reported numbers:
every item the model can legitimately recommend is scored, seen items are
excluded, and the top *K* is taken. Sampled-negative evaluation exists for
development speed and is never used for a result.

Users who receive no recommendations **score zero** and stay in the denominator.

### Two views, always together

| View | Population | Answers |
|---|---|---|
| **strict** | every held-out user | end-to-end performance; a cold target is a miss |
| **warm** | users whose target is in the fit catalogue | collaborative ranking quality |

On PixelRec50K, **98.24%** of validation targets and **98.55%** of test targets
are reachable — the remainder are genuine new-item cold start and bound what any
purely collaborative model can achieve.

### Validation and test discipline

```text
selection   fit: train              targets: validation
final       fit: train+validation   targets: test        (read once)
```

Hyperparameters are chosen on validation, written to
`reports/metrics/phase_0{3,4}/selected_configuration.json`, and only then is test
touched — both `compare_baselines.py --stage final` and
`compare_retrievers.py --stage final` refuse to run without that file.

Phase 4 added a second discipline alongside it: **before comparing two models,
check whether each one finished.** A ranking between a converged model and one
still improving at its epoch budget is a statement about the budget, not the
models. See [`docs/models/model_selection.md`](docs/models/model_selection.md).

Details: [`docs/evaluation/`](docs/evaluation/), [`docs/models/`](docs/models/)
and [`docs/retrieval/`](docs/retrieval/).

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
| **1** | Architecture and foundation: structure, config, contracts, artifact registry, API skeleton, docs, tests | ✅ **Complete** |
| **2** | PixelRec50K data engineering: loaders, cleaning, filtering, ordered splitting, leakage validation, collaborative/graph/sequential datasets, feature alignment, slices, manifest | ✅ **Complete** |
| **3** | Offline evaluation, popularity and matrix-factorization baselines | ✅ **Complete** |
| **4** | LightGCN, SASRec, candidate aggregation, FAISS index | ✅ **Core complete**, with [documented limitations](docs/phase_reports/phase_04_report.md#limitations) |
| **5** | Multimodal two-tower retrieval and new-item cold start | 🔨 **Current** |
| **6** | Learning-to-rank, MMR and online serving | Planned |
| **7** | Monitoring and online experimentation | Planned |

Phase reports: [Phase 1](docs/phase_reports/phase_01_report.md) · [Phase 2](docs/phase_reports/phase_02_report.md) · [Phase 3](docs/phase_reports/phase_03_report.md).

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
| [009](docs/adr/ADR-009-faiss-torch-openmp-coexistence.md) | Accepting `KMP_DUPLICATE_LIB_OK` so FAISS and PyTorch share a process |

## License

MIT.
