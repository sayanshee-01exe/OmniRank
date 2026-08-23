# Phase 1 report — foundation

**Date:** 2026-08-24 · **Version:** 0.1.0 · **Status:** complete

---

## 1. Repository state before this phase

`/Users/shee_sayan/Recommendation_system` contained **zero files at any depth**,
was **not a git repository**, and had no hidden files. Greenfield.

The standard assessment targets — existing functionality, incomplete modules,
placeholder code, duplicate logic, hard-coded paths, missing abstractions, broken
imports, architectural inconsistencies, data leakage, training-serving skew,
unnecessary dependencies — **all resolve to N/A**: there was nothing to preserve,
duplicate, or break.

The second working directory (`smart_loan_approval_system_claude/sprints`) is an
unrelated project. It was read for house conventions only and **was not
modified**.

### Toolchain probe

| Item | Found | Action |
|---|---|---|
| Arch / OS | arm64, Darwin 25.5 | Confirms Apple Silicon, no CUDA |
| RAM / disk | 16 GB / 100 GB free | Subset-scale datasets only |
| Python | 3.14.7 — **3.11 absent** | Provisioned via `uv python 3.11` (3.11.15) |
| uv, docker, git, mypy, pytest | present | Used |
| poetry, ruff | absent | ruff installed into the project venv; uv used instead of poetry |

## 2. Architecture decisions

Eight ADRs, each with Status / Context / Decision / Alternatives / Consequences.

| ADR | Decision | Core reason |
|---|---|---|
| [001](../adr/ADR-001-modular-monolith.md) | Modular monolith before microservices | Distributed cost, no independent-scaling benefit on one machine. Layering enforced by an AST test, not convention. |
| [002](../adr/ADR-002-temporal-splitting.md) | Temporal splitting | Random splits leak the future; the inflation is invisible in the offline numbers. |
| [003](../adr/ADR-003-offline-embeddings.md) | Offline text/image embeddings | Encoding at request time cannot fit a 300 ms budget, and puts an untracked model version in the serving path. |
| [004](../adr/ADR-004-faiss-initial-index.md) | FAISS first, behind `VectorIndex` | In-memory at this scale; exact search removes recall-vs-model ambiguity while models are developed. |
| [005](../adr/ADR-005-postgres-redis-responsibilities.md) | PostgreSQL truth, Redis regenerable | "Flush Redis and lose nothing but latency" is the test that keeps the split honest. |
| [006](../adr/ADR-006-versioned-artifacts.md) | Versioned model/index compatibility | Every mismatch failure mode here is *silent*; enforcement moves it to load time. |
| [007](../adr/ADR-007-baselines-before-advanced-models.md) | Baselines before advanced models | An unbaselined neural result is unfalsifiable. Popularity is also the fallback floor. |
| [008](../adr/ADR-008-lightgbm-over-catboost.md) | LightGBM for ranking | Smaller, faster on dense numeric features, clean arm64 build. CatBoost's edge doesn't apply here. |

## 3. Files created

**124 files** (59 Python modules under `src/`, 14 test files, 16 docs, 5 config files;
5 628 lines of source, 2 201 of tests, 2 250 of documentation).
**No files were modified** — the repository was empty — and none deleted.

<details>
<summary>Root and infrastructure (9)</summary>

`README.md` · `pyproject.toml` · `Makefile` · `.env.example` · `.gitignore` ·
`.python-version` · `docker-compose.yml` · `.github/workflows/ci.yml` ·
`data/README.md`
</details>

<details>
<summary>Configuration (5)</summary>

`configs/base.yaml` · `configs/data/ecommerce.yaml` ·
`configs/models/retrieval.yaml` · `configs/evaluation/default.yaml` ·
`configs/serving/local.yaml`
</details>

<details>
<summary>Source — 34 modules under <code>src/omnirank/</code></summary>

- **core** (5): `config.py`, `logging.py`, `exceptions.py`, `device.py`, `__init__.py`
- **data** (7): `schemas.py`, `validation.py`, `id_mapping.py`, `loaders.py`, `preprocessing.py`, `splitting.py`, `__init__.py`
- **features** (2), **models** (6, incl. 4 reserved packages), **retrieval** (2), **ranking** (2), **reranking** (2), **evaluation** (2)
- **artifacts** (3): `metadata.py`, `registry.py`, `__init__.py`
- **database** (3): `base.py`, `schema.sql`, `__init__.py` · **cache** (2) · **monitoring** (2)
- **api** (21): `app.py`, `errors.py`, `middleware.py`, 7 route modules, 9 schema modules, 2 dependency modules
</details>

<details>
<summary>Scripts (4), tests (11), docs (16), placeholders</summary>

`scripts/{prepare_data,train,evaluate,serve}.py` ·
`tests/conftest.py` + 6 unit + 2 integration + 3 package inits ·
4 architecture docs, 2 data docs, 1 API doc, 8 ADRs, this report ·
`artifacts/README.md`, `notebooks/README.md`, `.gitkeep` files
</details>

## 4. Interfaces defined

| Interface | Module | Kind |
|---|---|---|
| `CandidateGenerator` | `models.base` | ABC |
| `Ranker` | `models.base` | ABC |
| `Evaluator` | `evaluation.base` | ABC |
| `CandidateAggregator` | `retrieval.base` | ABC |
| `VectorIndex` | `retrieval.base` | Protocol |
| `FeatureBuilder` | `ranking.base` | ABC |
| `FeatureStore`, `SequenceBuilder` | `features.base` | Protocol / ABC |
| `PostRankingFilter`, `Reranker` | `reranking.base` | ABC |
| `DatasetLoader`, `StreamingInteractionSource` | `data.loaders` | Protocol |
| `Preprocessor`, `Splitter` | `data.preprocessing`, `data.splitting` | Protocol |
| `UserRepository`, `ItemRepository`, `InteractionRepository` | `database.base` | Protocol |
| `CacheBackend` | `cache.base` | Protocol |
| `MetricsSink` | `monitoring.base` | Protocol |

Supporting types: `Candidate`, `RankedItem`, `AggregationResult`, `FeatureRow`,
`FeatureBatch`, `UserSequence`, `GroundTruth`, `DataSplit`, `SplitBoundaries`,
`DatasetBundle`, `PreprocessedDataset`, `ValidationReport`, `ValidatedBatch`.

## 5. Schemas defined

**Data contracts:** `User`, `Item`, `Interaction`, `EventType` (6 events).
**Artifact:** `ArtifactMetadata` — all 14 mandated fields plus 4 bookkeeping ones.
**Database:** 7 tables with PKs, FKs, unique constraints, 12 indexes, monthly
range partitioning on `interactions`, and two partition-management functions.
**API:** 20 Pydantic models across 8 modules, covering all 9 endpoints.

## 6. Tests added

**359 tests**, all offline, no GPU, no external services, no model downloads.

| File | Tests | Covers |
|---|---|---|
| `unit/test_config.py` | 36 | Loading, `include:` merging, env/dotenv overrides, 9 validation failures, secret masking, hash stability |
| `unit/test_data_schemas.py` | 34 | Every field rule on all three entities, timezone handling, dedup key |
| `unit/test_data_validation.py` | 28 | All 12 validation rules, report semantics, referential integrity, injectable clock |
| `unit/test_id_mapping.py` | 24 | Append-only guarantee, fingerprint collision resistance, tamper detection |
| `unit/test_artifacts.py` | 32 | Metadata contract, compatibility matrix, registry CRUD, corrupt-manifest handling |
| `unit/test_interfaces.py` | 31 | ABC enforcement, fitted-state guard, candidate merging, ranker determinism |
| `unit/test_logging.py` | 25 | Configuration, correlation, recursive redaction |
| `unit/test_device.py` | 12 | `auto` never picks CUDA; behaviour without torch installed |
| `integration/test_api.py` | 48 | Health, readiness, model listing, all six 501 contracts, correlation, OpenAPI |
| `integration/test_repository_smoke.py` | 89 | Import integrity (58 modules), layering, no `print()`, file layout, ADR structure, script exit codes |

Notable properties the suite pins down:

- **No module requires the `ml` extra** — asserted by checking `sys.modules` after importing all 58.
- **Layering is enforced by AST walk**, not convention.
- **No `print()` under `src/`**, by AST walk.
- **Every ADR has all five required sections.**
- **No script emits a metric-shaped string** — the fabricated-benchmark tripwire.

## 7. Commands executed

```bash
uv venv --python 3.11              # fetched cpython-3.11.15-macos-aarch64
uv pip install -e ".[dev]"
.venv/bin/python -m ruff format src tests scripts
.venv/bin/python -m ruff check --fix src tests scripts
.venv/bin/python -m mypy
.venv/bin/python -m pytest
```

## 8. Results

| Gate | Result |
|---|---|
| `ruff format --check` | **77 files already formatted** |
| `ruff check` | **All checks passed** |
| `mypy` (strict) | **Success: no issues found in 77 source files** |
| `pytest` | **359 passed** in ~1.8 s |

One deprecation warning remains, from `fastapi.testclient` importing `httpx` —
third-party, not from our code.

### Failures found and fixed during the phase

Three real defects, all in code written this phase:

1. **`UserSequence.__post_init__`** used `zip(ts, ts[1:], strict=True)`, which
   raises on every valid input because the sequences differ in length by one.
   Replaced with `itertools.pairwise`. Caught by `test_valid_sequence`.
2. **Logging correlation was untestable through `structlog.testing.capture_logs`**,
   which replaces the processor chain — precisely where contextvar merging and
   redaction live. Rewritten to assert on real rendered JSON output.
3. **A capsys/stream-binding bug in the test harness**: `configure_logging` binds
   a handler to `sys.stderr`, and pytest swaps that between the setup and call
   phases, so a handler bound in a fixture wrote to a buffer the test never read.
   Fixed by configuring inside the call phase.

## 9. Known limitations

1. **No model exists.** `/ready` returns 503 on a fresh checkout. This is correct,
   not a defect.
2. **No data pipeline.** `prepare_data.py --check-only` validates configuration
   and reports resolved paths; without the flag it exits 3.
3. **No database or cache client.** Protocols and DDL exist; nothing connects.
   PostgreSQL and Redis are deliberately absent from `/ready`, since reporting an
   unchecked dependency as ready would be a false claim.
4. **No evaluation metrics.** The `Evaluator` contract is defined; no metric is
   computed, and none is invented.
5. **`docker-compose` up is untested end-to-end.** The DDL is syntactically
   reviewed but has not been applied to a live PostgreSQL in this phase.
6. **No authentication anywhere**, including `/v1/admin/reload-artifacts` — which
   is why that endpoint is unimplemented rather than merely unauthenticated.
7. **No git history.** `git init` has not been run, so `detect_git_commit`
   returns `None` and artifacts would record no commit until a repository exists.
8. **MLflow and DVC are declared but not wired.** Phase 2.
9. **`faiss-cpu` on Apple Silicon** has historically had patchy wheels. Untested
   here because Phase 1 does not install the `ml` extra — a Phase 2 risk.

## 10. Deferred work

**By phase:** data pipeline, baselines, metrics, DB/cache clients (2) · LightGCN,
SASRec, FAISS index, aggregation (3) · multimodal embeddings, two-tower (4) ·
feature builder, LightGBM ranker, MMR, serving pipeline, hot reload (5) ·
Prometheus/Grafana, online evaluation, A/B (6).

**Explicitly out of scope, not merely later:** Kubernetes, multiple deployable
services, real-time streaming, online learning, reinforcement learning.

## 11. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Contracts prove wrong under a real implementation | Rework in Phase 2 | Every interface was exercised by a minimal in-test implementation, so none is merely aspirational |
| 16 GB RAM insufficient for LightGCN/SASRec at scale | Phase 3 blocked | Subset datasets first; `configs/data/*` makes scale a config change; cloud GPU for large runs |
| `faiss-cpu` arm64 build problems | Phase 3 delay | `VectorIndex` protocol allows swapping to pgvector or hnswlib without touching callers |
| Temporal splitting yields metrics below published baselines | Looks like underperformance | ADR-002 documents why; every reported number must state its protocol |
| Feature skew between offline and online | Silent quality loss | `FeatureBuilder` is the single sanctioned source; `feature_version` in metadata makes a mismatch detectable |
| Config surface grows unwieldy | Onboarding friction | `extra="forbid"` everywhere means dead keys fail loudly rather than accumulating |
| Reserved model packages read as "started" | Overstated progress | Each `__init__.py` says **NOT IMPLEMENTED** with its phase, and exports nothing |

## 12. How to run

```bash
# Install
uv venv --python 3.11 && uv pip install -e ".[dev]"    # or: make setup

# Run the API
.venv/bin/python scripts/serve.py                       # or: make serve
curl localhost:8000/health      # 200
curl -i localhost:8000/ready    # 503 — no artifacts registered (expected)
curl localhost:8000/v1/models   # 200 — {"models": [], "count": 0, ...}
open http://localhost:8000/docs

# Run the tests
.venv/bin/python -m pytest                              # or: make test
make check                                              # lint + typecheck + test
```

## 13. Acceptance criteria

| Criterion | Status |
|---|---|
| Clear modular structure | ✅ 14 subpackages, layering enforced by test |
| Configuration centralised and validated | ✅ YAML + env + `.env`, `extra="forbid"`, 9 cross-field validators |
| Data contracts documented | ✅ `docs/data/data_contracts.md`, 12 validation rules |
| Model interfaces documented and testable | ✅ 13 interfaces, each exercised by a test implementation |
| Artifact metadata versioned | ✅ 14 mandated fields, compatibility enforced at load |
| API contracts defined | ✅ 9 endpoints, 20 Pydantic models, full OpenAPI |
| Health checks work | ✅ `/health` 200, `/ready` honest 503 |
| Tests pass locally | ✅ 359 passed |
| README accurate | ✅ Per-area status table, ✅/📋/❌ throughout |
| Architectural decisions documented | ✅ 8 ADRs |
| No advanced model falsely presented | ✅ Endpoints 501; reserved packages say NOT IMPLEMENTED |
| Project installs and imports | ✅ 58 modules, verified by test |
| Phase 2 needs no restructuring | ✅ Every Phase 2 deliverable has a contract and a home |

## 14. Recommended Phase 2 scope

Ordered so each step unblocks the next, and so the fallback chain gets its floor
first.

1. **Dataset selection and a CSV/parquet `DatasetLoader`** for a manageable
   e-commerce subset. Wire `prepare_data.py` end to end.
2. **`Preprocessor`** — iterative k-core filtering, id-mapping construction,
   mapping registration as artifacts.
3. **`TemporalSplitter`**, tested against the existing `check_split_integrity`.
4. **Evaluation metrics** — recall, precision, NDCG, MAP, MRR, hit-rate, plus the
   four beyond-accuracy metrics. Validate against hand-computed fixtures *before*
   scoring any model.
5. **Popularity baseline** — time-decayed, global and per-category. This is also
   the terminal fallback stage, so it lands before anything else that serves.
6. **Matrix factorization baseline** (implicit ALS/BPR), measured against
   popularity under the same protocol.
7. **PostgreSQL and Redis clients** implementing the existing protocols; add both
   to `/ready`; implement `POST /v1/interactions` and `GET /v1/items/{item_id}`.
8. **Alembic**, introduced at the first `schema.sql` change after data exists.
9. **MLflow and DVC** wired for experiment tracking and data versioning.

**Phase 2 exit criterion:** a registered popularity artifact and a registered MF
artifact, each with real recorded metrics from the same evaluation protocol, and
`/ready` returning 200.
