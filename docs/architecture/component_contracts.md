# Component contracts

Twenty components, each with one responsibility and an explicit interface.
Nothing calls into another component's internals; everything crosses these
boundaries.

**Status key:** ✅ implemented and tested · 📋 contract defined, no implementation

| # | Component | Module | Interface | Status |
|---|---|---|---|---|
| 1 | Data ingestion | `data.loaders` | `DatasetLoader` | 📋 P2 |
| 2 | Schema validation | `data.validation` | `validate_batch` | ✅ |
| 3 | Preprocessing | `data.preprocessing` | `Preprocessor` | 📋 P2 |
| 4 | ID mapping | `data.id_mapping` | `IdMapping` | ✅ |
| 5 | Temporal splitting | `data.splitting` | `Splitter`, `check_split_integrity` | 📋 P2 / ✅ checker |
| 6 | Feature generation | `features.base` | `FeatureStore` | 📋 P2 |
| 7 | Sequence generation | `features.base` | `SequenceBuilder` | 📋 P2 |
| 8 | Model training | `models.baselines`, `models.lightgcn`, `models.sasrec`, `models.two_tower` | `CandidateGenerator`, `Ranker` | ✅ P5 (popularity, BPR, LightGCN, SASRec, two-tower) |
| 9 | Model evaluation | `evaluation.*` | `Evaluator`, `GroundTruth`, `OfflineEvaluator` | ✅ P3 |
| 10 | Candidate generation | `models.baselines`, `models.two_tower` | `CandidateGenerator` | ✅ P5 (five sources) |
| 11 | Candidate aggregation | `retrieval.aggregation` | `CandidateAggregator` | ✅ P4 (RRF, five-source in P5) |
| 12 | Ranking | `models.base`, `ranking.base` | `Ranker`, `FeatureBuilder` | 📋 P6 |
| 13 | Post-ranking | `reranking.base` | `PostRankingFilter`, `Reranker` | 📋 P6 |
| 14 | Artifact management | `artifacts` | `ArtifactMetadata`, `ArtifactRegistry` | ✅ |
| 15 | Vector-index management | `retrieval.faiss_index`, `retrieval.two_tower_index` | `VectorIndex` | ✅ P5 (exact, verified tie-aware) |
| 16 | Recommendation serving | `api` | FastAPI app | ✅ skeleton |
| 17 | Interaction ingestion | `api.routes.interactions`, `database.base` | `InteractionRepository` | 📋 P2 |
| 18 | Caching | `cache.base` | `CacheBackend`, `CacheKey` | 📋 P2 |
| 19 | Database access | `database.base` | `*Repository` protocols | 📋 P2 |
| 20 | Monitoring | `monitoring.base` | `MetricsSink` | ✅ logging sink |

---

## Cross-cutting: configuration

**Module:** `omnirank.core.config` — ✅ implemented

Resolution order, later winning:

1. `configs/base.yaml`
2. overlays named in its `include:` list, in order
3. overlays passed to `load_config(overlays=[...])`
4. `.env` in the project root
5. real environment variables

Environment keys use a double-underscore path:
`OMNIRANK__DATA__VALIDATION__MIN_PRICE=1` → `data.validation.min_price`.
Values are coerced with YAML scalar rules, so `true` becomes a bool and `42` an
int rather than strings that silently fail a type check later.

**Every model forbids unknown keys.** A typo in YAML or in an env var is a
startup failure naming the offending path — not a setting that is silently
ignored while you wonder why your change did nothing.

**Secrets never appear in YAML.** `SecretStr` fields have no YAML default, mask
themselves in every dump and repr, and are excluded from config hashes.

**Startup validation catches what is legal-but-wrong**, including: split
fractions leaving no training window, a positive-event threshold above every
declared weight, aggregation weights naming an undeclared generator, a fallback
chain not ending in `global_popularity`, and — in `staging`/`production` — a
missing DB password, `api.reload` left on, or console log format.

## Cross-cutting: logging and errors

**Modules:** `omnirank.core.logging`, `omnirank.core.exceptions` — ✅ implemented

Structured logging via structlog, with the stdlib root logger routed through the
same processor chain so uvicorn's and SQLAlchemy's records render identically to
ours.

- **Correlation.** `run_context()` / `bound_context()` bind a run or request id
  into every event emitted inside the block, via context variables — nothing has
  to thread a logger through call signatures.
- **Redaction.** A processor masks any key matching the configured sensitive-key
  list, recursively, at any depth, inside dicts *and* lists, case-insensitively.
  It is a backstop, not a licence.
- **No `print()` in library code.** Enforced by ruff (`T20`) and by a test that
  walks the AST of everything under `src/`.

Every deliberate failure derives from `OmniRankError` and carries a stable
`code` used verbatim as the API `error.code` and as a log key, so alerting keys
on identifiers rather than prose.

---

## Interface reference

### `CandidateGenerator` (components 8, 10)

```python
class CandidateGenerator(ABC):
    name: str

    def fit(self, data) -> None: ...
    def recommend(self, user_id: str, k: int, context: dict | None = None) -> list[Candidate]: ...
    def score(self, user_id: str, item_ids: list[str]) -> list[float]: ...
    def save(self, path: str | Path) -> None: ...
    @classmethod
    def load(cls, path: str | Path) -> Self: ...
```

Obligations:

- `recommend` **may return fewer than `k`.** A cold user yields nothing; the
  fallback chain handles that, not the generator.
- `score` returns **one score per input item, in input order**, and scores
  unknown items `0.0` rather than raising — the ranker needs a value for every
  candidate, including ones this generator did not nominate.
- `context` keys a generator does not understand must be **ignored, not
  rejected**, so adding a request-time signal does not break every generator.
- `load` returns an immediately usable instance: `is_fitted` is true.
- Scores are **generator-local** and not comparable across generators.

### `Ranker` (component 12)

```python
class Ranker(ABC):
    def fit(self, features, labels, groups=None) -> None: ...
    def rank(
        self, candidates: list[Candidate], context: dict | None = None
    ) -> list[RankedItem]: ...
    def save(self, path) -> None: ...
    @classmethod
    def load(cls, path) -> Self: ...
```

`groups` carries query-group sizes, required by pairwise/listwise objectives
such as LambdaRank. `rank` must be **order-stable for equal scores** — otherwise
a cached response and a freshly computed one disagree.

### `Evaluator` (component 9)

```python
class Evaluator(ABC):
    def evaluate(self, recommendations, ground_truth, k_values: list[int]) -> dict[str, float]: ...
```

Returns flat `"<metric>@<k>"` keys, stable across runs so two reports diff
mechanically. Denominator rules are in
[`offline_training_flow.md`](offline_training_flow.md#7-offline-evaluation).

### `CandidateAggregator` (component 11)

```python
class CandidateAggregator(ABC):
    def aggregate(
        self, per_source: dict[str, Sequence[Candidate]], *, limit: int
    ) -> AggregationResult: ...
```

Must normalise within source before cross-source comparison, preserve every
contributing source on merged candidates, and be deterministic.

### `VectorIndex` (component 15)

```python
class VectorIndex(Protocol):
    dimension: int
    index_version: int

    def build(self, embeddings, *, metric="inner_product") -> None: ...
    def search(self, query, k) -> tuple[list[list[int]], list[list[float]]]: ...
    def save(self, path) -> None: ...
    @classmethod
    def load(cls, path) -> Self: ...
```

Row order is the dense item index from `IdMapping`. The index deliberately does
**not** know about string ids, so it cannot drift out of sync with the mapping in
a way that silently resolves. `search` pads with `-1` so output is rectangular.

### `ArtifactMetadata` (component 14)

Fourteen mandated fields, all present and validated:

`model_name` · `model_version` · `model_type` · `created_at` ·
`training_data_version` · `feature_version` · `configuration_hash` ·
`random_seed` · `framework_version` · `python_version` · `metrics` ·
`supported_device` · `required_index_version` · `git_commit`

Plus `format_version`, `artifact_path`, `id_mapping_fingerprints`, `notes`.

Enforced: `created_at` must be timezone-aware; metrics must not be NaN;
retrieval-participating types (`retrieval_model`, `embedding`, `index`) **must**
declare `required_index_version`.

### Repository protocols (component 19)

`UserRepository` · `ItemRepository` · `InteractionRepository`.

`ItemRepository.get_many` is batched because hydrating a 200-item candidate list
one row at a time would dominate the latency budget.
`InteractionRepository.append_many` must be idempotent, relying on the
`uq_interactions_event` business-key index so a retried delivery does not
double-count.

`get` returning `None` for an unknown entity is **not** an exception —
anonymous and brand-new users are the common case.

### `CacheBackend` (component 18)

`get` returns `None` on miss **or on backend failure** — indistinguishable by
design, because the caller must proceed either way. `set` takes a mandatory
`ttl_seconds`; there is no unbounded write.

### `MetricsSink` (component 20)

`increment` / `observe` / `gauge`. The only Phase 1 implementation is
`LoggingMetricsSink`, which emits structured log events — so the numbers are not
lost while Prometheus is deferred, and swapping in a Prometheus client later
touches one wiring line rather than every call site. Canonical metric names are
fixed now, because renaming a metric after dashboards exist is expensive.
