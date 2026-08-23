# Online serving flow

## Happy path

```text
Client request
    │
    ▼
FastAPI                      ── request id assigned, bound to logs
    │
    ▼
User and context lookup      ── PostgreSQL + Redis
    │
    ▼
Candidate generators         ── run concurrently, each with its own budget
    │
    ▼
Candidate aggregation        ── merge, dedupe, truncate to ranker budget
    │
    ▼
Ranking                      ── feature build + LightGBM
    │
    ▼
Post-ranking filters         ── availability, already-purchased, blocked
    │
    ▼
Redis cache                  ── write-through, TTL'd, model-version-keyed
    │
    ▼
Recommendation response
```

**Phase 1 implements the FastAPI layer and nothing below it.** `/health`,
`/ready`, and `/v1/models` answer from real state; every recommendation endpoint
returns **501** naming the phase that delivers it.

---

## Stage detail

### FastAPI ✅ implemented

`RequestContextMiddleware` assigns a request id (or honours a caller-supplied
`X-Request-ID`, truncated to 64 chars), binds it into the structlog context for
the duration of the request, echoes it in the response header, and includes it in
error bodies. That single value is what turns "a user reported a bad
recommendation" into a retrievable trace.

**What the access log deliberately omits:** query strings and request bodies.
Both carry user and item identifiers, and an access log is the easiest place to
accidentally build a permanent record of individual browsing behaviour. The
logged path is the *route template* (`/v1/items/{item_id}`), not the concrete
path — which also keeps log cardinality bounded.

### User and context lookup 📋 Phase 2

Session history from Redis, durable profile and recent interactions from
PostgreSQL. Bounded by `before` so an offline backfill can reproduce exactly what
serving would have seen at a given instant.

### Candidate generators 📋 Phase 2–4

Each enabled generator runs with its own `top_k` budget from
`models.candidate_generators`. A generator that fails or exceeds its budget is
recorded in `AggregationResult.degraded_sources` and the request continues —
one broken retriever must not take the response down.

A generator returning fewer than `k` is **normal**, not an error: a cold user
legitimately yields nothing. Handling that is the fallback chain's job.

### Candidate aggregation 📋 Phase 3

The hard part is not merging but **comparison**. Generator scores live on
incomparable scales — a dot product, a decayed count, a softmax probability.
Implementations must normalise within source before any cross-source comparison,
or blend by rank rather than by score.

An item nominated by several generators keeps *all* contributing sources
(`Candidate.merged_with`), because that list is both a ranking feature and the
raw material for the response's `reason`.

### Ranking 📋 Phase 5

`FeatureBuilder` is the **only** sanctioned source of ranking features, and the
same instance must be callable from a training job and a request handler.
Training/serving skew in a ranker is nearly invisible — offline metrics look
fine, production quality is quietly worse — and the usual cause is a feature
computed one way in a notebook and another way in a handler.
`feature_version` is recorded in artifact metadata so a mismatch is detectable.

### Post-ranking 📋 Phase 5

Two distinct concerns, deliberately separated:

- **`PostRankingFilter` — correctness.** Remove what must not be shown:
  unavailable items, already-purchased items, blocked categories. Never optional.
- **`Reranker` — quality.** Reorder what may be shown, trading relevance for
  diversity (MMR).

Filters run **first**. Reordering a list that still contains items about to be
dropped spends the diversity budget on items nobody will see.

### Caching 📋 Phase 2

Every cache write carries a TTL — there is no unbounded `set`. Cache keys embed
`model_version`, so a deploy self-invalidates without an explicit flush.

**A cache failure is never a request failure.** The backend catches its own
transport errors and degrades to a miss. The recommendation path must survive
Redis being down entirely.

---

## Graceful fallback

```text
Primary models unavailable
        │
        ▼
Category popularity
        │
        ▼
Global popularity
        │
        ▼
Safe non-empty response
```

The chain is configured in `configs/serving/local.yaml` under `fallback.chain`,
and **validated at startup**: an empty chain is rejected, and the chain must end
with `global_popularity` — the only stage that can answer for any user, with no
history, no session, and no models loaded.

### When each stage fires

| Trigger | Result |
|---|---|
| Every generator returned nothing | fall through to category popularity |
| A generator failed | continue with the rest; record in `degraded_sources` |
| Ranker unavailable | serve aggregated candidates in retrieval order |
| Latency budget exceeded mid-pipeline | cut to the current best list |
| Artifacts not loaded at all | straight to global popularity |

### The contract this enforces

**A 200 response is never empty.** If the chain is exhausted,
`FallbackExhaustedError` is raised and the request fails loudly with a 5xx —
which is alert-worthy, because it means even global popularity could not answer.

**A degraded response always says so.** `fallback_used` is a mandatory field, not
an optional one, and `fallback_stage` names which stage answered. A degraded
response that looks identical to a healthy one is how quality regressions go
unnoticed for weeks.

### Why popularity is the floor

`models/baselines/popularity.py` is the one model that must never be
unavailable. It has no user-side input, no embeddings, no index, and no
dependency beyond an item-count table — so it can answer when everything else
cannot. That is why it is the first thing Phase 2 builds, not the least
interesting.

---

## Readiness semantics ✅ implemented

| Endpoint | Question | Checks |
|---|---|---|
| `/health` | Is the process alive? | Nothing external. A Redis outage must not get a working container killed by a liveness probe. |
| `/ready` | Can it serve useful traffic? | Device resolved, ≥1 compatible artifact registered, config valid. |

On a fresh checkout `/ready` returns **503** because no model has been trained.
That is the correct answer. Reporting "ready" there would be the single most
misleading thing this API could do.

PostgreSQL and Redis are **absent** from the readiness list in Phase 1 — no
client is wired, and reporting an unchecked dependency as ready would be a false
claim. They join the list in Phase 2 alongside their clients.
