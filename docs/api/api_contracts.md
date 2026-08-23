# API contracts

Base URL: `http://localhost:8000` · Interactive docs: `/docs` · Machine-readable: `/openapi.json`

Every schema below is **complete and validated today**. Whether the logic behind
it exists is a separate question, marked per endpoint.

| Endpoint | Method | Status |
|---|---|---|
| `/health` | GET | ✅ Implemented |
| `/ready` | GET | ✅ Implemented |
| `/v1/models` | GET | ✅ Implemented |
| `/v1/recommendations/users/{user_id}` | GET | 📋 501 → Phase 5 |
| `/v1/recommendations/similar/{item_id}` | GET | 📋 501 → Phase 4 |
| `/v1/recommendations/session` | POST | 📋 501 → Phase 5 |
| `/v1/interactions` | POST | 📋 501 → Phase 2 |
| `/v1/items/{item_id}` | GET | 📋 501 → Phase 2 |
| `/v1/admin/reload-artifacts` | POST | 📋 501 → Phase 5 |

## Why 501 and not stub data

A stubbed recommendation is indistinguishable from a real one to a caller. It
gets screenshotted into status updates, it makes integration tests pass against a
system that does not work, and it is discovered late. A 501 naming the
delivering phase cannot be mistaken for anything.

The contract is still fully usable: schemas are in `/openapi.json`, so client
code and type definitions can be generated today.

---

## Implemented

### `GET /health` — liveness

Answers from in-process state only. No dependency checks, so a Redis outage
cannot cause a liveness probe to restart a container that is working fine.

```json
{
  "status": "ok",
  "service": "omnirank",
  "version": "0.1.0",
  "environment": "local",
  "uptime_seconds": 12.34,
  "timestamp": "2026-08-24T10:00:00Z"
}
```

Always `200`.

### `GET /ready` — readiness

Checks whether this process can serve recommendation traffic: device resolved,
at least one **compatible** artifact registered, configuration valid.

`200` when ready, `503` when not, with a per-dependency breakdown.

```json
{
  "ready": false,
  "service": "omnirank",
  "version": "0.1.0",
  "environment": "local",
  "device": "cpu",
  "dependencies": [
    {"name": "device", "ready": true, "detail": "resolved to cpu", "required": true},
    {"name": "artifact_registry", "ready": false,
     "detail": "no artifacts registered; train and register a model before serving (Phase 1 ships no trained models)",
     "required": true},
    {"name": "configuration", "ready": true, "detail": "loaded for environment 'local'", "required": true}
  ],
  "timestamp": "2026-08-24T10:00:00Z"
}
```

**503 on a fresh checkout is correct.** PostgreSQL and Redis are absent from this
list in Phase 1 — no client is wired, and reporting an unchecked dependency as
ready would be a false claim.

### `GET /v1/models` — registry listing

Reads the real filesystem registry. Each entry is flagged with whether it can be
loaded on this host's device and index version.

```json
{
  "models": [
    {
      "model_name": "popularity",
      "model_version": "v1",
      "model_type": "ranker",
      "created_at": "2026-06-01T12:00:00Z",
      "training_data_version": "ecommerce-subset@v0",
      "feature_version": "f1",
      "supported_device": "any",
      "required_index_version": null,
      "metrics": {"recall@20": 0.11},
      "compatible": true,
      "incompatibility_reason": null
    }
  ],
  "count": 1,
  "device": "cpu",
  "serving_ready": true
}
```

`metrics` contains only what the manifest recorded. It is never populated with
placeholder numbers.

---

## Declared contracts (501)

### `GET /v1/recommendations/users/{user_id}` → Phase 5

| Parameter | Type | Default | |
|---|---|---|---|
| `k` | int 1–200 | 20 | |
| `category` | str | — | restrict to one category |
| `exclude_seen` | bool | true | drop already-interacted items |

Response — **the canonical payload of the whole system**:

```json
{
  "user_id": "user_123",
  "model_version": "v1",
  "recommendations": [
    {
      "item_id": "item_456",
      "rank": 1,
      "score": 0.87,
      "sources": ["lightgcn", "sasrec"],
      "reason": "Recommended from your recent activity"
    }
  ],
  "fallback_used": false,
  "fallback_stage": null,
  "latency_ms": 42,
  "request_id": "a1b2c3d4e5f6",
  "generated_at": "2026-08-24T10:00:00Z"
}
```

Three decisions are baked in deliberately:

- **`sources` is a list.** An item can be nominated by several generators;
  knowing which ones is what makes a result debuggable after the fact.
  Collapsing to one string loses that permanently.
- **`fallback_used` is mandatory, not optional.** A degraded response that looks
  identical to a healthy one is how quality regressions go unnoticed for weeks.
- **`reason` is nullable.** Emitted only when the pipeline can actually justify
  the item from its sources and features. There is no default string, because a
  fabricated explanation is worse than none.

### `GET /v1/recommendations/similar/{item_id}` → Phase 4

`k` (1–200, default 20), `space` ∈ `content` | `collaborative` | `hybrid`.
`content` works for cold items; `collaborative` does not. Same response shape.

### `POST /v1/recommendations/session` → Phase 5

For anonymous and cold-start users — driven by the item sequence in the body
rather than by stored history. This is the request shape SASRec is built to serve.

```json
{
  "session_id": "sess_abc",
  "item_ids": ["item_1", "item_2"],
  "user_id": null,
  "k": 20,
  "context": {"locale": "en-GB"}
}
```

`item_ids` is **oldest first** — the ordering is meaningful, max 200.

### `POST /v1/interactions` → Phase 2

```json
{
  "events": [
    {"user_id": "u1", "item_id": "i1", "event_type": "click",
     "timestamp": "2026-08-24T10:00:00Z", "session_id": "s1",
     "interaction_id": "client-key-1", "request_id": "a1b2c3d4e5f6"}
  ]
}
```

1–500 events. `timestamp` defaults to server receipt time; `interaction_id` is a
client idempotency key — re-sending the same one is a no-op, which is what makes
at-least-once delivery from a client safe. `request_id` links the event to the
recommendation that preceded it, enabling attribution.

Response (`202`), with **partial success as the norm**:

```json
{"accepted": 48, "rejected": 2, "duplicates": 3,
 "rejections": [{"index": 12, "rule": "unknown_event_type", "message": "..."}],
 "request_id": "..."}
```

A batch with two bad rows out of fifty is accepted for the forty-eight; the
client fixes and resends only what failed. Rule identifiers match
[`../data/data_contracts.md`](../data/data_contracts.md#validation-rules).

### `GET /v1/items/{item_id}` → Phase 2

Hydrates an item id into displayable fields. A **projection** of the internal
`Item`: the free-form `attributes` map is not dumped wholesale, because that is
where vertical-specific and occasionally sensitive data ends up.

### `POST /v1/admin/reload-artifacts` → Phase 5

```json
{"model_names": null, "force": false, "dry_run": false}
```

Two properties specified up front because retrofitting them is painful:

- **Atomic swap.** The new set is fully loaded and compatibility-checked *before*
  it replaces the live one; a failure leaves the previous set serving.
- **Reported, not assumed.** The response names exactly which artifacts changed,
  so a deploy pipeline can assert on it rather than trusting a 200.

Writing this contract before the pipeline exists is what keeps a handler from
capturing a model object at import time — which would make hot-swapping
impossible.

**Requires authentication before it is enabled.** It is the one endpoint that
changes server behaviour. Not implemented in Phase 1, and nothing is exposed.

---

## Errors

Every non-2xx response shares one shape, so a client needs one error-parsing
path rather than one per failure mode:

```json
{"error": {"code": "not_implemented_yet",
           "message": "...",
           "context": {"feature": "...", "planned_phase": 5},
           "request_id": "a1b2c3d4e5f6"}}
```

| `code` | HTTP | Meaning |
|---|---|---|
| `not_implemented_yet` | 501 | Defined contract, no implementation. `context.planned_phase` names the phase. |
| `artifact_not_found` | 404 | Unknown model name/version |
| `schema_validation_error` | 422 | Payload violated a data contract |
| `service_not_ready` | 503 | Required dependency unavailable |
| `artifact_compatibility_error` | 503 | Device or index-version mismatch (ADR-006) |
| `configuration_error` | 500 | Invalid configuration |
| `internal_error` | 500 | Unexpected. Nothing disclosed beyond `request_id`. |

Codes are stable across releases: alerting keys on them rather than on prose.

**Validation precedes the 501.** `POST /v1/interactions` with an empty `events`
array returns **422**, not 501 — the contract is enforced before the
unimplemented body is reached.

## Request correlation

Every response carries `X-Request-ID` and `X-Response-Time-ms`. A caller-supplied
`X-Request-ID` is honoured (truncated to 64 chars) so a trace can span services;
otherwise one is generated. The same value is bound into every log line emitted
while handling the request and appears in error bodies.

## Versioning

`/v1` is in the path. Additive changes (a new optional field) ship within `v1`;
anything that could break a client gets `/v2`. `model_version` in the response
body is orthogonal — it identifies the serving pipeline, not the API shape.
