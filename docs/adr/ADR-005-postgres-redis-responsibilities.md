# ADR-005: PostgreSQL and Redis responsibilities

## Status

Accepted — 2026-08-24.

## Context

Two stores are in the stack, and the common failure is letting their
responsibilities blur — session state that only exists in Redis and is lost on
eviction, or per-request derived data written to PostgreSQL until the write path
becomes the bottleneck. Both are recoverable only by rework, so the split is
worth deciding once, explicitly.

## Decision

**PostgreSQL holds durable truth. Redis holds only what can be regenerated.**

| | PostgreSQL | Redis |
|---|---|---|
| Users, items | ✅ | ✗ |
| Interaction event log | ✅ | ✗ |
| Recommendation audit (requests, results) | ✅ | ✗ |
| Feedback events | ✅ | ✗ |
| Model registry | ✅ | ✗ |
| Rendered recommendation responses | ✗ | ✅ TTL'd |
| Hot item/user features | ✗ | ✅ TTL'd |
| Live session history | ✗ | ✅ TTL'd |

The test: **if Redis is flushed entirely, nothing is lost except latency.**
That property is what makes `allkeys-lru` eviction correct behaviour rather than
data loss, and it is why `docker-compose.yml` runs Redis with `--save ""` and a
256 MB cap.

Consequences encoded in the interfaces:

- **Every cache write carries a TTL.** `CacheBackend.set` has a mandatory
  `ttl_seconds`; there is no unbounded write. An entry with no expiry is a memory
  leak with extra steps, and a stale recommendation that never expires is worse
  than a slow one.
- **A cache failure is never a request failure.** `get` returns `None` on miss
  *and* on transport failure — indistinguishable by design, because the caller
  must proceed either way. The recommendation path must survive Redis being down.
- **Cache keys embed `model_version`**, so a deploy self-invalidates without an
  explicit flush.
- **Session history is Redis-only and ephemeral.** Durable history is the
  interaction log; the session key is a working set with a TTL.

### Migrations

No migration framework in Phase 1, deliberately: there is no deployed database
and no prior schema, so Alembic would add a version table and a toolchain for
nothing. `schema.sql` is applied by docker-compose on first start.

**The adoption trigger is explicit: the first change to `schema.sql` after data
exists in a database somebody cares about.** At that point Alembic is introduced
with the current schema as the baseline revision.

## Alternatives considered

**PostgreSQL only.** Fewer moving parts, and `UNLOGGED` tables are quite fast.
Rejected: response caching is a high-rate read/write path with natural expiry, and
implementing TTLs and eviction in PostgreSQL means rebuilding what Redis already
does — plus the cache would then compete with the event log for the same
connection pool.

**Redis as primary store with PostgreSQL for archive.** Faster writes. Rejected:
the interaction log is the training corpus and must be durable and queryable by
time range. Losing it to an eviction policy is unrecoverable.

**Adding a message queue now.** Would decouple ingestion from the write path.
Deferred: batch retraining does not need it, and it adds a third stateful system
before there is evidence of a write bottleneck.

## Consequences

**Positive.** A clean recovery story — restore PostgreSQL and the system is
whole. Redis can be resized, flushed, or restarted without ceremony. The
partitioned event log makes temporal reads and retention cheap
([`database_schema.md`](../data/database_schema.md)).

**Negative.** Two systems to run locally, though `make up` covers it and neither
is required by the Phase 1 test suite. A cold cache after a Redis restart means a
latency spike until it refills. Session history in Redis means an eviction under
memory pressure degrades session recommendations — acceptable, because the
fallback chain covers it, and because the alternative is a per-request write to
PostgreSQL.
