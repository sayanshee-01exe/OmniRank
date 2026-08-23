# PostgreSQL schema

Authoritative definition: [`src/omnirank/database/schema.sql`](../../src/omnirank/database/schema.sql).
Applied automatically by `docker compose up` on first start of an empty data
directory. Idempotent — safe to re-run.

## Scope discipline

PostgreSQL holds **durable truth**. Everything derived and disposable —
candidate lists, response caches, embeddings, indexes — lives in Redis or on
disk. See [ADR-005](../adr/ADR-005-postgres-redis-responsibilities.md).

## Tables

| Table | Purpose | Key |
|---|---|---|
| `users` | Recommendation subjects | `user_id` |
| `items` | Catalogue | `item_id` |
| `interactions` | The event log (partitioned) | `(interaction_id, occurred_at)` |
| `model_registry` | Artifact manifests, shared across processes | `id`, unique `(model_name, model_version)` |
| `recommendation_requests` | What was asked | `request_id` |
| `recommendation_results` | What was shown | `(request_id, rank)` |
| `feedback_events` | What happened to what was shown | `feedback_id` |

### Identity

External opaque ids are the primary keys — no surrogate integers. A second
identity space would have to be kept in sync with the `IdMapping` files, and
dense indices deliberately never reach the database.

### `interactions` — partitioning

Declaratively partitioned by **RANGE on `occurred_at`, monthly**. Three reasons,
in order of weight:

1. **Temporal splitting reads a contiguous window.** Partition pruning turns a
   full scan into a few partition scans — and temporal splitting is the single
   most frequent heavy read in the system.
2. **Retention becomes `DETACH` + `DROP`** — instant, no table bloat, no
   long-running `DELETE` holding locks on the largest table.
3. **Index maintenance stays bounded** per partition as the log grows.

The cost, stated plainly: the partition key must appear in every unique
constraint, which is why the primary key is composite
`(interaction_id, occurred_at)` rather than just `interaction_id`.

Partitions are created **on demand**, not pre-created for a decade:

```sql
SELECT omnirank.ensure_month_partition('2026-08-01');
```

A `DEFAULT` partition catches rows outside every declared range, so ingestion
never fails on an unexpected date. **It should stay empty** — a non-zero count
there is the alert that a partition is missing.

### `interactions` — no foreign keys

Deliberate. Events arrive before (or without) their entities in every real
ingestion pipeline, and a rejected event is lost signal that cannot be
recovered. Referential integrity is enforced in
`omnirank.data.validation.validate_interactions`, which **reports** dangling
references instead of destroying them.

### `interactions` — deduplication

```sql
CREATE UNIQUE INDEX uq_interactions_event
    ON interactions (user_id, item_id, event_type, occurred_at);
```

The database-level enforcement of `Interaction.dedup_key`. Re-delivery of the
same event with a fresh `interaction_id` is idempotent rather than
double-counted — which is what makes at-least-once delivery from a client safe.

### `recommendation_requests` + `recommendation_results`

The serving audit trail. Without a record of what was shown, no offline
evaluation can be connected to online behaviour, and no counterfactual analysis
is possible. This pair is defined in Phase 1 even though nothing writes to it
until Phase 6, because retrofitting an audit log means losing the history
between now and then.

`chk_request_identity` requires at least one of `user_id` / `session_id` —
anonymous requests are legitimate, identity-less ones are not.

### `feedback_events` — why separate from `interactions`

An **interaction** is an observation about the world. A **feedback event** is an
observation about *our own recommendation*, joined to `request_id`. Merging the
two would make "was this click organic, or did we cause it?" permanently
unanswerable — and that question is the entire basis of CTR, online evaluation,
and any future counterfactual training.

### `model_registry`

Mirrors `ArtifactMetadata`. The filesystem registry stays authoritative for a
single-machine setup; this table is what makes the registry shared once more
than one process writes to it.

```sql
CREATE UNIQUE INDEX uq_model_active ON model_registry (model_name) WHERE is_active;
```

"At most one active version per model" enforced in the database, not in
application code where a race between two reload calls could break it.

## Index summary

| Index | Table | Serves |
|---|---|---|
| `idx_users_created_at` | users | cohort queries |
| `idx_items_available` (partial) | items | availability filter on every request |
| `idx_items_category` (partial) | items | category-popularity fallback |
| `uq_interactions_event` | interactions | idempotent ingestion |
| `idx_interactions_user_time` | interactions | user history, sequence building |
| `idx_interactions_item_time` | interactions | item popularity |
| `idx_interactions_session` (partial) | interactions | session recommendations |
| `uq_model_active` (partial) | model_registry | one active version per model |
| `idx_rec_requests_user_time` (partial) | rec_requests | per-user audit |
| `idx_feedback_request` | feedback_events | attribution join |

Partial indexes are used wherever the filtered subset is the one actually
queried — they stay small and are cheap to maintain.

## Retention

| Table | Policy |
|---|---|
| `interactions` | Detach partitions older than the training window; archive before dropping |
| `recommendation_requests` / `_results` | Highest-volume tables. 90 days is a reasonable default; they exist for evaluation, not permanent record |
| `feedback_events` | Keep as long as the requests they join to |
| `users` / `items` | Indefinite; current state, not history |

```sql
SELECT omnirank.detach_partitions_before('2024-01-01');
```

Detaching is preferred over dropping so an archived partition can be exported
before destruction. **Deliberately not automatic** — silently deleting the
training corpus is not something a service should do on its own.

## Migrations

**None in Phase 1, deliberately.** There is no deployed database to migrate and
no prior schema to migrate from, so Alembic would add a version table and a
toolchain in exchange for nothing.

Adopted in Phase 2, at the first change to a schema that already holds data. The
trigger is explicit: **the moment `schema.sql` changes after data exists**, the
change goes through a migration instead. Recorded in
[ADR-005](../adr/ADR-005-postgres-redis-responsibilities.md).
