-- ===========================================================================
-- OmniRank PostgreSQL schema (Phase 1).
--
-- Applied automatically by docker-compose on first start of the postgres
-- container. Idempotent, so re-running it against an existing database is safe.
--
-- Scope discipline: PostgreSQL holds durable truth (entities, event log,
-- served-recommendation audit, model registry). Everything derived and
-- disposable - candidate lists, response caches, embeddings, indexes - lives in
-- Redis or on disk. See ADR-005.
--
-- Design notes are inline. The full rationale, including partitioning and
-- retention, is in docs/data/database_schema.md.
-- ===========================================================================

CREATE SCHEMA IF NOT EXISTS omnirank;
SET search_path TO omnirank, public;

-- ---------------------------------------------------------------------------
-- Enumerations
-- ---------------------------------------------------------------------------
-- An enum rather than a lookup table: the vocabulary is small, stable, and
-- shared with the EventType StrEnum in src/omnirank/data/schemas.py. Adding a
-- value is an explicit migration, which is the desired friction - an unnoticed
-- new event type would silently skew every implicit-feedback weight.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'event_type') THEN
        CREATE TYPE omnirank.event_type AS ENUM (
            'view', 'click', 'wishlist', 'cart', 'purchase', 'rating'
        );
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'artifact_type') THEN
        CREATE TYPE omnirank.artifact_type AS ENUM (
            'mapping', 'retrieval_model', 'ranker', 'embedding', 'index', 'feature_set'
        );
    END IF;
END
$$;

-- ---------------------------------------------------------------------------
-- Users
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    -- The external, opaque identifier is the primary key. No surrogate integer:
    -- a second identity space would have to be kept in sync with the id
    -- mappings, and dense indices deliberately never reach the database.
    user_id       TEXT        PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL,
    -- Domain-specific context. JSONB rather than columns, so onboarding a new
    -- vertical needs no migration.
    attributes    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_users_created_at ON users (created_at);

-- ---------------------------------------------------------------------------
-- Items
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS items (
    item_id       TEXT        PRIMARY KEY,
    title         TEXT        NOT NULL CHECK (length(title) > 0),
    description   TEXT,
    category      TEXT,
    brand         TEXT,
    -- Non-negative, and NULL-able because plenty of verticals have no price.
    price         NUMERIC(12, 2) CHECK (price IS NULL OR price >= 0),
    image_id      TEXT,
    available     BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL,
    attributes    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Serving filters unavailable items on every request; a partial index keeps
-- that cheap and stays small because most of the catalogue is available.
CREATE INDEX IF NOT EXISTS idx_items_available   ON items (item_id) WHERE available;
CREATE INDEX IF NOT EXISTS idx_items_category    ON items (category) WHERE category IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_items_created_at  ON items (created_at);

-- ---------------------------------------------------------------------------
-- Interactions  (the event log - by far the largest table)
-- ---------------------------------------------------------------------------
-- PARTITIONING: declaratively partitioned by RANGE on occurred_at, monthly.
-- Reasons, in order of importance:
--   1. Temporal splitting reads a contiguous time window; partition pruning
--      turns that from a full scan into a few partition scans.
--   2. Retention is a DETACH + DROP of an old partition - instant, no bloat,
--      no long-running DELETE holding locks.
--   3. Index maintenance stays bounded per partition as the log grows.
-- Cost: the partition key must be part of every unique constraint, hence the
-- composite primary key below.
CREATE TABLE IF NOT EXISTS interactions (
    interaction_id TEXT        NOT NULL,
    user_id        TEXT        NOT NULL,
    item_id        TEXT        NOT NULL,
    event_type     omnirank.event_type NOT NULL,
    occurred_at    TIMESTAMPTZ NOT NULL,
    session_id     TEXT,
    event_value    DOUBLE PRECISION,
    weight         DOUBLE PRECISION CHECK (weight IS NULL OR weight >= 0),
    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (interaction_id, occurred_at),

    -- A rating with no value is unusable; reject it at the boundary, matching
    -- the same rule in the Pydantic contract.
    CONSTRAINT chk_rating_has_value
        CHECK (event_type <> 'rating' OR event_value IS NOT NULL)
) PARTITION BY RANGE (occurred_at);

-- Business-key uniqueness. This is the database-level enforcement of
-- Interaction.dedup_key: re-delivery of the same event with a fresh
-- interaction_id is idempotent rather than double-counted.
CREATE UNIQUE INDEX IF NOT EXISTS uq_interactions_event
    ON interactions (user_id, item_id, event_type, occurred_at);

-- The two access patterns that matter: "this user's recent history" (sequence
-- building, serving) and "this item's recent traffic" (popularity, analytics).
CREATE INDEX IF NOT EXISTS idx_interactions_user_time ON interactions (user_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_interactions_item_time ON interactions (item_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_interactions_session
    ON interactions (session_id, occurred_at) WHERE session_id IS NOT NULL;

-- NOTE ON FOREIGN KEYS: interactions deliberately carry NO foreign keys to
-- users/items. Events arrive before (or without) their entities in every real
-- ingestion pipeline, and a rejected event is lost signal. Referential
-- integrity is enforced in the validation stage
-- (omnirank.data.validation.validate_interactions), which *reports* dangling
-- references instead of destroying them. See docs/data/database_schema.md.

-- A default partition catches rows outside every declared range, so ingestion
-- never fails on an unexpected date. It should stay empty: a non-zero count is
-- the alert that a partition is missing.
CREATE TABLE IF NOT EXISTS interactions_default PARTITION OF interactions DEFAULT;

-- ---------------------------------------------------------------------------
-- Model registry
-- ---------------------------------------------------------------------------
-- Mirrors ArtifactMetadata (src/omnirank/artifacts/metadata.py). The filesystem
-- registry remains authoritative for a single-machine setup; this table is what
-- makes the registry shared once more than one process writes to it.
CREATE TABLE IF NOT EXISTS model_registry (
    id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name              TEXT        NOT NULL,
    model_version           TEXT        NOT NULL,
    model_type              omnirank.artifact_type NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL,
    training_data_version   TEXT        NOT NULL,
    feature_version         TEXT        NOT NULL,
    configuration_hash      TEXT        NOT NULL,
    random_seed             INTEGER     NOT NULL,
    framework_version       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    python_version          TEXT        NOT NULL,
    git_commit              TEXT,
    metrics                 JSONB       NOT NULL DEFAULT '{}'::jsonb,
    supported_device        TEXT        NOT NULL DEFAULT 'any',
    required_index_version  INTEGER,
    artifact_path           TEXT,
    -- Exactly one version per model may be active for serving at a time.
    is_active               BOOLEAN     NOT NULL DEFAULT FALSE,
    registered_at           TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT uq_model_version UNIQUE (model_name, model_version)
);

-- Enforces "at most one active version per model" in the database rather than
-- in application code, where a race between two reload calls could break it.
CREATE UNIQUE INDEX IF NOT EXISTS uq_model_active
    ON model_registry (model_name) WHERE is_active;

CREATE INDEX IF NOT EXISTS idx_model_registry_created ON model_registry (created_at DESC);

-- ---------------------------------------------------------------------------
-- Recommendation requests and results  (serving audit trail)
-- ---------------------------------------------------------------------------
-- Why store these at all: without a record of what was shown, no offline
-- evaluation can be connected to online behaviour, and no counterfactual
-- analysis is possible. This pair is the log that makes future online
-- evaluation feasible - which is why it is defined in Phase 1 even though
-- nothing writes to it until Phase 6.
CREATE TABLE IF NOT EXISTS recommendation_requests (
    request_id     UUID        PRIMARY KEY,
    user_id        TEXT,               -- NULL for anonymous/session requests
    session_id     TEXT,
    endpoint       TEXT        NOT NULL,
    requested_k    INTEGER     NOT NULL CHECK (requested_k > 0),
    context        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    model_version  TEXT,
    fallback_used  BOOLEAN     NOT NULL DEFAULT FALSE,
    latency_ms     INTEGER     CHECK (latency_ms IS NULL OR latency_ms >= 0),
    requested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT chk_request_identity CHECK (user_id IS NOT NULL OR session_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_rec_requests_user_time
    ON recommendation_requests (user_id, requested_at DESC) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_rec_requests_time ON recommendation_requests (requested_at DESC);

CREATE TABLE IF NOT EXISTS recommendation_results (
    request_id   UUID        NOT NULL REFERENCES recommendation_requests (request_id)
                             ON DELETE CASCADE,
    item_id      TEXT        NOT NULL,
    rank         INTEGER     NOT NULL CHECK (rank > 0),
    score        DOUBLE PRECISION NOT NULL,
    -- Which generators nominated the item, for attribution.
    sources      TEXT[]      NOT NULL DEFAULT ARRAY[]::TEXT[],
    reason       TEXT,

    PRIMARY KEY (request_id, rank),
    -- An item must not appear twice in one response.
    CONSTRAINT uq_result_item UNIQUE (request_id, item_id)
);

CREATE INDEX IF NOT EXISTS idx_rec_results_item ON recommendation_results (item_id);

-- ---------------------------------------------------------------------------
-- Feedback events  (what happened to what we showed)
-- ---------------------------------------------------------------------------
-- Separate from `interactions` on purpose. An interaction is an observation
-- about the world; a feedback event is an observation about *our own
-- recommendation*, and joining it to request_id is what enables CTR, online
-- evaluation, and eventually counterfactual training. Merging the two tables
-- would make "was this organic or recommended?" unanswerable.
CREATE TABLE IF NOT EXISTS feedback_events (
    feedback_id  UUID        PRIMARY KEY,
    request_id   UUID        REFERENCES recommendation_requests (request_id) ON DELETE SET NULL,
    user_id      TEXT,
    item_id      TEXT        NOT NULL,
    event_type   omnirank.event_type NOT NULL,
    position     INTEGER     CHECK (position IS NULL OR position > 0),
    occurred_at  TIMESTAMPTZ NOT NULL,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feedback_request ON feedback_events (request_id);
CREATE INDEX IF NOT EXISTS idx_feedback_time    ON feedback_events (occurred_at DESC);

-- ---------------------------------------------------------------------------
-- Partition management
-- ---------------------------------------------------------------------------
-- Monthly partitions are created on demand rather than pre-created for a decade.
-- Phase 2's ingestion job calls this before writing a batch; until then it is
-- callable by hand:  SELECT omnirank.ensure_month_partition('2026-08-01');
CREATE OR REPLACE FUNCTION omnirank.ensure_month_partition(target DATE)
RETURNS TEXT
LANGUAGE plpgsql
AS $$
DECLARE
    window_start DATE := date_trunc('month', target)::DATE;
    window_end   DATE := (date_trunc('month', target) + INTERVAL '1 month')::DATE;
    part_name    TEXT := format('interactions_%s', to_char(window_start, 'YYYY_MM'));
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_class WHERE relname = part_name AND relkind = 'r'
    ) THEN
        EXECUTE format(
            'CREATE TABLE omnirank.%I PARTITION OF omnirank.interactions '
            'FOR VALUES FROM (%L) TO (%L)',
            part_name, window_start, window_end
        );
    END IF;
    RETURN part_name;
END;
$$;

-- Retention: detaching is preferred over dropping, so an archived partition can
-- be exported before it is destroyed. Called by an operator or a scheduled job;
-- deliberately NOT automatic, because silently deleting the training corpus is
-- not a thing a service should do on its own.
--   SELECT omnirank.detach_partitions_before('2024-01-01');
CREATE OR REPLACE FUNCTION omnirank.detach_partitions_before(cutoff DATE)
RETURNS SETOF TEXT
LANGUAGE plpgsql
AS $$
DECLARE
    part RECORD;
BEGIN
    FOR part IN
        SELECT c.relname
        FROM pg_inherits i
        JOIN pg_class c   ON c.oid = i.inhrelid
        JOIN pg_class p   ON p.oid = i.inhparent
        WHERE p.relname = 'interactions'
          AND c.relname ~ '^interactions_[0-9]{4}_[0-9]{2}$'
          AND to_date(right(c.relname, 7), 'YYYY_MM') < date_trunc('month', cutoff)
    LOOP
        EXECUTE format('ALTER TABLE omnirank.interactions DETACH PARTITION omnirank.%I',
                       part.relname);
        RETURN NEXT part.relname;
    END LOOP;
END;
$$;
