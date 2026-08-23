# ADR-001: Modular monolith before microservices

## Status

Accepted — 2026-08-24. Revisit when a component's scaling profile measurably
diverges from the rest.

## Context

The pipeline has clearly separable stages — ingestion, preprocessing, training,
retrieval, ranking, serving — and it is tempting to give each its own service.
The failure mode that argues against it is specific and common: a recommendation
system split into services early acquires a network hop between retrieval and
ranking, a serialisation format for candidate lists, and a distributed failure
mode on the request path — before anyone has measured whether retrieval and
ranking even have different scaling needs.

The initial deployment target is one developer machine: Apple Silicon, 16 GB
RAM, no GPU. There is nothing to scale independently because there is only one
of everything.

## Decision

**One deployable process, with strictly enforced module boundaries.**

- Every component in [`component_contracts.md`](../architecture/component_contracts.md)
  is a module with an explicit interface — an ABC or a Protocol.
- Components communicate only through those interfaces, never through each
  other's internals.
- `core` imports nothing from any other subpackage; `data` imports nothing from
  `models`, `retrieval`, or `api`. **This is enforced by a test**
  (`TestLayering` in `tests/integration/test_repository_smoke.py`) that walks the
  AST, not by convention.
- `docker-compose.yml` runs PostgreSQL and Redis only. There is no application
  container in Phase 1, because there is nothing to orchestrate.

The boundaries are real; the deployment is one unit.

## Alternatives considered

**Microservices from the start** — separate retrieval, ranking, and serving
services. Rejected: it pays the full distributed-systems cost (service discovery,
network partitions, distributed tracing, deploy coordination) for a benefit
(independent scaling) that a single-machine system cannot use. It also makes the
serving path slower, since retrieval → ranking becomes a network call carrying a
few hundred candidates.

**A single-file script / notebook pipeline** — fastest to a first result.
Rejected: it makes the training and serving paths structurally different, which
is the root cause of most training/serving skew. It also has no seam at which
serving code could ever be added.

**A modular monolith with no enforced layering** — interfaces by convention.
Rejected: dependency rules that are not tested are dependency rules that will be
violated, usually by a quick fix under time pressure. The AST test costs almost
nothing and makes the rule real.

## Consequences

**Positive.** One process to run, debug, and profile. In-process calls between
stages, so no serialisation on the request path. A refactor that moves logic
between components is a normal code change. Interfaces exist from day one, so
extracting a service later is mechanical rather than archaeological.

**Negative.** No independent scaling — a memory-heavy embedding model shares a
process with the API. No language heterogeneity. A crash takes everything down;
mitigated because the fallback chain lives in the same process and needs only a
popularity table.

**Extraction trigger.** Split a component out when its resource profile
measurably diverges — e.g. embedding inference needing a GPU host while serving
stays on CPU. The interfaces make that a wrapper, not a rewrite.
