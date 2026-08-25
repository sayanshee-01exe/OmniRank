# ADR-009: Accepting `KMP_DUPLICATE_LIB_OK` for FAISS and PyTorch coexistence

## Status

Accepted — 2026-08-25, in Phase 4.

## Context

`faiss-cpu` and `torch` each ship their own copy of `libomp.dylib`:

```
.venv/lib/python3.11/site-packages/faiss/.dylibs/libomp.dylib   755 KB
.venv/lib/python3.11/site-packages/torch/lib/libomp.dylib       856 KB
```

On macOS the LLVM OpenMP runtime **aborts the process** when the second copy
initialises:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

This is not a test-only problem. Phase 4's central workflow is *train a torch
model, export its item embeddings, build a FAISS index over them* — so every
path that matters loads both libraries into one process. The full test suite
aborted the interpreter at the first FAISS call after any torch test.

Import order does not help. Both copies load regardless of which package is
imported first; this was measured, not assumed.

### Options considered

**Symlink one libomp to the other.** This is what the OpenMP hint itself
recommends — a single runtime in the process. Rejected as the primary mechanism
because it mutates `site-packages`, does not survive `pip install
--force-reinstall`, silently reverts on any dependency upgrade, and would have
to be reproduced identically in CI and on every contributor's machine. A fix
that disappears without warning is worse than a documented one.

**A different FAISS build.** conda-forge's FAISS links the system libomp and
would avoid the duplication. Rejected because the project standardises on `uv`
and PyPI wheels; introducing conda for one package changes the install story for
everyone.

**Avoid coexistence.** Build indexes in a separate process from training.
Rejected: it adds a process boundary and a serialisation step to the middle of
the offline pipeline, to work around a library packaging detail.

**Set `KMP_DUPLICATE_LIB_OK=TRUE`.** Permits the second initialisation. LLVM
documents this as "an unsafe, unsupported, undocumented workaround" that "may
cause crashes or silently produce incorrect results."

## Decision

Set `KMP_DUPLICATE_LIB_OK=TRUE` **and** pin FAISS to a single OpenMP thread,
both inside `_require_faiss()` in
[`retrieval/faiss_index.py`](../../src/omnirank/retrieval/faiss_index.py) — the
one place FAISS is imported. `os.environ.setdefault` is used, so an operator who
has set the variable deliberately is not overridden.

Two things make this acceptable rather than a hopeful workaround:

1. **The failure mode is disarmed.** The documented danger is two OpenMP
   runtimes contending over a shared thread pool. `faiss.omp_set_num_threads(1)`
   removes the contention: FAISS does not spawn a parallel region at all. The
   cost is single-threaded index queries, which at 69,347 vectors is not the
   bottleneck.

2. **The remaining risk is measured, not assumed.** The claim that results are
   uncorrupted is verified on every test run: `flat_ip` must reproduce exact
   brute force in both set *and* order, and those tests execute **after** torch
   has initialised, in the same process. Silent numerical corruption is exactly
   what they fail on. This is the difference between "we set a flag and hope"
   and "we set a flag and check".

## Consequences

**Positive.** One process trains a model and indexes its embeddings, with no
process boundary in the offline pipeline. The workaround lives in one function
with the reasoning next to it, and applies identically on every machine and in
CI without any environment setup.

**Negative.** FAISS index queries are single-threaded. If index size grows to
where query throughput matters, this becomes a real constraint and the decision
should be revisited — most likely by moving to a managed vector service, which
[ADR-004](ADR-004-faiss-initial-index.md) already anticipates as the successor.

**Watch item.** If the exactness tests ever fail on a machine where they pass
elsewhere, this ADR is the first place to look. That is the intended tripwire.

## Related

- [ADR-004](ADR-004-faiss-initial-index.md) — FAISS as the initial index
- [`docs/retrieval/faiss_index.md`](../retrieval/faiss_index.md)
