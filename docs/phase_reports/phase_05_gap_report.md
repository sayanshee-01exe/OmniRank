# Phase 5 gap report — the Phase 6 gate did not pass

> **SUPERSEDED — historical record.** This document captures the gate's verdict
> on 2026-08-25, when Phase 5 was incomplete and Phase 6 was correctly refused.
> Every gap listed below has since been closed. The current state is in
> [phase_05_report.md](phase_05_report.md), and the gate now exits 0.
>
> Kept rather than deleted because it records *why* Phase 6 was not started, and
> a closed gate with no record of what it once refused is a weaker artifact than
> one with the history attached.

**Date:** 2026-08-25 · **Gate:** `scripts/validate_phase5.py` · **Exit code:** 1 ·
**Result:** 3 / 15 checks passed, **11 critical failures**, 1 warning ·
**Machine-readable:** `reports/metrics/phase_06/phase5_gate_report.json`

---

## Verdict

**Phase 6 has not been started.** Phase 6's own §1 makes the Phase 5 completion
gate mandatory and instructs that a failing gate produces this document and
stops. It failed, so this document is the deliverable.

Phase 5 was in progress when the Phase 6 brief arrived. Its data foundation is
complete and verified against the real corpus; its model is not built.

---

## What the gate checked, and what it found

| Check | Result | Detail |
|---|---|---|
| `two_tower` source modules | **FAIL** | `model.py` missing |
| `two_tower` supporting modules | WARN | no `dataset`/`training`/`persistence`/`features` modules |
| `MultimodalTwoTower` importable | **FAIL** | `ImportError` — the class does not exist |
| Phase 5 configurations | **FAIL** | all three missing |
| Phase 5 commands | **FAIL** | `compare_multimodal_retrievers.py` missing |
| Phase 5 evidence | **FAIL** | report and `selected_configuration.json` missing |
| Multimodal feature manifest | ✅ PASS | text and image both available |
| Feature coverage non-zero | ✅ PASS | text 1.000, image 1.000 |
| Features carry mapping identity | ✅ PASS | `item_mapping_checksum` present |
| Registered two-tower model | **FAIL** | no `artifacts/metadata/two_tower/*.json` |
| Registered two-tower FAISS index | **FAIL** | no `artifacts/indexes/*/two_tower/*` |
| Model/index/feature/mapping compatibility | **FAIL** | skipped — nothing registered to check |
| Saved-model smoke recommendation | **FAIL** | skipped — nothing to load |
| Cold item present in the index | **FAIL** | skipped — nothing to load |
| Phase 5 report records cold-item recall | **FAIL** | report does not exist |

---

## What Phase 5 *did* complete

This is real, verified work, not partial credit.

### Phase 4 closeout (§3 of the Phase 5 brief)

| Item | State |
|---|---|
| §3.1 README status corrected | ✅ Phase 4 "core complete with documented limitations", Phase 5 "current" |
| §3.2 Tracked selection provenance | ✅ `provenance/phase_0{3,4}_selection.json`, CI gate added |
| §3.3 SASRec registered for fusion | ⏳ **refit in progress** — 45-epoch fit on train+validation, ~epoch 25/45 |
| §3.4 Rolling folds run on locked models | ❌ not started |
| §3.5 Paired bootstrap | ❌ not started |

A real defect was fixed during §3.2: the Phase 4 lock record carried no dataset
identity, so a locked configuration could not be tied to the data it was
selected on. `_run_lock` now records it, and the tracked provenance file
verifies it.

### Multimodal feature acquisition (§4–§8)

Complete and verified against the **real** PixelRec files, not fixtures:

| | Measured |
|---|---|
| Download | 18.5 GB in ~12 min at 25 MB/s |
| `text_feature.json` | 8.65 GB, sha256 `d3376b5ec9593fde…` |
| `image_feature.json` | 8.60 GB, sha256 `b74fb95313afe53b…` |
| Dimensions | **1024** for both — read from the files, not assumed |
| Coverage | **100%** — all 69,347 catalogue items carry text *and* image |
| Data quality | 0 NaN, 0 inf, 0 duplicate ids, 0 dimension mismatches |
| Alignment cost | 168 s (text) + 141 s (image), 320 MB peak |
| Storage | float32 memory-mapped `.npy`, 271 MB per modality |
| Encoder identity | recorded as `unknown` — PixelRec documents no encoder, so none is claimed |

Delivered: `scripts/prepare_multimodal_features.py`,
`src/omnirank/features/multimodal_store.py`,
`src/omnirank/models/two_tower/{config,losses}.py`.

### Findings from that work

**A latent O(n²) bug in Phase 2's `align_features`.** It performed
`index.loc[index["internal_item_id"] == internal, ...] = True` *inside* the
per-item loop — a full DataFrame scan per matched item, roughly 4.8 billion
comparisons at this catalogue size. It had never fired because feature coverage
had always been 0.0, so the loop body never executed. Vectorised; alignment now
completes in five minutes.

**Float16 storage measured and rejected.** §6 required a numeric comparison
before adopting it. Measured: max absolute error 9.4e-04, max dot-product error
7.5e-04 — enough to reorder adjacent candidates — and max *relative* error of
1.0, meaning some values round entirely to zero. The saving is 284 MB on a 16 GB
machine. float32 retained; the measurement is recorded in the manifest.

**100% modality coverage has a consequence for §11 and §16.** There are **no**
text-only, image-only, or no-modality items in the real corpus. The
missing-modality evaluation views will be empty on real data. The handling must
still be implemented and fixture-tested — the code path has to be correct for
other corpora — but the real-data result will be reported as *not exercised*
rather than filled in.

---

## Every missing item, in dependency order

### 1. Phase 4 closeout remainder

- [ ] §3.3 — finish the SASRec refit, register the artifact, evaluate test once, build and verify its FAISS index
- [ ] §3.4 — run locked Phase 4 models on rolling folds at offsets 3 and 2
- [ ] §3.5 — paired user-level bootstrap for LightGCN−popularity, LightGCN−BPR, 4-source RRF−LightGCN

### 2. Two-tower model (§10)

- [ ] `model.py` — user tower, item tower, shared embedding space
- [ ] `dataset.py` — training examples from the Phase 2 sequential contract
- [ ] `training.py` — contrastive training loop, MPS with safe fallback
- [ ] `persistence.py` — save/load with feature, mapping and modality identity
- [ ] `__init__.py` — export `MultimodalTwoTower`
- [ ] §10.3 warm-masked ID residual — `content + warm_mask × id_residual`

### 3. Configuration (§24)

- [ ] `configs/models/two_tower.yaml`
- [ ] `configs/features/pixelrec_published.yaml`
- [ ] `configs/models/phase5_selected.yaml` (generated after locking)

### 4. Cold-item capability (§16) — the phase's primary purpose

- [ ] Cold-inclusive item catalogue
- [ ] Cold-item evaluation views
- [ ] **Mandatory cold-item fixture** where a collaborative model cannot return the target and the two-tower can
- [ ] Cold Recall@K > 0 at a meaningful K

### 5. Retrieval and fusion (§17–§19)

- [ ] Exact FAISS index over two-tower embeddings, verified against brute force
- [ ] Five-source fusion vs the existing four
- [ ] Source overlap and unique-contribution diagnostics

### 6. Selection and evaluation (§14, §15, §20)

- [ ] Modality ablations (§15 lists seven, plus six leave-one-out variants)
- [ ] Rolling-fold selection, lock before test
- [ ] Final test evaluation, paired bootstrap

### 7. Commands, tests, CI, docs (§23, §26, §27, §30)

- [ ] `scripts/compare_multimodal_retrievers.py`
- [ ] `train.py` / `evaluate.py` / `build_index.py` support for `two_tower`
- [ ] Feature-store, tower, loss, leakage, cold-start, FAISS and fusion tests
- [ ] `multimodal-retrieval` CI job
- [ ] Eight Phase 5 documents and `phase_05_report.md`

---

## Why the gate is worth honouring rather than working around

Phase 6 §5 defines the ranker as scoring a candidate pool it does not itself
generate, and §15 requires reporting candidate recall precisely because *the
ranker cannot recover a target absent from its pool*. Phase 6's cold-item
slices, its five-source snapshot schema (§7), its two-tower similarity source
for MMR (§18), and its bundle manifest (§21) all reference a two-tower model
that does not exist.

Building Phase 6 now would mean a `two_tower_present` column that is always
false, an MMR similarity source with no embeddings, and a bundle that validates
components it cannot load. None of that fails loudly — it produces a system that
looks complete and quietly has four sources where five were designed.

The gate exists to stop exactly that, and its own brief says so: *"Do not
silently complete Phase 5 and Phase 6 in one uncontrolled change set."*

---

## Recommended next step

Finish Phase 5, then re-run the gate:

```bash
python scripts/validate_phase5.py
```

The critical path is the two-tower model itself. The feature foundation it
needs is built and verified, so the remaining work is the model, its ablations,
selection, and evidence — not another data-acquisition cycle.

Estimated compute, from measured Phase 4 and Phase 5 rates: the outstanding
SASRec refit is ~35 minutes; two-tower training throughput has not yet been
measured and will be, as in Phase 4, before the search grid is sized.
