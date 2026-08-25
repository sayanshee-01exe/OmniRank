# Phase 5a report — two-tower model core

**Date:** 2026-08-25 · **Scope:** the first missing Phase 5 milestone — a
trainable, testable, persistable multimodal two-tower model ·
**Status:** complete · **Phase 5 overall:** still incomplete (see below)

---

## Headline

The two-tower model core exists, trains on real PixelRec data, and **encodes
cold items from content alone** — verified against the 770 genuine PixelRec50K
cold items, not only on fixtures.

| | Result |
|---|---|
| Unit tests | **108 passing** across dataset, towers, loss, training, persistence |
| Integration test | **11 passing**, including the mandatory cold-item workflow |
| Full suite | **1,422 passing** |
| Ruff / MyPy `--strict` | clean |
| Real-data smoke run | 1,000 users, 8,558 examples, 12.6M params, loss 5.530 → 4.997 |
| Cold guarantee on real data | 64/64 sampled cold items encode identically with and without the ID residual |

**The one decision the milestone turns on** is §10.3's warm-masked residual:

```python
final_item_embedding = content_embedding + warm_item_mask * item_id_residual
```

If that gate leaks, nothing fails. The model trains, the loss falls, warm
metrics look normal, and cold recall reads zero for a reason no warm number
reveals. So it is asserted as an exact equality against the content-only path,
with a companion test proving the residual is non-trivial for warm items — so
the first test cannot pass merely because nothing is being added.

---

## Repository state before this milestone

Phase 5 had its data foundation and nothing else. Present: verified feature
download, 100%-coverage alignment, `MultimodalFeatureStore`, `config.py`,
`losses.py`, the Phase 5 validator and gap report. Absent: the model itself,
its dataset, trainer, persistence, configuration, tests and CLI.

---

## Files created

| File | Purpose |
|---|---|
| `models/two_tower/dataset.py` | Training examples and collation from Phase 2 sequences |
| `models/two_tower/model.py` | `ModalityEncoder`, `ItemTower`, `UserTower`, `MultimodalTwoTower` |
| `models/two_tower/training.py` | `TwoTowerTrainer`, `TrainingHistory` |
| `models/two_tower/persistence.py` | Device-neutral save/load with identity enforcement |
| `configs/models/two_tower.yaml` | Development configuration |
| `tests/unit/models/two_tower/` | conftest + 4 suites, 108 tests |
| `tests/integration/test_two_tower_training.py` | End-to-end cold-item workflow, 11 tests |
| `docs/models/multimodal_two_tower_core.md` | Architecture and the cold-item argument |
| `docs/models/two_tower_training.md` | Objective, false negatives, device, measurements |
| `docs/models/two_tower_persistence.md` | Format and identity enforcement |

## Files modified

| File | Change |
|---|---|
| `models/two_tower/config.py` | Six fields the suggested YAML needed but the config lacked |
| `models/two_tower/__init__.py` | Replaced the "NOT IMPLEMENTED" placeholder |
| `retrieval/runner.py` | `fit_two_tower`, `load_item_tags` |
| `scripts/train.py` | `two_tower`, `--subset-users`, `--model-config`, `development` stage |
| `tests/integration/test_repository_smoke.py` | Torch-import skip list now prefix-based |

---

## Design decisions worth stating

### Padding is `num_items`

One past the last valid internal id, matching SASRec. Reusing 0 would collide
with a real item and train the model on content belonging to something else.
Padded positions are additionally flagged as having *no* modality, so a padded
slot cannot look like an item that happens to have text.

### History items use the content path only

The user tower encodes history through `encode_content`, never `forward`. Using
the identity residual there would make a user's query depend on item ids, and an
unknown-user request built from a supplied history would then require those ids
to be warm.

### The dataset is torch-free

It produces numpy and holds only item *ids*; features are fetched from the
memory-mapped store at collate time. Materialising history features per example
would turn 776k rows into hundreds of gigabytes, and copying matrices into
workers would multiply the store by the worker count for no benefit. Every
device transfer happens in one place in the trainer, which is also what makes a
partially-moved model impossible.

### Engagement counters are excluded structurally

`item_metadata.parquet` carries PixelRec's platform counters (`view_number`,
`thumbup_number`, …) in `source_metadata`. `load_item_tags` reads only
`internal_item_id` and `category` — excluded by column selection rather than by
discipline, because they are dataset-wide totals with no guarantee of reflecting
what was known at any historical prediction time.

---

## What the work surfaced

**A wrong assumption I checked rather than assumed.** The dataset asserts a
target never appears inside its own history. That would be wrong if users
re-watch items — a legitimate repeat interaction, not leakage. Measured against
50,000 real rows: **zero** targets reappear in their own history, because Phase 2
deduplicates repeat events. The assertion is correct *for this dataset*, and now
demonstrably so. My own fixture was the thing that was unrealistic.

**A test whose premise was wrong.** The early-stopping test drove a tiny
learning rate expecting a plateau. A 1e-8 rate does not plateau, it improves
*slowly* — 1.8e-5 per epoch, above the improvement threshold — so patience never
accumulated. Replaced with a pinned validation loss, which tests the stopping
logic rather than hoping a run stalls.

**A vacuous assertion.** `float(tensor).__abs__() >= 0.0` is always true and
crashed on a multi-element tensor. Removed; the real assertion underneath it
(cold items resemble their own tag block) passes.

**A skip list with a hole.** The torch-import test listed
`omnirank.models.two_tower` but not `two_tower.config` — and importing any
submodule executes the package `__init__`, which pulls torch. Now matched by
prefix, so adding a module cannot silently reopen the hole.

---

## Real-data smoke run

```bash
python scripts/train.py --model two_tower --stage development \
  --version phase5-two-tower-dev --subset-users 1000 --epochs 3 --device cpu
```

| | |
|---|---|
| Users / examples | 1,000 / 8,558 |
| Catalogue | 69,347 items (60,445 cold within this subset) |
| Tags | 110 (108 categories + unknown + offset) |
| Parameters | 12,648,800 |
| Device | CPU |
| Loss | 5.530 → 4.997 (best epoch 2 of 3) |
| In-batch accuracy | 0.004 → 0.022 |
| Proxy recall@20 | 0.041 → 0.076 |
| Peak memory | 425 MB |
| Time | ~3.7 s/epoch |
| Artifact | 48 MB `model.pt` + config + metadata + history |

**Three epochs on 1.3% of users is a smoke run.** It demonstrates that feature
reads, dataset construction, batch shapes, training, checkpointing and loading
all work on real data. **No claim about retrieval quality follows from it.**

---

## Known limitations

**No retrieval path yet.** The model encodes users and items but cannot rank a
catalogue. `recommend`, `recommend_batch` and `score` are not implemented, so it
is not yet a `CandidateGenerator` and cannot enter fusion. The Phase 5 gate
still fails its `issubclass(..., CandidateGenerator)` check — correctly.

**Missing-modality handling is fixture-tested only.** PixelRec50K has 100%
coverage of both modalities, so there are no text-only, image-only or
no-modality items in the real corpus. The code path is implemented and tested
because it must be correct for other corpora; the real-data result is *not
exercised*, not *passing*.

**No hyperparameters have been selected.** Everything in `two_tower.yaml` is a
development default. Nothing has been chosen on validation.

**Cold-item recall is unmeasured.** The cold *encoding* guarantee is verified.
Whether those encodings actually retrieve the right cold items is a benchmark
that needs full-catalogue retrieval, which does not exist yet.

**Attention pooling not implemented**, per the milestone's instruction to
measure simpler pooling first.

---

## Remaining Phase 5 work

1. **Full-catalogue retrieval** — `encode_all_items`, `recommend`,
   `recommend_batch`, `score`; the `CandidateGenerator` wrapper
2. **FAISS index** over two-tower embeddings, verified against brute force,
   with cold items present in the catalogue
3. **Cold-item Recall@K benchmark** — the number this milestone makes possible
   but does not produce
4. **Modality ablations** — the seven combinations plus six leave-one-out variants
5. **Rolling-fold selection**, lock, multi-seed, final test
6. **Five-source fusion** versus the existing four, with overlap diagnostics
7. `scripts/compare_multimodal_retrievers.py`, `phase5_selected.yaml`,
   `configs/features/pixelrec_published.yaml`
8. CI `multimodal-retrieval` job; remaining Phase 5 docs and `phase_05_report.md`
9. Phase 4 closeout §3.4 (rolling folds) and §3.5 (paired bootstrap)

---

## Recommended next milestone

**Full-catalogue retrieval and the cold-item benchmark**, in that order.

Everything else in Phase 5 depends on being able to rank the catalogue: the
ablations need a metric to compare, fusion needs a `CandidateGenerator`, and the
selection stage needs something to select on. The cold-item Recall@K that
follows is the number that decides whether Phase 5 achieved its purpose — and
this milestone has made it a measurement rather than a hope.

---

## Commands executed

```bash
ruff format src tests scripts && ruff check src tests scripts   # clean
mypy --strict src                                               # 106 files, clean
pytest tests/unit/models/two_tower                              # 108 passed
pytest tests/integration/test_two_tower_training.py             # 11 passed
pytest tests/                                                   # 1,422 passed
python scripts/train.py --model two_tower --stage development \
  --version phase5-two-tower-dev --subset-users 1000 --epochs 3 --device cpu
python scripts/validate_phase5.py                               # still exit 1, as expected
```
