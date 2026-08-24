# Phase 2 report — PixelRec50K data engineering

**Date:** 2026-08-24 · **Pipeline version:** 2.0.0 · **Schema version:** 2 · **Status:** complete

---

## 1. Repository state before Phase 2

Phase 1's foundation, intact and unmodified: 124 files, 359 passing tests, ruff
and mypy-strict clean. **Not a git repository. DVC not configured.** `data/`,
`artifacts/`, and `reports/` held only `.gitkeep` files.

Reused rather than rebuilt: `core/config.py` (YAML + env + validation),
`core/logging.py`, `core/exceptions.py`, `artifacts/` (metadata + registry),
`data/validation.py`, `data/id_mapping.py`, `data/splitting.py::check_split_integrity`,
and the `DatasetLoader` / `Preprocessor` / `Splitter` protocols. `pandas`/`pyarrow`
existed as an unused `[data]` extra and are now installed and exercised.

### Phase 1 contracts that had to change

PixelRec cannot satisfy the Phase 1 schemas. All five changes are **relaxations
or additions**, and all 359 Phase 1 tests still pass unmodified.

| Change | Why |
|---|---|
| `EventType` += `INTERACTION` | PixelRec has one unlabelled implicit signal; no existing member is honest |
| `Item.title` → optional | 192 real items have none |
| `Item.created_at` → optional | No publication date exists; fabricating one is forbidden |
| `User.created_at` → optional | There is no user table at all |
| `SplittingConfig` += `per_user_leave_last_n` + N-parameters | Phase 1 had only fraction strategies |

One design bug was found and fixed while wiring the profile: domain overlays
**deep-merged** with the e-commerce vocabulary, so PixelRec would have accepted
`click`/`purchase` events it does not have. `load_config(data_profile=…)` now
*replaces* the domain profile instead of merging onto it.

## 2–4. Source, version, licence

| | |
|---|---|
| Repository | <https://github.com/westlake-repl/PixelRec> |
| PixelRec50K folder | [Google Drive `1bQPgM…`](https://drive.google.com/drive/folders/1bQPgM-6yAnzcD0jKBoUUheA9LL5xnCHG) |
| Dataset version | `v1` — `interaction.csv` (2023-09-26), `item_info.csv` (2024-11-18) |
| Licence | **Non-commercial research and education only.** No rights to copy, modify, publish, distribute, or commercialise. Secondary downloads of modified copies explicitly prohibited. |

Consequences enforced, not just noted: `data/` is git-ignored, no PixelRec data
is committed, test fixtures are **generated** rather than sampled, and the
processed outputs are equally non-redistributable derivatives.

## 5–6. Raw files and schemas

| File | Bytes | Rows | SHA-256 |
|---|---:|---:|---|
| `interaction.csv` | 28,124,439 | 989,494 | `638b53ec100f760c…` |
| `item_info.csv` | 24,973,166 | 82,865 | `a073c2c65900f215…` |

`interaction.csv` → `item_id, user_id, timestamp`.
`item_info.csv` → `item_id`, 7 engagement counters, `title, tag, description`.

Measured, not assumed: 50,000 users · 82,865 items · **exactly the official
figures** · timestamps are **real Unix epoch seconds** (2012-02-03 → 2022-06-24
UTC, 3,793 days) · **zero** duplicate rows · **zero** per-user timestamp ties ·
**perfect** 1:1 item referential integrity · nulls: description 23.84%, title
0.23% (192), and 5 rows missing all counters · per-item minimum 1, with **13,518
singleton items**.

Full detail: [`docs/data/pixelrec50k_raw_schema.md`](../data/pixelrec50k_raw_schema.md).

## 7. Source-to-canonical mapping

[`docs/data/source_to_canonical_mapping.md`](../data/source_to_canonical_mapping.md).
Three decisions carry weight:

**`event_type = "interaction"`.** PixelRec measured engagement and nothing finer.
Naming it `click` would assert intent the source never recorded, and the domain
profile's per-event weights would inherit that fiction.

**Engagement counters are metadata, never features.** They describe the *whole
platform's* lifetime totals — not this dataset's 50,000 users — and carry no
timestamp, so they cannot be point-in-time bounded. Preserved in
`source_metadata`; structurally excluded from every feature table.

**`interaction_id` is a derived surrogate** (`pr50k-<source_row_id>`):
deterministic, traceable to a source line, asserting nothing about the world.
Categorically different from inventing a `price`.

**Never fabricated:** item price, brand, rating, inventory, `created_at`; user
demographics or signup date; session ids; any second event type. Tests assert
these column names appear in no output.

## 8–9. Cleaning rules and removals

Ten interaction rules and two item rules, in fixed order, each with a stable
reason code and every rejection written to `rejected_records.parquet` with its
source row. Every step asserts `input == output + removed` **and** that recorded
reasons account for every removal.

| Step | Input | Output | Removed |
|---|---:|---:|---:|
| `clean_items` | 82,865 | 82,865 | **0** |
| `clean_interactions` | 989,494 | 989,494 | **0** |

**Zero rejections** — PixelRec50K is genuinely clean. The rules are exercised
instead by unit tests that inject each violation and assert the matching code
fires. `rejected_records.parquet` is written empty-but-typed regardless.

## 10–13. Processed dataset

| | Raw | Processed | Removed |
|---|---:|---:|---:|
| Users | 50,000 | **50,000** | 0 |
| Items | 82,865 | **69,347** | 13,518 |
| Interactions | 989,494 | **975,976** | 13,518 |

**Sparsity: 0.999718524** (raw 0.999761179).
History length after processing: min 3 · median 14 · mean 19.52 · max 416.

## 14. Interaction ordering

Real timestamps, so nothing is invented. Sort key
`(external_user_id, timestamp, source_row_id)`; `interaction_order` is the
0-based rank **within each user's own history**, defined once so the splitter,
the sequence builder, and every leakage check share one definition rather than
three that can disagree. The `source_row_id` tiebreak never fires on this
dataset (0 ties) but is applied unconditionally.

## 15. Filtering

`min_interactions_per_user: 3`, `min_interactions_per_item: 2`, iterative.
**Converged in 1 iteration**, removing 13,518 singleton items and their 13,518
interactions. No user fell below the threshold, so the loop terminated
immediately. Pre-filtering cold-start snapshot captured before removal.

## 16. Mapping strategy

Dense contiguous integers from **0**, assigned over **sorted** external ids —
so source row order cannot change the mapping. Fitted on the full
post-filtering population (an identifier registry, not a learned statistic).
Written as Parquet with reverse lookups plus `mapping_metadata.json` carrying
versions, counts, content checksums, Phase 1 `IdMapping` fingerprints, and
explicit unknown-entity and reassignment policies. Unknown ids resolve to `-1`.

## 17–21. Splits and derived datasets

**Per-user leave-last-N**, ordering field `timestamp`, all 50,000 users eligible.

| Split | Rows | Users | Items |
|---|---:|---:|---:|
| train | **875,976** | 50,000 | 68,577 |
| validation | **50,000** | 50,000 | 23,760 |
| test | **50,000** | 50,000 | 20,770 |

Sequential examples:

| Split | Examples | Users | Skipped (short history) | Truncated | Mean length |
|---|---:|---:|---:|---:|---:|
| train | **775,977** | 49,999 | 99,999 | 13,418 | 18.03 |
| validation | **49,999** | 49,999 | 1 | 327 | 17.25 |
| test | **50,000** | 50,000 | 0 | 337 | 18.24 |

Graph edges are built from training rows only, with **binary** weights (the
source carries no intensity) and the raw repeat count preserved separately.

## 22–24. Multimodal features

PixelRec publishes `text_feature.json` (**8.65 GiB**) and `image_feature.json`
(**8.60 GiB**), both `{item_id: [1024 floats]}` covering all 408,374
full-PixelRec items — 17.3 GB to obtain vectors for the 20% PixelRec50K uses.
Structure and sizes measured by streaming the first record of each.

**Not downloaded by default.** Current coverage:

| Modality | Available | Dimension | Matched | Coverage |
|---|---|---:|---:|---:|
| text | **no** | — | 0 | **0.0** |
| image | **no** | — | 0 | **0.0** |

Absence is a first-class state: schema-identical index tables are still emitted
with `has_*_feature = False` and coverage reported as 0.0, never assumed. A
streaming parser handles the 8.65 GiB file in bounded memory and is unit-tested
against synthetic files of the same shape, so the download is not needed to test
it. Storage format is memory-mappable `.npy` float32, row index = internal id —
chosen and justified in [`multimodal_feature_alignment.md`](../data/multimodal_feature_alignment.md).

## 25–27. Evaluation slices

| Slice | Size | | Slice | Size |
|---|---:|---|---|---:|
| `users_activity_1-3` | 25 | | `items_cold_start` | **770** |
| `users_activity_4-10` | 19,901 | | `users_cold_start` | **0** |
| `users_activity_11-30` | 24,028 | | `items_missing_text_features` | 69,347 |
| `users_activity_31+` | 6,046 | | `items_missing_image_features` | 69,347 |
| `items_head` | 27,799 | | `items_missing_both_modalities` | 69,347 |
| `items_long_tail` | **40,778** | | `items_both_modalities` | 0 |

Long tail = items outside the head accounting for 80% of training interactions
(threshold recorded in the slice manifest). `users_cold_start` is empty **by
construction** under leave-last-N and is emitted empty rather than fabricated.

Training-only statistics: `item_training_popularity.parquet` (68,577 rows) and
`user_training_statistics.parquet` (50,000 rows), both independently verified.

## 28. Leakage results

```text
PASSED — 12/13 checks passed · 0 critical failures · 1 warning
```

| Check | Result |
|---|---|
| L01 no interaction in two splits | pass |
| L02 train precedes validation | pass |
| L03 validation precedes test | pass |
| L04 train precedes test | pass |
| L05 sequence histories strictly past × 3 splits | pass |
| L07 graph edges training-only | pass |
| L08 popularity = training-only recount | pass |
| L09 user statistics = training-only recount | pass |
| L10 one mapping across splits | pass |
| L12 no labels in feature tables | pass |
| L11 cold items in held-out splits | **warning** — 770, expected |

A critical failure aborts the pipeline with exit code 3.

## 29–30. Outputs and manifest

**34 output files**, each with row count, bytes, and SHA-256 in the manifest at
`data/processed/pixelrec50k/dataset_manifest.json`. Sizes: 38 MB processed,
35 MB interim, ~1 MB mappings, 1.4 MB reports. Full layout in
[`processed_schemas.md`](../data/processed_schemas.md).

## 31. Tests added

**319 new tests** (359 → **678 total**), all offline, CPU-only, no database, no
network, no pretrained weights.

| File | Tests | Covers |
|---|---:|---|
| `unit/data/test_loaders.py` | 18 | File checks, header assertions, chunk-invariance, subsetting |
| `unit/data/test_canonical.py` | 24 | Mapping, honest event type, non-fabrication, Pydantic conformance |
| `unit/data/test_cleaning.py` | 29 | All 12 rules, reconciliation, rejected-record trail |
| `unit/data/test_filtering.py` | 14 | Convergence, cascading, snapshot, determinism |
| `unit/data/test_mapping.py` | 24 | Determinism, reversibility, contiguity, persistence |
| `unit/data/test_splitters.py` | 21 | Assignment, ordering invariants, eligibility, determinism |
| `unit/data/test_sequences.py` | 16 | Target exclusion, truncation direction, minimum length |
| `unit/data/test_statistics.py` | 15 | Training-only counts verified by manual recount |
| `unit/data/test_slices.py` | 11 | Partitioning, bucket contiguity, cold-start rules |
| `unit/data/test_features.py` | 26 | Streaming at 4 block sizes, NaN/inf/duplicates, honest absence |
| `unit/data/test_manifest.py` | 21 | Required fields, checksums, byte-identical writes |
| `unit/data/test_leakage.py` | 30 | **Each check verified by injecting the leak it catches** |
| `integration/test_pipeline.py` | 46 | Full pipeline over a synthetic fixture, twice, byte-comparing |

Fixture: `tests/fixtures/pixelrec.py` generates PixelRec-shaped CSVs with
deliberately seeded edge cases (duplicates, dangling references, missing titles,
singleton items). **Generated, never sampled** — the licence forbids
redistribution and the tests must run on a fresh checkout.

## 32–36. Commands, results, real-data validation

```bash
uv pip install -e ".[data,dev]"
ruff format src tests scripts && ruff check src tests scripts
mypy
pytest
python scripts/prepare_data.py --config configs/data/pixelrec50k.yaml --overwrite
```

| Gate | Result |
|---|---|
| `ruff format --check` | **110 files already formatted** |
| `ruff check` | **All checks passed** |
| `mypy` (strict) | **Success — no issues in 110 source files** |
| `pytest` | **678 passed** in ~6 s |

### Real-data validation

**The full PixelRec50K dataset was processed end to end — not a subset.**
989,494 interactions, 50,000 users, 82,865 items, in **15.1 seconds**, peak
memory ~1 GB. All 34 outputs written, all 13 leakage checks run, exit code 0.

The one thing *not* validated against real data is multimodal feature alignment,
because the 17.3 GB of vectors was not downloaded. That path is covered by unit
tests against synthetic files of the identical shape, and by an integration test
that runs the whole pipeline with a real 1024-d feature file present.

Full-dataset command with features:

```bash
python scripts/download_pixelrec50k.py --with-features    # 17.3 GB
python scripts/prepare_data.py --config configs/data/pixelrec50k.yaml --overwrite
```

### Defects found and fixed during Phase 2

1. **Domain profiles deep-merged instead of replacing** — PixelRec inherited the
   e-commerce event vocabulary and would have accepted `click`/`purchase`.
2. **`--subset-users` cascades filtering to zero** — sampling users destroys item
   density. Correct behaviour; the error message now names the cause.
3. Three test-harness defects (capsys stream binding, a `strict=True` zip over
   offset sequences, a dimension-mismatch assertion working as designed).

## 37. Known limitations

1. **Multimodal coverage is 0.0.** 17.3 GB not downloaded; alignment validated
   on synthetic and fixture data only.
2. **`--subset-users` is unusable with the shipped filtering thresholds.**
   Documented, with an actionable error.
3. **No cover images processed.** `cover.7z` is not downloaded; Phase 2 runs no
   image model by design.
4. **No new-user cold-start slice.** Leave-last-N cannot produce one; fabricating
   it was refused.
5. **`git_commit` is `null`** — still not a git repository.
6. **DVC not integrated** — not configured in Phase 1, and there is no second
   machine to share with. Manifest checksums provide the same guarantee.
7. **Engagement counters unused.** Preserved but excluded from features; a
   point-in-time-safe use would need a timestamped snapshot the source lacks.
8. **Frames, not records, on the bulk path.** The Pydantic contracts remain the
   schema authority and conformance is tested, but a million rows are not
   validated one model at a time.
9. **Only PixelRec50K has a loader.** The e-commerce profile has no dataset.

## 38. Technical debt

- `pipeline.py` `_write_outputs` takes 17 arguments — cohesive but at the edge
  of what is comfortable; a writer object would be cleaner.
- Item metadata is loaded whole rather than chunked (60 MB; fine at this scale,
  not at Pixel8M).
- The streaming JSON parser assumes flat numeric arrays. Correct for this source,
  guarded by explicit errors, but not a general JSON parser.
- pandas-stubs forced disabling six mypy error codes **for tests only**; `src`
  remains fully strict.
- `tests/fixtures/pixelrec.py` duplicates the header constants from the loader.

## 39. Deferred

Phase 3+: baselines, evaluation metrics, LightGCN, SASRec, two-tower,
multimodal training, ranking, serving. Out of scope entirely: Kubernetes, Kafka,
streaming, online learning, RL.

## 40. Recommended Phase 3 scope

The data foundation is complete; Phase 3 should not touch it.

1. **Evaluation metrics first** — recall, precision, NDCG, MAP, MRR, hit-rate,
   plus coverage/novelty/intra-list-diversity/gini, implementing the Phase 1
   `Evaluator` contract. **Validate against hand-computed fixtures before any
   model is scored**, so a metric bug never masquerades as a model result.
2. **Per-slice reporting from the start**, using the 12 slices already written.
   A model that improves the aggregate while regressing on `items_long_tail` has
   usually just learned popularity better.
3. **Time-decayed popularity baseline** — also the terminal fallback stage, so it
   must exist before anything else serves.
4. **Implicit matrix factorization** (ALS/BPR) over `graph/train_graph_edges.parquet`,
   measured against popularity under the identical protocol.
5. **Register both as artifacts** with real metrics, the mapping fingerprints
   from `mapping_metadata.json`, and the config hash — making `/ready` return 200
   for the first time.

Expect popularity to score respectably overall and near-zero on
`items_long_tail`, and **both** models to score exactly zero on the 770
`items_cold_start` items. Those gaps are the argument for Phases 4–5.

**Phase 3 exit criterion:** two registered artifacts with real, reproducible
metrics from one protocol, reported per slice, and `/ready` returning 200.
