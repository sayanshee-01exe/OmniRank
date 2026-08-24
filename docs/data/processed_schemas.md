# Processed dataset schemas

Everything `scripts/prepare_data.py` writes. All tables are Parquet with zstd
compression, no index column, fixed column order, and deterministic row order —
so two runs over the same input produce byte-identical files and the manifest
checksums mean something.

## Layout

```text
data/interim/pixelrec50k/
├── canonical_users.parquet
├── canonical_items.parquet
├── canonical_interactions.parquet
└── rejected_records.parquet

data/processed/pixelrec50k/
├── train_interactions.parquet
├── validation_interactions.parquet
├── test_interactions.parquet
├── split_metadata.json
├── dataset_manifest.json
├── collaborative/interactions.parquet
├── graph/train_graph_edges.parquet
├── sequential/{train,validation,test}_sequences.parquet
├── metadata/item_metadata.parquet
├── features/
│   ├── user_training_statistics.parquet
│   ├── item_training_popularity.parquet
│   ├── text_feature_index.parquet
│   ├── image_feature_index.parquet
│   └── {text,image}_features.npy        # only when vectors were downloaded
└── evaluation_slices/
    ├── slice_manifest.json
    └── <slice_name>.parquet             # 12 slices

artifacts/mappings/pixelrec50k/
├── user_id_mapping.parquet
├── item_id_mapping.parquet
└── mapping_metadata.json
```

## Collaborative interactions

`collaborative/interactions.parquet` and the three per-split files share one
schema. Consumers: popularity, matrix factorization, item-item CF, LightGCN.

| Column | Type | Notes |
|---|---|---|
| `internal_user_id` | int64 | 0-based dense index |
| `internal_item_id` | int64 | 0-based dense index |
| `interaction_order` | int64 | 0-based rank within this user's history |
| `event_type` | string | always `interaction` |
| `interaction_weight` | float64 | always `1.0` |
| `split` | string | `train` \| `validation` \| `test` |

Rows: 975,976 (875,976 / 50,000 / 50,000).

## Graph edges

`graph/train_graph_edges.parquet` — **training interactions only**.

| Column | Type | Notes |
|---|---|---|
| `internal_user_id` | int64 | |
| `internal_item_id` | int64 | |
| `edge_weight` | float64 | always `1.0` — **binary** |
| `interaction_order` | int64 | earliest order for this pair |
| `interaction_count` | int64 | raw repeats, preserved separately |

**Weights are binary, not count-based.** PixelRec records one implicit event
type with no intensity, so a count or confidence weight would express a signal
the source does not carry. The raw repeat count is preserved in its own column
so a later weighting scheme has the data it needs without re-deriving it.

## Sequential examples

`sequential/{split}_sequences.parquet` — one row per evaluation target.

| Column | Type | Notes |
|---|---|---|
| `internal_user_id` | int64 | |
| `item_sequence` | list[int64] | History, **oldest first**, strictly before the target |
| `interaction_order_sequence` | list[int64] | Aligned with `item_sequence` |
| `sequence_length` | int64 | `len(item_sequence)` |
| `target_item` | int64 | Never present in `item_sequence` |
| `target_order` | int64 | Strictly greater than every history order |
| `split` | string | |

| Split | Examples | Users | Skipped (short history) | Truncated | Mean length |
|---|---:|---:|---:|---:|---:|
| train | 775,977 | 49,999 | 99,999 | 13,418 | 18.03 |
| validation | 49,999 | 49,999 | 1 | 327 | 17.25 |
| test | 50,000 | 50,000 | 0 | 337 | 18.24 |

Sequences are **variable-length and unpadded**. Padding is a training-time
concern belonging to the collate function that knows the model's expected shape;
baking a padding value into the dataset would fix a decision no model has made.

Truncation keeps the **most recent** `max_length` events (100 configured) — what
a self-attentive model attends to.

Histories cross split boundaries by design: a test target's history legitimately
includes the user's train and validation events, because at prediction time
those had already happened.

## Item metadata

`metadata/item_metadata.parquet` — 69,347 rows, one per post-filtering item.

| Column | Type | Coverage |
|---|---|---|
| `internal_item_id` | int64 | 100% |
| `external_item_id` | string | 100% |
| `title` | string | ~99.8% |
| `description` | string | ~76% |
| `category` | string | ~100% (108 distinct) |
| `image_reference` | string | 100% (`<item_id>.jpg`) |
| `text_feature_reference` | string | 100% (the item id) |
| `image_feature_reference` | string | 100% |
| `source_metadata` | string | JSON of the 7 engagement counters |

No `price`, `brand`, `rating`, `inventory`, or `created_at` — PixelRec has none.

## Training-only feature tables

`features/item_training_popularity.parquet` — 68,577 rows (items present in
training; cold items are omitted, not zero-filled).

| Column | Type | Notes |
|---|---|---|
| `internal_item_id` | int64 | |
| `training_interaction_count` | int64 | Training rows only |
| `training_unique_user_count` | int64 | |
| `training_popularity_rank` | int64 | 1 = most popular, unique |
| `training_popularity_percentile` | float64 | |
| `long_tail_flag` | bool | Outside the 80% head |

`features/user_training_statistics.parquet` — 50,000 rows.

| Column | Type |
|---|---|
| `internal_user_id` | int64 |
| `training_interaction_count` | int64 |
| `unique_training_items` | int64 |
| `first_training_interaction_order` | int64 |
| `last_training_interaction_order` | int64 |
| `first_training_timestamp` | int64 |
| `last_training_timestamp` | int64 |
| `mean_item_popularity` | float64 |
| `sequence_length` | int64 |

Both are verified against an independent training-only recount by leakage checks
L08 and L09.

## Feature index tables

`features/{text,image}_feature_index.parquet` — 69,347 rows, one per item,
**whether or not the vectors were downloaded**.

| Column | Type | Notes |
|---|---|---|
| `internal_item_id` | int64 | |
| `external_item_id` | string | |
| `has_text_feature` / `has_image_feature` | bool | |
| `text_feature_row` / `image_feature_row` | int64 | Row in the `.npy` matrix; `-1` when absent |

Current coverage: **0.0** for both — the 17.3 GB of vectors is not downloaded by
default. See [`multimodal_feature_alignment.md`](multimodal_feature_alignment.md).

## Evaluation slices

`evaluation_slices/<name>.parquet`, each with `slice_name`, `entity_type`,
`entity_id`, plus `slice_manifest.json`. Twelve slices; sizes and definitions in
[`cold_start_evaluation.md`](cold_start_evaluation.md).

## ID mappings

`artifacts/mappings/pixelrec50k/` — see
[`data_versioning.md`](data_versioning.md#id-mappings).

## Interim tables

`canonical_*.parquet` hold the cleaned entities before filtering and splitting.
`rejected_records.parquet` holds every discarded row with
`source_file`, `source_row_identifier`, `entity_type`, `rejection_reason`,
`original_identifier` — empty for PixelRec50K, because nothing was rejected.

## Total size

~38 MB processed, ~35 MB interim, ~1 MB mappings, ~1.4 MB reports. All
git-ignored.
