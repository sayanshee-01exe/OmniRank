# Processed data profile - pixelrec50k

## Raw → processed

| Entity | Raw | Processed | Removed |
|---|---:|---:|---:|
| Users | 50,000 | 50,000 | 0 |
| Items | 82,865 | 69,347 | 13,518 |
| Interactions | 989,494 | 975,976 | 13,518 |

Sparsity: **0.999718524** · Duplicates removed: **0**

## Splits

| Split | Rows | Users | Items |
|---|---:|---:|---:|
| train | 875,976 | 50,000 | 68,577 |
| validation | 50,000 | 50,000 | 23,760 |
| test | 50,000 | 50,000 | 20,770 |

Strategy: `per_user_leave_last_n` · ordering field: `timestamp` · eligible users: 50,000 · ineligible: 0

## Sequential examples

| Split | Examples | Users | Skipped (short history) | Truncated | Mean length |
|---|---:|---:|---:|---:|---:|
| train | 775,977 | 49,999 | 99,999 | 13,418 | 18.03 |
| validation | 49,999 | 49,999 | 1 | 327 | 17.25 |
| test | 50,000 | 50,000 | 0 | 337 | 18.24 |

## Multimodal feature coverage

| Modality | Available | Dimension | Matched | Coverage |
|---|---|---:|---:|---:|
| text | **no** | 0 | 0 | 0.0000 |
| image | **no** | 0 | 0 | 0.0000 |

## Evaluation slices

| Slice | Entity | Size |
|---|---|---:|
| `users_activity_1-3` | user | 25 |
| `users_activity_4-10` | user | 19,901 |
| `users_activity_11-30` | user | 24,028 |
| `users_activity_31+` | user | 6,046 |
| `items_long_tail` | item | 40,778 |
| `items_head` | item | 27,799 |
| `items_cold_start` | item | 770 |
| `users_cold_start` | user | 0 |
| `items_missing_text_features` | item | 69,347 |
| `items_missing_image_features` | item | 69,347 |
| `items_missing_both_modalities` | item | 69,347 |
| `items_both_modalities` | item | 0 |

## Leakage checks

**PASSED** — 12/13 checks passed, 0 critical failures, 1 warnings.

| Check | Severity | Result | Detail |
|---|---|---|---|
| `L01_no_duplicate_interaction_across_splits` | critical | pass | 0 interactions span multiple splits |
| `L02_train_precedes_validation` | critical | pass | 0 users have training events at or after a validation target |
| `L03_validation_precedes_test` | critical | pass | 0 users have validation events at or after a test target |
| `L04_train_precedes_test` | critical | pass | 0 users have training events at or after a test target |
| `L10_mapping_consistent_across_splits` | critical | pass | 0 unmapped users, 0 unmapped items, 0 sentinel ids |
| `L11_cold_items_in_held_out` | warning | warn | 770 items are evaluated but never seen in training (genuine cold start) |
| `L05_train_sequence_history_is_past` | critical | pass | 0 sequences include a non-past event; 0 include the target in the input |
| `L05_validation_sequence_history_is_past` | critical | pass | 0 sequences include a non-past event; 0 include the target in the input |
| `L05_test_sequence_history_is_past` | critical | pass | 0 sequences include a non-past event; 0 include the target in the input |
| `L07_graph_training_only` | critical | pass | 0 edges exist only in a held-out split |
| `L08_popularity_training_only` | critical | pass | 0 items have counts that do not match a training-only recount |
| `L09_user_statistics_training_only` | critical | pass | 0 users have counts that do not match a training-only recount |
| `L12_no_labels_in_feature_tables` | critical | pass | 0 feature tables contain forbidden columns |
