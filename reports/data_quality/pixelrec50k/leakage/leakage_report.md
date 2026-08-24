# Leakage report - pixelrec50k

**PASSED** — 12/13 checks passed · 0 critical failures · 1 warnings

A critical failure aborts the pipeline: leakage makes offline metrics *better*, so it cannot be caught by looking at results.

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
