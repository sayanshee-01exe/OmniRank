# Data directories

Nothing here is committed to git (see `.gitignore`); only the directory
structure is. From Phase 2 these become DVC-tracked stages.

| Directory | Holds | Written by | Read by |
|---|---|---|---|
| `raw/` | Untouched source exports. **Never modified in place** — a raw file you edited is a dataset you can no longer reproduce. | manual download / ingestion | `scripts/prepare_data.py` |
| `interim/` | Intermediate results between preprocessing steps. Safe to delete; regenerable. | `prepare_data.py` | `prepare_data.py` |
| `processed/` | The train/validation/test splits, features, and sequences that training consumes. | `prepare_data.py` | `scripts/train.py` |
| `external/` | Third-party reference data not produced by us (taxonomies, embeddings, lookup tables). | manual | preprocessing |

## Phase 1 status

All four directories are empty. No dataset has been downloaded — that is a
Phase 2 task, and the [non-goals](../docs/phase_reports/phase_01_report.md)
explicitly exclude it.
