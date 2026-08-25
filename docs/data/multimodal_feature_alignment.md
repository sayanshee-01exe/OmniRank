# Multimodal feature alignment

Implemented in [`src/omnirank/data/pixelrec/features.py`](../../src/omnirank/data/pixelrec/features.py).

Phase 2 **validates and aligns** the vectors PixelRec published. It does not
compute features: no CLIP, no SentenceTransformers, no fine-tuning.

## What PixelRec publishes

Measured by streaming the first record of each file on 2026-08-24 — not quoted
from documentation.

| File | Size | Shape | Coverage |
|---|---:|---|---|
| `text_feature.json` | **9,290,203,646 B (8.65 GiB)** | `{item_id: [1024 floats]}` | all 408,374 full-PixelRec items |
| `image_feature.json` | **9,235,894,623 B (8.60 GiB)** | `{item_id: [1024 floats]}` | all 408,374 full-PixelRec items |

Both are flat JSON objects. The source does not document which encoders produced
them, so `text_encoder` and `image_encoder` are `null` in the config — a guess
would end up in artifact metadata and be treated as fact.

## Why they are not downloaded by default

17.3 GB must be read to keep vectors for the 82,865 items PixelRec50K uses —
**20% of the payload**. On a 16 GB laptop that is a poor default, and the
pipeline is fully useful without them: collaborative, graph, and sequential
datasets do not depend on them.

```bash
python scripts/download_pixelrec50k.py --with-features   # opt in
```

## Absence is a first-class state

When the files are missing, the pipeline still emits schema-correct index tables
with `has_*_feature = False` throughout and coverage reported as **0.0**.

This matters more than it looks. The alternative — omitting the tables, or
filling them with zeros — would make "we have no text features" indistinguishable
from "every text feature is the zero vector". A missing modality must degrade,
never crash, and never be silently reported as present
([ADR-003](../adr/ADR-003-offline-embeddings.md)).

The index schema is **identical** whether or not vectors exist, so Phase 4 code
reads one shape either way. A test asserts that.

## Streaming, not parsing

`json.load` on an 8.65 GiB file would need tens of gigabytes of Python objects.
`stream_feature_vectors` walks the file incrementally with a small state machine
and emits only the wanted ids, so peak memory is proportional to the *output*,
not the input.

The parser exploits the format's one guarantee — values are arrays of numbers,
never nested — so the first `]` closes a record and no bracket matching is
needed. It raises a clear `DataSourceError` if that assumption is violated, if
the file is not a JSON object, or if a single record exceeds an 8 MiB bound.

Unit tests cover records spanning many read blocks, block sizes from 16 bytes to
1 MB, whitespace, empty objects, and malformed input — all against synthetic
files of the same shape, so the 8.65 GiB download is not needed to test it.

## Validation performed

Per modality, recorded in the processed profile and the manifest:

| Check | Recorded as |
|---|---|
| File exists | `available` |
| Vector width consistent | `dimension`, `dimension_mismatches` |
| Ids align with the item mapping | `rows_matched`, `items_missing`, `coverage` |
| No NaN values | `rows_with_nan` |
| No infinite values | `rows_with_inf` |
| No duplicate ids | `duplicate_ids` |
| Source row count | `rows_in_source` |
| dtype | `dtype` (`float32`) |
| Normalisation status | `normalized` (`false`) |
| Source encoder | `encoder` (`null` — undocumented) |

Rows failing a check are **excluded and counted**, never silently coerced.

### Normalisation

`normalize_precomputed_features: false`. The source publishes raw encoder
outputs and documents no normalisation, so none is claimed and none is applied.
Renormalising would silently change every vector and make results
incomparable with the published PixelRec baselines.

This also sidesteps a leakage question: fitting a scaler across splits would be
a statistic learned from validation and test data. Fixed pretrained
representations require no fitting at all.

## Storage format

**NumPy `.npy`, float32, one row per `internal_item_id`.**

At 82,865 × 1024 that is 339 MiB — small enough to keep, large enough that it
must be memory-mappable rather than loaded:

```python
matrix = np.load("features/text_features.npy", mmap_mode="r")
vector = matrix[internal_item_id]  # row index == internal id
```

Alternatives considered:

| Format | Rejected because |
|---|---|
| Parquet fixed-size lists | Needs a decode step on every read; no zero-copy |
| PyTorch `.pt` | Makes torch a dependency of the data layer |
| Raw JSON | The 8.65 GiB problem, unchanged |

Row index equals `internal_item_id` exactly, so no lookup table sits between the
matrix and the mapping — one fewer thing that can drift out of sync. Rows for
items with no vector remain zero, and `has_*_feature = False` is the
authoritative signal, never a zero-check.

## Missing-modality slices

Four slices partition the catalogue by what it actually has:
`items_missing_text_features`, `items_missing_image_features`,
`items_missing_both_modalities`, `items_both_modalities`. With vectors absent,
all 69,347 items land in `items_missing_both_modalities` and
`items_both_modalities` is empty — which is the truthful picture.

## Current state

| Modality | Available | Dimension | Matched | Coverage |
|---|---|---:|---:|---:|
| text | **no** | — | 0 | **0.0** |
| image | **no** | — | 0 | **0.0** |

Recorded in `known_limitations` in the dataset manifest.
