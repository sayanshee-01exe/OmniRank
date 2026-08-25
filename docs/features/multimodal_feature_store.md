# Multimodal feature store

[`src/omnirank/features/multimodal_store.py`](../../src/omnirank/features/multimodal_store.py)

## What it holds

Two aligned matrices, one per modality, indexed by internal item id:

| | |
|---|---|
| Shape | 69,347 × 1024, float32 |
| Size | 271 MB per modality |
| Storage | memory-mapped `.npy` |
| Coverage | 100% text, 100% image |

Both would fit in memory at this size. They are mapped anyway for two reasons
that outlast this catalogue: training reads them in batches, where mapping
avoids a second copy per worker, and the same code has to keep working when the
catalogue is ten times larger without anyone revisiting the decision.

## Identity is checked, not assumed

A feature matrix is a list of vectors with no intrinsic connection to any item
id. Loaded against a different mapping it does not fail — it silently describes
the wrong items, and every recommendation built on it is confidently wrong in a
way no metric reveals.

So the manifest carries the item mapping checksum, the feature version and the
dimensions, and `require_compatible()` refuses a mismatch. This is
[ADR-006](../adr/ADR-006-versioned-artifacts.md) applied to features rather than
to indexes.

`manifest_checksum()` hashes only the identity-bearing fields — mapping,
version, dimensions, source checksums, match counts — and deliberately not the
creation timestamp or runtime measurements. Re-running alignment on the same
inputs must produce the same identity.

## Absence is a state, not an error

An item may have text, image, both, or neither. `get_batch` returns zero-filled
rows for missing modalities **and** a per-item mask:

```python
batch.text  # (rows, dim) — zeros where absent
batch.text_mask  # (rows,)     — False where absent
```

The zeros are a placeholder the model must gate on the mask. A model that reads
the vectors and ignores the masks is training on fabricated content, which is
the failure this structure exists to make hard to reach by accident. Rows the
mask says are absent are explicitly zeroed rather than left as whatever
alignment happened to leave behind.

`content_representable` reports items carrying at least one modality — the
catalogue a content model can encode. An item with neither cannot be represented
from content, and saying so is what keeps a genuinely unreachable item from
being counted as a retrieval failure.

## Order is preserved

`get_batch` returns rows in the requested order, not sorted. Callers align these
rows against their own label tensors positionally, so reordering would silently
pair the wrong content with the wrong item.

## Reads are read-only

Matrices are opened with `mmap_mode="r"`. A bug that writes to the store fails
loudly instead of corrupting a 271 MB file that every later run depends on.

## Alignment cost

Measured on the real source files:

| | text | image |
|---|---:|---:|
| Source size | 8.65 GB | 8.60 GB |
| Runtime | 168 s | 141 s |
| Peak memory | 320 MB | 320 MB |
| Items matched | 69,347 / 69,347 | 69,347 / 69,347 |
| NaN / inf / duplicates | 0 | 0 |

Roughly 17 GB is read to keep 570 MB, because PixelRec publishes vectors for all
408,374 full-corpus items and PixelRec50K needs 17% of them. Nothing is ever
fully parsed: the source is walked incrementally and only wanted ids are
materialised, so peak memory is the output, not the input.

## float32, measured rather than assumed

float16 would halve the store. It was compared against float32 before being
rejected:

| | text | image |
|---|---:|---:|
| Max absolute error | 1.2e-04 | 9.4e-04 |
| Max **relative** error | 1.0 | 1.0 |
| Max dot-product error | 6.3e-05 | 7.5e-04 |

A max relative error of 1.0 means some values round entirely to zero, and a
dot-product error of 7.5e-04 is enough to reorder adjacent candidates. The
saving is 284 MB on a 16 GB machine. float32 stays; the measurement is recorded
in the manifest rather than the decision being asserted.

## Related

- [Published vectors](pixelrec_published_vectors.md)
- [Two-tower model core](../models/multimodal_two_tower_core.md)
