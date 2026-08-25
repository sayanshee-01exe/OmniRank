# PixelRec published multimodal vectors

What the item tower actually consumes, where it came from, and what is *not*
known about it.

## What these files are

PixelRec publishes two per-item matrices alongside the interaction log:

| Modality | Dimension | Items matched | Coverage | Source SHA-256 (first 16) |
| --- | ---: | ---: | ---: | --- |
| text | 1024 | 69,347 | 1.000 | `d3376b5ec9593fde` |
| image | 1024 | 69,347 | 1.000 | `b74fb95313afe53b` |

Both cover the full 69,347-item catalogue after k-core filtering. There is no
item in the catalogue with only one modality, and none with neither.

## What is not known

**The encoders are undocumented.** PixelRec ships the vectors; it does not
publish which model produced them, at which checkpoint, with which
preprocessing, or in which normalisation. Nothing in the release identifies
them.

The configuration therefore records:

```yaml
text:
  encoder_identity: unknown
image:
  encoder_identity: unknown
```

`unknown` is the accurate value. A 1024-dimensional text vector is *consistent*
with several well-known encoders, and it would be easy to write `clip` or
`bert` and have nobody notice. That would be a fabricated provenance claim, and
every downstream comparison would inherit it — including any future statement
about how OmniRank's representations compare to a published baseline, which
would then be comparing against a guess.

The practical consequences of not knowing:

- **No text/image alignment can be assumed.** The two matrices are not
  presumed to share a space, so the item tower projects each modality
  separately before fusing. A design that assumed a shared CLIP-style space
  would silently depend on an unverified property.
- **No normalisation is claimed.** `normalization_applied: false` in the
  manifest is a statement about what this pipeline did, not about what the
  vectors are. The model L2-normalises its own outputs; it does not assume its
  inputs arrived normalised.
- **Re-encoding is not currently possible.** Reproducing these vectors from the
  raw media would require knowing the encoder. Should that become necessary,
  it is a new feature version, not an in-place fix.

## Identity and drift

Each matrix is stored as a memory-mapped `float32` array beside an index
mapping `internal_item_id` to row. The manifest records:

- `item_mapping_checksum` — the id mapping the rows are aligned to. A model
  built against a different mapping resolves every row to the wrong item, and
  would still return plausible-looking recommendations, so this is checked at
  load rather than trusted.
- `source_sha256` per modality — the bytes the alignment consumed.
- `feature_version` — bumped when the *definition* changes, which is the
  training/serving skew tripwire.

`MultimodalFeatureStore.require_compatible` refuses a mismatch. See
[multimodal_feature_store.md](multimodal_feature_store.md) for the store, and
[../data/multimodal_feature_alignment.md](../data/multimodal_feature_alignment.md)
for how alignment produced these files.

## Why float32

float16 was measured, not assumed. Over the real matrices the maximum relative
element error was 1.0 and the dot-product error 7.5e-4 — against retrieval
score gaps that are frequently smaller than that. Halving the memory would have
changed which items came back. Rejected on the measurement.

## Licence

> This dataset is provided by the Westlake Representation Learning Lab
> exclusively for non-commercial research and educational purposes... No rights
> are granted with respect to copying, modifying, publishing, distributing, or
> commercializing the dataset.

The aligned matrices are derived data. They are git-ignored, they are not
redistributed, and no per-item PixelRec-derived rows are committed. CI never
downloads them; tests that need features build synthetic ones.
