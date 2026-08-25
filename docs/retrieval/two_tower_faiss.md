# Two-tower FAISS index

[`src/omnirank/retrieval/two_tower_index.py`](../../src/omnirank/retrieval/two_tower_index.py)

## What it adds over the generic index

The building, searching, bounded exclusion search and persistence all come from
[`FaissVectorIndex`](faiss_index.md) unchanged. Duplicating that for one model
would mean two implementations to keep exact.

What this module adds is **a third way to be wrong that a collaborative index
does not have**.

A LightGCN index is wrong if paired with the wrong model or the wrong item
mapping. A two-tower index inherits both of those *and* one more: its vectors
are derived from a feature store, so a store holding different content — same
items, same mapping, same dimensions — produces a different index that nothing
downstream would notice. Feature version and feature-manifest checksum therefore
travel with the index, and a mismatch is refused.

It also records **warm and cold counts**. "The index contains cold items" is the
claim Phase 5 rests on, and an index that quietly contained none would still
answer every query with plausible results.

## Exactness, and why bit-exact is the wrong bar

Every flat index is checked against `brute_force_top_k`. That comparison is the
only thing separating "fast" from "fast and wrong": an index built with the
wrong metric, over a transposed matrix, or from a different model's embeddings
still returns k neighbours with plausible scores, and nothing raises.

But **demanding identical orderings fails on correct indexes at catalogue
scale.** A 128-dimensional dot product accumulates float32 rounding error of
roughly 1e-7, and FAISS and numpy do not sum in the same order. Two items whose
true scores differ by less than that can be ordered either way.

Measured on the real 69,347-item catalogue: 254 of 256 queries matched exactly,
set overlap 0.9996, **max score difference 4.2e-07**. The two disagreements were
adjacent items separated by less than float32 could resolve.

So verification distinguishes the two cases:

```text
exact_order_agreement        positions that matched bit-for-bit
order_agreement_within_ties  positions that matched, or differed only where
                             the two scores are numerically indistinguishable
unexplained_disagreements    positions that differed where they are not
matches_brute_force          unexplained_disagreements == 0
```

A disagreement counts against the index only when the scores at that position
are *distinguishable* — anything else is arithmetic, not a defect. The tolerance
is explicit (`TIE_TOLERANCE = 1e-5`) rather than implied by a rounded assertion.

On the final artifact this reports 1.0 exact agreement and 0 unexplained
disagreements.

## Metric and normalisation

`IndexFlatIP` with inner product, because the towers are L2-normalised — which
makes a dot product a cosine similarity. The normalisation rule is recorded in
the index metadata and checked, because building under one convention and
querying under the other returns confident nonsense rather than an error.

Approximate index types are available but **not used**: Phase 4 measured HNSW at
0.67 recall and IVF at 0.26 against exact, for 39 µs of savings on a catalogue
this size. Exact is the default until the catalogue makes it too slow.

## Artifacts

```text
artifacts/embeddings/two_tower/<version>/
├── item_embeddings.npy        # float32, memory-mappable
├── item_index.parquet         # the catalogue, positionally aligned
├── catalogue_manifest.json    # warm/cold/excluded counts + checksum
├── embedding_manifest.json    # dimensions, checksums, identity
└── index_manifest.json        # index metadata + exactness report
```

The embedding matrix and the item table are **positionally aligned**, so the
catalogue checksum is verified on load. A reordered table would pair every
embedding with the wrong item and produce a fully-formed, entirely wrong result;
the checksum turns that into a refusal.

## Related

- [Generic FAISS index](faiss_index.md) · [Five-source fusion](five_source_fusion.md)
- [Cold-item evaluation](../evaluation/cold_item_evaluation.md)
