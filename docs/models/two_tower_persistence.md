# Two-tower persistence

[`src/omnirank/models/two_tower/persistence.py`](../../src/omnirank/models/two_tower/persistence.py)

## Why identity travels with the weights

A saved two-tower model is meaningless apart from three things it was fitted
against:

| Identity | What breaks without it |
|---|---|
| Item mapping checksum | Every recommended id resolves to a different item |
| Feature version and manifest checksum | Vectors describe different items than the model learned |
| Feature dimensions | The projections receive inputs they were not built for |

None of these fail loudly on load. A model paired with the wrong mapping returns
a full, confident, correctly-shaped list of recommendations — for the wrong
items. This is the argument [ADR-006](../adr/ADR-006-versioned-artifacts.md)
makes for indexes, applied to features.

So `load()` accepts optional `expected_*` arguments and refuses any supplied
identity that disagrees. They are optional because a caller that legitimately
does not know an identity should not be forced to invent one; what is *not*
optional is failing when a supplied one mismatches.

## Layout

```text
artifacts/models/<dataset>/two_tower/<version>/
├── model.pt              # CPU tensors
├── config.json           # TwoTowerConfig
├── metadata.json         # identity, scoring semantics, provenance
└── training_history.json # per-epoch losses and diagnostics
```

## What metadata records

Beyond identity, two categories:

**Scoring semantics** — `embedding_dim`, `normalization`, `temperature`,
`history_pooling`, and the full `modality_schema` including the cold-item
policy. FAISS must be built under the same normalisation rule; recording it is
what lets the next milestone check rather than assume.

**Provenance** — git commit, Python and PyTorch versions, seed, best epoch,
epochs run, and the device it trained on.

## Safety

**Tensors are saved on CPU**, so an artifact trained on MPS loads anywhere.

**`weights_only=True` on load.** A checkpoint is data. Executing arbitrary
pickle from one is a remote-code-execution path that buys nothing here.

**`strict=True` on `load_state_dict`.** A missing key would leave a parameter at
its random initialisation — silent, and producing a subtly wrong model. The
mismatch is raised with an explanation instead.

## Verified round trips

Asserted as exact equality, not approximate:

- user embeddings before == after
- warm item embeddings before == after
- **cold item embedding before == after**, and still equal to the content-only
  path after reload

That last one matters: a cold-start guarantee that holds only in the live model
is not a guarantee. It is checked on both the fixture and the real artifact.

Rejections tested: wrong mapping checksum, wrong feature version, wrong text or
image dimension, missing files, missing metadata, invalid JSON, an artifact from
a different model type, an unsupported format version, corrupt weights, and
weights that do not match the configuration.

## Related

- [Model core](multimodal_two_tower_core.md) · [Training](two_tower_training.md)
