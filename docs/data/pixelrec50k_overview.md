# PixelRec50K

The primary development and evaluation dataset for OmniRank.

## What it is

A sampled subset of [PixelRec](https://github.com/westlake-repl/PixelRec), a
short-video recommendation dataset from the Westlake Representation Learning
Lab. Users interacted with short videos; each video has a cover image, a title,
a category tag, a description, and platform engagement counters.

| | Official figure | Verified in the downloaded files |
|---|---:|---:|
| Users | 50,000 | **50,000** ✅ |
| Items | 82,865 | **82,865** ✅ |
| Interactions | 989,494 | **989,494** ✅ |

## Why PixelRec50K

1. **It is genuinely multimodal.** Cover images and text metadata exist for
   every item, which is what the Phase 4 two-tower retriever needs. Most public
   recommendation datasets are ids and ratings only.
2. **It has real timestamps.** 2012-02-03 → 2022-06-24 UTC, epoch seconds. The
   temporal split protocol ([ADR-002](../adr/ADR-002-temporal-splitting.md)) does
   not have to fall back to file order.
3. **It fits a laptop.** 51 MB of CSV; the full pipeline runs in ~15 seconds on
   an M-series Mac. The full PixelRec is ~200 million interactions and does not.
4. **It is honest about what it measures.** One implicit engagement signal, not
   a synthetic multi-event taxonomy — which forces the pipeline to handle the
   common real-world case rather than a convenient one.

## Licence — read before downloading

Verbatim from [`dataset/LICENSE`](https://github.com/westlake-repl/PixelRec/blob/main/dataset/LICENSE):

> This dataset is provided by the Westlake Representation Learning Lab
> exclusively for non-commercial research and educational purposes. In exchange
> for the permission to the access to the dataset from Westlake Representation
> Learning Lab, you automatically agree to the following terms and conditions:
> Researcher accepts full responsibility for his or her use of the Dataset.
> Under no circumstances will the Westlake Representation Learning Lab be liable
> for any damages or losses arising from the use of the dataset. The dataset is
> provided "as-is," without any express or implied warranties, including but not
> limited to, warranties of merchantability, fitness for a particular purpose,
> non-infringement, or the absence of defects, errors, or viruses. **No rights
> are granted with respect to copying, modifying, publishing, distributing, or
> commercializing the dataset.**

The repository additionally states that it is *"prohibited to privately modify
the dataset and then offer secondary downloads"*.

**Consequences for this project**, enforced rather than merely noted:

- `data/` is git-ignored; no raw or processed PixelRec data is ever committed.
- Test fixtures are **generated**, never sampled from the download.
- The processed outputs are derivatives and are equally not redistributable.
- Commercial use is out of scope while this dataset is the development corpus.

## Getting it

```bash
python scripts/download_pixelrec50k.py            # 51 MB
python scripts/download_pixelrec50k.py --dry-run  # show the plan first
```

The script downloads exactly four known file ids and can never fetch the full
PixelRec. It prints the licence, checks sizes, writes checksums, skips files
already present, and prints manual instructions if Google Drive rate-limits it.

### Manual download

If the automatic download fails:

1. Open <https://drive.google.com/drive/folders/1bQPgM-6yAnzcD0jKBoUUheA9LL5xnCHG>
2. Download `interaction.csv` and `item_info.csv`.
3. Place both in `data/raw/pixelrec50k/`.

### The files

| File | Size | Rows | SHA-256 (verified 2026-08-24) |
|---|---:|---:|---|
| `interaction.csv` | 28,124,439 B | 989,494 | `638b53ec100f760c…` |
| `item_info.csv` | 24,973,166 B | 82,865 | `a073c2c65900f215…` |
| `cover.7z` | — | 82,865 images | not needed by Phase 2 |

### Multimodal vectors — optional, and large

PixelRec also publishes pre-extracted feature vectors. **These are not part of
the PixelRec50K folder**: they cover all 408,374 full-PixelRec items.

| File | Size | Shape |
|---|---:|---|
| `text_feature.json` | **8.65 GiB** | `{item_id: [1024 floats]}` |
| `image_feature.json` | **8.60 GiB** | `{item_id: [1024 floats]}` |

17.3 GB to obtain vectors for the 20% of items PixelRec50K actually uses. They
are therefore **not downloaded by default**. The pipeline runs without them and
reports text/image coverage as **0.0** — it never assumes a modality it does not
have. See [`multimodal_feature_alignment.md`](multimodal_feature_alignment.md).

```bash
python scripts/download_pixelrec50k.py --with-features   # +17.3 GB
```

## Using it

```bash
python scripts/prepare_data.py --config configs/data/pixelrec50k.yaml
```

Outputs, schemas, and reports: [`processed_schemas.md`](processed_schemas.md).

## Citation

```bibtex
@inproceedings{cheng2024image,
  title     = {An Image Dataset for Benchmarking Recommender Systems with Raw Pixels},
  author    = {Cheng, Yu and Yin, Hongyu and Yuan, Zheng and others},
  booktitle = {Proceedings of the 2024 SIAM International Conference on Data Mining (SDM)},
  year      = {2024}
}
```
