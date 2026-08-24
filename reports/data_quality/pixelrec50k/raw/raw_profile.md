# Raw data profile - pixelrec50k

Computed from the source files **before any cleaning**, so it is the baseline every later row-count claim is measured against.

## Source files

| File | Bytes | Rows | SHA-256 |
|---|---:|---:|---|
| `interaction.csv` | 28,124,439 | 989,494 | `638b53ec100f760c…` |
| `item_info.csv` | 24,973,166 | 82,865 | `a073c2c65900f215…` |

## Interactions

| Metric | Value |
|---|---:|
| Rows | 989,494 |
| Unique users | 50,000 |
| Unique items | 82,865 |
| Sparsity | 0.999761179 |
| Exact duplicate events | 0 |
| Interactions referencing an unknown item | 0 |
| Timestamp range | 2012-02-03T18:15:31+00:00 → 2022-06-24T16:16:00+00:00 |
| Span (days) | 3,793 |

### Interactions per user

| min | p25 | median | mean | p95 | p99 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 11 | 14 | 19.79 | 50 | 91 | 434 |

### Interactions per item

| min | p25 | median | mean | p95 | p99 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 6 | 11.94 | 43 | 73 | 146 |

## Items

| Metric | Value |
|---|---:|
| Rows | 82,865 |
| Duplicate item ids | 0 |
| Items with no interaction | 0 |
| Distinct categories | 108 |

## Metadata field coverage

| Field | Present | Missing | Coverage |
|---|---:|---:|---:|
| `title` | 82,673 | 192 | 0.9977 |
| `description` | 63,105 | 19,760 | 0.7615 |
| `category` | 82,860 | 5 | 0.9999 |
| `image_reference` | 82,865 | 0 | 1.0000 |
