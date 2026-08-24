# PixelRec50K raw schema

Recorded by inspecting the actual downloaded files on 2026-08-24, not from
documentation. Every figure below was computed, not quoted.

## Files

| File | Bytes | Modified | Data rows |
|---|---:|---|---:|
| `interaction.csv` | 28,124,439 | 2023-09-26 | 989,494 |
| `item_info.csv` | 24,973,166 | 2024-11-18 | 82,865 |

Both are UTF-8 CSV with a header row. `item_info.csv` uses quoted fields for
free text; no field contains an embedded newline.

## `interaction.csv`

```text
item_id,user_id,timestamp
i72138,u209296,1605059546
i15530,u2444520,1628914341
```

| Column | Type | Nulls | Notes |
|---|---|---:|---|
| `item_id` | string | 0 | `i` + integer. Universal PixelRec id space (`i0`–`i408373`); the 50K subset uses a sparse subset. |
| `user_id` | string | 0 | `u` + integer. |
| `timestamp` | int64 | 0 | **Unix epoch seconds.** |

### Measured properties

| Property | Value |
|---|---|
| Unique users | 50,000 |
| Unique items | 82,865 |
| Timestamp range | 1328292931 → 1656087360 |
| As UTC | 2012-02-03T18:15:31Z → 2022-06-24T16:16:00Z (3,793 days) |
| Non-positive timestamps | **0** |
| Exact duplicate rows | **0** |
| Duplicate `(user, item)` pairs | **0** |
| Duplicate `(user, timestamp)` pairs | **0** — no per-user ordering ties |
| Interactions per user | min 6 · median 14 · mean 19.79 · max 434 |
| Interactions per item | min 1 · median 6 · mean 11.94 · max 146 |
| Items with a single interaction | **13,518** |
| Sparsity | 0.999761179 |

**There is no event-type column.** Every row is the same undifferentiated
implicit signal. There is no rating, no watch duration, no session id.

**There is no user table.** Users exist only as ids inside this file.

## `item_info.csv`

```text
item_id,view_number,comment_number,thumbup_number,share_number,coin_number,favorite_number,barrage_number,title,tag,description
i192714,799668.0,739.0,8050.0,220.0,84.0,1049.0,510.0,"My boyfriend gave me a turtle…",Pet Reptiles,Should I brush it off?
```

| Column | Type | Nulls | Null % | Notes |
|---|---|---:|---:|---|
| `item_id` | string | 0 | 0.00% | Joins to `interaction.csv`. No duplicates. |
| `view_number` | float64 | 5 | 0.006% | min 94 · median 315,472 · max 39,738,386 |
| `comment_number` | float64 | 5 | 0.006% | min 0 · median 387 |
| `thumbup_number` | float64 | 5 | 0.006% | min 3 · median 9,099 |
| `share_number` | float64 | 5 | 0.006% | min 0 · median 467 |
| `coin_number` | float64 | 5 | 0.006% | min 0 · median 968 |
| `favorite_number` | float64 | 5 | 0.006% | min 0 · median 2,778 |
| `barrage_number` | float64 | 5 | 0.006% | min 0 · median 465 |
| `title` | string | **192** | 0.232% | max length 685 |
| `tag` | string | 5 | 0.006% | **108 distinct values** — a single category per item |
| `description` | string | **19,758** | **23.844%** | max length 2,159 |

No negative counter values. Titles and descriptions are English (translated by
the dataset authors from the original platform).

Most common tags: Film and Television Editing (6,459), Single-player Games
(4,657), Celebrities Mix (4,429), Miscellaneous (4,353), Daily Life (4,340),
Comedy (4,088).

### The five malformed rows

Five items (`i7117`, `i224131`, `i202789`, `i31766`, `i202996`) have all seven
counters, `tag`, and `description` null while retaining a valid `item_id` and
`title`. They are **kept**: an item with an id is still recommendable from
collaborative signal, and dropping it would silently shrink the catalogue. The
gap is recorded in the missingness report.

## Referential integrity

| Check | Result |
|---|---|
| Interaction items missing from `item_info.csv` | **0** |
| Items in `item_info.csv` with no interaction | **0** |

Exactly 82,865 items on both sides — a perfect 1:1 correspondence.

## Fields PixelRec does **not** have

Absent, and never fabricated: item price, brand, publication date, stock,
rating; user demographics, signup date, locale; session identifiers; event
types beyond the single implicit signal; explicit feedback of any kind.

## Cover images and feature vectors

`cover.7z` holds 82,865 JPGs named `<item_id>.jpg`. Not needed by Phase 2, which
does not run any image model.

The pre-extracted `text_feature.json` and `image_feature.json` are separate
full-PixelRec artifacts (~8.6 GiB each, 1024-d, keyed by item id). Structure
confirmed by streaming the first record of each; see
[`multimodal_feature_alignment.md`](multimodal_feature_alignment.md).
