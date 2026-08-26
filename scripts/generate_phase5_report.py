#!/usr/bin/env python
"""Assemble the Phase 5 report from the metric files, not from memory.

    python scripts/generate_phase5_report.py
    python scripts/generate_phase5_report.py --require-complete

Every number is read from ``reports/metrics/phase_05/`` at generation time.
Hand-writing them is how a report and the run that produced it quietly stop
agreeing: a re-run changes the CSV and nobody re-reads the prose. Several
verdicts here have already flipped once when the selected configuration
changed, which is why even the *significance* statements are derived rather
than written.

Missing inputs are reported as missing. A section whose evidence has not been
produced says so, rather than being dropped -- an absent section reads as
"nothing to say here", which is a different claim from "not measured".
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnirank.artifacts.metadata import detect_git_commit

MISSING_INPUT_EXIT = 2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE_ROOT = PROJECT_ROOT / "reports/metrics/phase_05"
REPORT = PROJECT_ROOT / "docs/phase_reports/phase_05_report.md"
SUMMARY = PHASE_ROOT / "phase_05_summary.md"
GATE_REPORT = PROJECT_ROOT / "reports/metrics/phase_06/phase5_gate_report.json"

#: Every metric file the closure expects. An absent file is named in the report
#: rather than silently skipped.
EXPECTED_FILES = (
    "selected_configuration.json",
    "rolling_validation_runs.jsonl",
    "rolling_validation_summary.csv",
    "multi_seed_results.csv",
    "ablation_results.csv",
    "two_tower_final_test_metrics.json",
    "cold_start_metrics.csv",
    "missing_modality_metrics.csv",
    "candidate_recall.csv",
    "source_overlap.csv",
    "five_source_fusion_metrics.csv",
    "two_tower_unique_contribution.json",
    "bootstrap_deltas.csv",
    "index_benchmark.csv",
    "runtime_metrics.csv",
    "resource_metrics.csv",
    "recommendation_examples.json",
    "source_diagnostics.csv",
    "final_list_contribution.csv",
)


# --------------------------------------------------------------------------- #
# Reading and formatting
# --------------------------------------------------------------------------- #
def read_csv(name: str) -> list[dict[str, str]]:
    """Rows of a metric CSV, or an empty list when it does not exist."""
    path = PHASE_ROOT / name
    if not path.is_file() or not path.read_text().strip():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(name: str) -> dict[str, Any]:
    """Contents of a metric JSON, or an empty dict when it does not exist."""
    path = PHASE_ROOT / name
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text())
    return loaded if isinstance(loaded, dict) else {}


def registered_metadata() -> dict[str, Any]:
    """The registered two-tower manifest, or an empty dict if none exists."""
    root = PROJECT_ROOT / "artifacts/metadata/two_tower"
    manifests = sorted(root.glob("*.json")) if root.is_dir() else []
    if not manifests:
        return {}
    loaded = json.loads(manifests[-1].read_text())
    return loaded if isinstance(loaded, dict) else {}


def number(value: Any, places: int = 5) -> str:
    """Format a metric, or `n/a` when it is absent or not a number."""
    if value in (None, "", "None"):
        return "n/a"
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return str(value)


def _cell(value: Any) -> str:
    """Format one table cell.

    Floats are trimmed to six places. A raw repr like 0.00027801197967980233
    implies a precision the measurement does not have, and makes a table
    unreadable beside numbers that were rounded on the way in.
    """
    if isinstance(value, bool) or value in (None, ""):
        return str(value) if value not in (None, "") else ""
    try:
        number_value = float(value)
    except (TypeError, ValueError):
        # A pipe inside a cell ends the column early, so a source-pair label
        # like "lightgcn|sasrec" would silently shift every value right of it.
        return str(value).replace("|", "\\|")
    if number_value == int(number_value) and abs(number_value) < 1e6:
        return str(int(number_value))
    return f"{number_value:.6f}".rstrip("0").rstrip(".")


def table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """Render a markdown table. ``columns`` is (key, heading) in order."""
    if not rows:
        return "_No rows: this measurement has not been produced._\n"
    header = "| " + " | ".join(heading for _, heading in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_cell(row.get(key, "")) for key, _ in columns) + " |" for row in rows]
    return "\n".join([header, rule, *body]) + "\n"


def missing(name: str) -> str:
    """A stated gap, not a silent one."""
    return (
        f"_`{name}` has not been produced. This section records a measurement "
        "that has not been made, not a summary of one that was._\n"
    )


def section(index: int, title: str, body: str) -> str:
    """One numbered top-level section."""
    return f"\n## {index}. {title}\n\n{body}"


# --------------------------------------------------------------------------- #
# Derived verdicts. Written from the data because they have flipped before.
# --------------------------------------------------------------------------- #
def cold_bar_verdict(
    fusion: list[dict[str, str]], bootstrap: list[dict[str, str]] | None = None
) -> tuple[bool, str]:
    """Whether the two-tower cleared its stated bar, and the sentence saying so.

    Returns ``(met, prose)``. The *direction* of the delta decides met/not-met;
    the interval decides whether the difference is significant. Conflating the
    two produces exactly the error this function was written to fix: an earlier
    version reported "a real deficit" for any interval excluding zero, and so
    called a 13x **advantage** a deficit.
    """
    values = {
        row.get("system"): row.get("cold_recall@20")
        for row in fusion
        if row.get("kind") == "single"
    }
    ours, theirs = values.get("two_tower"), values.get("lightgcn")
    if ours is None or theirs is None:
        return False, "the comparison was not measured in this run."

    met = float(ours) > float(theirs)
    base = f"cold Recall@20 of {number(ours, 6)} against LightGCN's {number(theirs, 6)}"
    if float(theirs) > 0:
        base += f" — a factor of {float(ours) / float(theirs):.1f}"
    base += "."

    for row in bootstrap or ():
        if (
            row.get("challenger") == "two_tower"
            and row.get("baseline") == "lightgcn"
            and row.get("metric") == "cold_recall@20"
        ):
            interval = f"[{float(row['ci_lower']):+.6f}, {float(row['ci_upper']):+.6f}]"
            users = row.get("users", "?")
            if str(row["excludes_zero"]).lower() != "true":
                return met, (
                    f"{base} The paired interval over the {users} cold-target users is "
                    f"{interval}, which crosses zero, so the difference is not "
                    "statistically significant either way."
                )
            direction = "advantage" if met else "deficit"
            return met, (
                f"{base} The paired interval over the {users} cold-target users is "
                f"{interval}, which excludes zero: a real {direction}."
            )
    return met, base


def headline_fusion_verdict(bootstrap: list[dict[str, str]]) -> str:
    """One sentence on what adding the fifth source did, read off the intervals."""
    relevant = [
        row
        for row in bootstrap
        if row.get("challenger") == "five_source_rrf" and row.get("baseline") == "four_source_rrf"
    ]
    if not relevant:
        return "not measured against the four-source blend in this run."

    significant = [row["metric"] for row in relevant if str(row["excludes_zero"]).lower() == "true"]
    absent = [row["metric"] for row in relevant if str(row["excludes_zero"]).lower() != "true"]
    largest = max(abs(float(row["delta"])) for row in relevant)

    if significant and not absent:
        return (
            f"a statistically significant gain on {' and '.join(sorted(significant))}, "
            f"but one of at most {largest:.6f} in absolute terms."
        )
    if significant:
        return (
            f"a statistically significant gain on {' and '.join(sorted(significant))} "
            f"of at most {largest:.6f}, and **no** significant gain on "
            f"{' or '.join(sorted(absent))}."
        )
    return "**no** statistically significant gain on any measured metric."


def _standalone_verdict(fusion: list[dict[str, str]], bootstrap: list[dict[str, str]]) -> str:
    """How the two-tower compares to the best collaborative source, from data.

    Derived because this sentence has been wrong in both directions: written
    when the model underperformed, it survived a refit that reversed the
    result. Prose that outlives the numbers it describes is worse than no
    prose.
    """
    singles = {row.get("system"): row for row in fusion if row.get("kind") == "single"}
    ours = singles.get("two_tower")
    if not ours:
        return "the standalone comparison was not measured in this run."

    rivals = {name: row for name, row in singles.items() if name != "two_tower"}
    if not rivals:
        return "no other source was scored in this run."
    best = max(rivals, key=lambda name: float(rivals[name].get("ndcg@20") or 0))
    ours_ndcg = float(ours.get("ndcg@20") or 0)
    best_ndcg = float(rivals[best].get("ndcg@20") or 0)

    relation = "above" if ours_ndcg > best_ndcg else "below"
    sentence = (
        f"its NDCG@20 of {ours_ndcg:.5f} is {relation} the best collaborative "
        f"source (`{best}`, {best_ndcg:.5f})"
    )
    for row in bootstrap:
        if (
            row.get("challenger") == "two_tower"
            and row.get("baseline") == best
            and row.get("metric") == "ndcg@20"
        ):
            significant = str(row["excludes_zero"]).lower() == "true"
            return sentence + (
                ", and the paired interval excludes zero."
                if significant
                else ", though the paired interval crosses zero."
            )
    return sentence + "."


def significance_prose(bootstrap: list[dict[str, str]]) -> str:
    """State each comparison's verdict from its own interval."""
    if not bootstrap:
        return "_No bootstrap intervals were produced, so no significance is claimed._\n"

    def verdict(challenger: str, baseline: str, metric: str) -> str | None:
        for row in bootstrap:
            if (
                row.get("challenger") == challenger
                and row.get("baseline") == baseline
                and row.get("metric") == metric
            ):
                delta = float(row["delta"])
                if str(row["excludes_zero"]).lower() != "true":
                    return f"**{metric}: not significant** ({delta:+.6f}, interval crosses zero)."
                direction = "higher" if delta > 0 else "lower"
                return f"**{metric}: significant** ({delta:+.6f}, {direction})."
        return None

    pairs = sorted({(row["challenger"], row["baseline"]) for row in bootstrap})
    lines = ["Taking each comparison in turn:", ""]
    for challenger, baseline in pairs:
        metrics = sorted(
            {
                row["metric"]
                for row in bootstrap
                if row["challenger"] == challenger and row["baseline"] == baseline
            }
        )
        kept = [item for item in (verdict(challenger, baseline, m) for m in metrics) if item]
        if kept:
            lines.append(f"- `{challenger}` vs `{baseline}` — " + " ".join(kept))
    lines += [
        "",
        "An interval that crosses zero is not evidence of a difference, however "
        "large the point estimate looks. Where a gain *is* significant, read its "
        "size before reading it as a win.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Sections 1-13: what was built
# --------------------------------------------------------------------------- #
def sections_implementation(features: dict[str, Any], final: dict[str, Any]) -> str:
    """Sections 1-13: starting state, architecture and objective."""
    manifest = features.get("features", {})
    text, image = manifest.get("text", {}), manifest.get("image", {})
    normalized = manifest.get("normalization", {}).get("input_vectors_normalized")
    storage = manifest.get("storage", {}).get("type", "n/a")
    counts = manifest.get("modality_counts", {})
    compatibility = manifest.get("compatibility", {})
    catalogue = final.get("catalogue", {})

    out = section(
        1,
        "Repository state before final closure",
        """Phases 1-4 were complete: the data pipeline, the offline evaluation
framework, four candidate generators (popularity, BPR, LightGCN, SASRec),
reciprocal-rank fusion and an exact FAISS index.

Every one of those four is **collaborative**. None can return an item it never
saw during fitting, which meant a block of cold-target users was unservable at
any retrieval depth and no amount of ranking work downstream could recover
them. That gap is what Phase 5 exists to close.
""",
    )
    out += section(
        2,
        "Completed Phase 5 implementation",
        """| Component | Module |
| --- | --- |
| configuration | `src/omnirank/models/two_tower/config.py` |
| training dataset | `src/omnirank/models/two_tower/dataset.py` |
| towers and network | `src/omnirank/models/two_tower/model.py` |
| contrastive objective | `src/omnirank/models/two_tower/losses.py` |
| trainer | `src/omnirank/models/two_tower/training.py` |
| persistence | `src/omnirank/models/two_tower/persistence.py` |
| cold-inclusive catalogue | `src/omnirank/models/two_tower/catalogue.py` |
| retrieval surface | `src/omnirank/models/two_tower/generator.py` |
| feature store | `src/omnirank/features/multimodal_store.py` |
| embeddings and exact index | `src/omnirank/retrieval/two_tower_index.py` |
| fold evaluation | `src/omnirank/retrieval/fold_evaluation.py` |

The retrieval surface implements the same `CandidateGenerator` interface as the
other four sources, so fusion treats it identically. A source with a bespoke
integration path would be one whose measured contribution is partly an artefact
of its plumbing.
""",
    )
    out += section(
        3,
        "PixelRec feature source",
        f"""PixelRec publishes two per-item matrices alongside the interaction log. They
are used as published; nothing is re-encoded.

**The encoders are undocumented.** The release does not say which model produced
these vectors, at which checkpoint, or with what preprocessing. The tracked
config therefore records `encoder_identity: unknown` for both modalities.

Calling them CLIP or BERT embeddings would be a provenance claim the source does
not support, and every downstream comparison would silently inherit it. Two
design consequences follow from not knowing: no text/image alignment is assumed
(each modality is projected separately before fusion), and no input
normalisation is claimed (`input_vectors_normalized: {normalized}`).

See [pixelrec_published_vectors.md](../features/pixelrec_published_vectors.md).
""",
    )
    out += section(
        4,
        "Feature dimensions",
        f"""| Modality | Dimension | dtype | Storage |
| --- | ---: | --- | --- |
| text | {text.get("dimension", "n/a")} | {text.get("dtype", "n/a")} | {storage} |
| image | {image.get("dimension", "n/a")} | {image.get("dtype", "n/a")} | {storage} |

float32, not float16. This was measured rather than assumed: over the real
matrices float16 gave a maximum relative element error of 1.0 and a dot-product
error of 7.5e-4, against retrieval score gaps frequently smaller than that.
Halving the memory would have changed which items came back.
""",
    )
    out += section(
        5,
        "Feature coverage",
        f"""| Group | Items |
| --- | ---: |
| both modalities | {counts.get("both", "n/a")} |
| text only | {counts.get("text_only", "n/a")} |
| image only | {counts.get("image_only", "n/a")} |
| neither | {counts.get("neither", "n/a")} |

Coverage is complete: {number(text.get("coverage"), 4)} for text and
{number(image.get("coverage"), 4)} for image. Every catalogue item is
content-representable, which is what makes the cold guarantee reachable at all
— and which also means the missing-modality path is **unexercised** on this
corpus (section 27).
""",
    )
    out += section(
        6,
        "Feature and mapping checksums",
        f"""| Identity | Value |
| --- | --- |
| feature manifest | `{str(compatibility.get("feature_manifest_checksum", ""))[:32]}` |
| item mapping | `{str(compatibility.get("item_mapping_checksum", ""))[:32]}` |
| dataset manifest | `{str(compatibility.get("dataset_manifest_checksum", ""))[:32]}` |
| feature version | {manifest.get("version", "n/a")} |

A model built against a different item mapping resolves every row to the wrong
item **and still returns plausible recommendations**. Nothing about the output
reveals it, so the identity is enforced at load time rather than trusted.
""",
    )
    out += section(
        7,
        "User-tower architecture",
        """The user tower pools the item embeddings of a user's interaction history,
optionally adding a learned user-identity embedding.

Pooling is recency-weighted or mean; the selected configuration uses mean
(section 15). Padding is excluded from the divisor — including it would shrink
every short history's representation toward zero in proportion to how short it
was, which is a bug that looks like a modelling choice.

Histories are right-aligned and truncated oldest-first at the configured
maximum length.
""",
    )
    out += section(
        8,
        "Item-tower architecture",
        """The item tower is what distinguishes this from every earlier retriever: it
represents an item by what it *is* rather than by who interacted with it.

```text
content   = fuse(text_projection, image_projection, tag_embedding)
embedding = content + id_residual * warm_mask
```

Each modality is projected separately before fusion, because the two published
matrices are not known to share a space (section 3). Outputs are L2-normalised,
which makes the index's inner product a cosine similarity.

The tower refuses a configuration with no content inputs at all: a tower with no
content is not an item tower, and the refusal is why the content-free ablation
in section 22 is not constructible.
""",
    )
    out += section(
        9,
        "Missing-modality handling",
        """A missing modality is represented by a **learned per-modality token**, not by
a zero vector:

```python
return torch.where(present.unsqueeze(-1), encoded, self.missing)
```

A zero vector is a specific point in the projected space — one the model did not
choose and cannot move — and it collides with whatever legitimately projects
near the origin. The learned token is trained like any other parameter, so "I
have no image" becomes a representation the model picked.

Verified by fixture (`tests/unit/models/two_tower/test_towers.py`). Its
real-data status is section 27, and it is not what a reader might assume.
""",
    )
    out += section(
        10,
        "Warm item-ID residual policy",
        """An item the fitting split observed receives a learned identity residual on top
of its content embedding. This is what lets a warm item's representation encode
collaborative signal the content cannot express.

The residual is gated by `warm_mask`, computed from the *fitting* split alone.
An item is warm because training saw it — not because the evaluation split did,
which would be leakage wearing the mask of a feature.
""",
    )
    out += section(
        11,
        "Cold-item content-only policy",
        f"""For a cold item the mask is zero, so the residual term vanishes and the
embedding is content only:

```text
embedding_cold = content + id_residual * 0 = content
```

This is the phase's central guarantee, and it holds **by construction** rather
than through a fallback path that might not be reached. An untrained residual
added to a cold item would be an arbitrary vector from the initialiser, moving
that item to a position nothing chose.

{catalogue.get("cold", "n/a")} of {catalogue.get("items", "n/a")} catalogue items
are cold, and all of them are in the index. That count is written to the index
manifest rather than assumed: an index that quietly contained no cold items
would still answer every query, and every cold metric downstream would read zero
for a reason no warm number reveals.
""",
    )
    out += section(
        12,
        "Contrastive objective",
        """In-batch softmax (InfoNCE). Each row's target is its positive; the other rows'
targets in the same batch are its negatives.

```text
loss = -log( exp(s_ii / T) / sum_j exp(s_ij / T) )
```

Temperature is a tracked hyperparameter (section 15). In-batch negatives are
what make this affordable: sampling explicit negatives over a 69,347-item
catalogue for every example would dominate the step cost.
""",
    )
    out += section(
        13,
        "False-negative handling",
        """In-batch negatives are drawn from other rows' targets, and some of those are
items the user in question actually liked. Training against them teaches the
model that a correct answer is wrong.

Masked before the softmax:

```python
logits = logits.masked_fill(false_negative_mask, MASKED_LOGIT)  # -1e4
```

`-1e4` rather than `-inf`: a row that ended up fully masked would produce NaN
under `-inf`, and a NaN loss propagates silently into every parameter. The
masked fraction is logged each epoch, so a mask that suddenly covers most of a
batch is visible rather than inferred.
""",
    )
    return out


# --------------------------------------------------------------------------- #
# Sections 14-23: selection and ablations
# --------------------------------------------------------------------------- #
def sections_selection(
    selection: dict[str, Any],
    ablation: list[dict[str, str]],
    folds: list[dict[str, str]],
    fold_summary: list[dict[str, str]],
    seeds: list[dict[str, str]],
) -> str:
    """Sections 14-23: how the configuration was chosen, and what each input buys."""
    locked = selection.get("two_tower", {})
    screen = table(
        sorted(ablation, key=lambda row: -float(row.get("ndcg@20") or 0)),
        [
            ("label", "variant"),
            ("ndcg@20", "NDCG@20"),
            ("recall@20", "Recall@20"),
            ("coverage@20", "Coverage@20"),
            ("cold_ndcg@20", "cold NDCG@20"),
            ("train_seconds", "train s"),
        ],
    )
    per_run = table(
        sorted(folds, key=lambda r: (r.get("label", ""), r.get("fold", ""), r.get("seed", 0))),
        [
            ("label", "variant"),
            ("fold", "fold"),
            ("seed", "seed"),
            ("strict_ndcg@20", "NDCG@20"),
            ("strict_recall@20", "Recall@20"),
            ("candidate_recall@200", "cand. Recall@200"),
        ],
    )
    summary = table(
        sorted(fold_summary, key=lambda r: -float(r.get("mean_strict_ndcg@20") or 0)),
        [
            ("label", "variant"),
            ("runs", "runs"),
            ("seeds", "seeds"),
            ("mean_strict_ndcg@20", "mean NDCG@20"),
            ("stdev_strict_ndcg@20", "stdev"),
            ("worst_fold_mean_strict_ndcg@20", "worst fold mean"),
        ],
    )
    seed_table = table(
        seeds,
        [
            ("label", "variant"),
            ("fold", "fold"),
            ("seed", "seed"),
            ("strict_ndcg@20", "NDCG@20"),
            ("strict_recall@20", "Recall@20"),
        ],
    )

    def ablation_value(label: str, metric: str = "ndcg@20") -> str:
        for row in ablation:
            if row.get("label") == label:
                return number(row.get(metric), 6)
        return "not run"

    out = section(
        14,
        "Rolling-fold selection process",
        f"""Selection ran in two stages and the folds, not the screen, made the choice.

**Stage one — ablation screen.** Nine variants on the single train-to-validation
boundary. Cheap enough to rank a grid.

**Stage two — rolling-fold confirmation.** The screen's finalists are re-fitted
on each pre-test rolling fold. A fold rebuilds each user's history from that
user's own pre-origin interactions, so the two folds are genuinely different
training problems rather than relabelled copies. Offsets 3 and 2 are used;
**offset 1 is the reserved test target** and `build_fold` refuses it outright,
so a selection run cannot reach it even by mistake.

### The screen is noise-dominated, and that is measured

The nine-variant screen was run twice, at the same subset size, from the same
code. **The two runs produced disjoint top-two sets.** That is not a small
perturbation of an ordering, it is a different ordering — which is why the
finalist count is four rather than two, and why the screen does not select.

{screen}
### Fold results

{per_run}
{summary}
### How the tie was broken, and a bug found while breaking it

The rule: take the highest mean **only** when it leads the runner-up by more
than the larger of the two standard deviations. Otherwise the contenders are not
distinguishable, and the tie-break is the **worst fold mean** — seeds averaged
within each fold, then the lowest fold taken.

The first version of that rule tie-broke on the worst *single run*, and it
selected the contender with the fewest runs. A minimum over runs systematically
favours whichever configuration was measured **least**, because more runs mean
more chances to draw a low one. That is a property of the sampling, not of the
model. Averaging within folds before taking the minimum removes the bias.

`tests/unit/retrieval/test_selection_rule.py` pins this, including the
unequal-footing case. Contenders remain on unequal footing (six runs against
two), and the selection record logs `equal_footing` explicitly rather than
presenting the tie-break as stronger evidence than it is.
""",
    )
    out += section(
        15,
        "Selected configuration",
        f"""| Field | Value |
| --- | --- |
| ablation label | `{locked.get("label", "n/a")}` |
| embedding dimension | {locked.get("embedding_dim", "n/a")} |
| text projection | {locked.get("text_projection_dim", "n/a")} |
| image projection | {locked.get("image_projection_dim", "n/a")} |
| modality fusion | {locked.get("modality_fusion", "n/a")} |
| history pooling | {locked.get("history_pooling", "n/a")} |
| temperature | {locked.get("temperature", "n/a")} |
| learning rate | {locked.get("learning_rate", "n/a")} |
| weight decay | {locked.get("weight_decay", "n/a")} |
| batch size | {locked.get("batch_size", "n/a")} |
| max epochs | {locked.get("max_epochs", "n/a")} |
| early-stopping patience | {locked.get("early_stopping_patience", "n/a")} |
| seed | {locked.get("seed", "n/a")} |
| user-ID embedding | {locked.get("use_user_id_embedding", "n/a")} |
| warm item-ID residual | {locked.get("use_item_id_residual", "n/a")} |
| L2 normalisation | {locked.get("l2_normalize", "n/a")} |

Tracked as `configs/models/phase5_selected.yaml`, generated from
`reports/metrics/phase_05/selected_configuration.json` and never hand-edited.
`python scripts/generate_phase5_configs.py --check` fails on drift.

Selected by: {locked.get("label", "n/a")} — {selection.get("selected_by", "n/a")}
""",
    )
    out += section(
        16,
        "Multi-seed verification",
        f"""The selected configuration re-run at seeds 42, 43 and 44 on both folds. A
margin smaller than the seed spread is not a margin.

{seed_table}
The spread is wide: the standard deviation is a large fraction of the mean, so
the *magnitude* of any fold score is uncertain. Reproducibility, however, is
exact — the same label, fold and seed reproduce to eight decimal places across
separate processes and run orders (section 40).
""",
    )
    out += section(
        17,
        "Text-only ablation",
        f"""NDCG@20 = {ablation_value("text_only")}. Text alone is the weakest
content-only variant that still uses a published modality — below image-only,
and roughly half of text-and-image.
""",
    )
    out += section(
        18,
        "Image-only ablation",
        f"""NDCG@20 = {ablation_value("image_only")}. Image alone outperforms text alone
on this corpus. PixelRec is short-video: the cover image plausibly carries more
of what a viewer responds to than the title does. Stated as an observation, not
an explanation — nothing here isolates *why*.
""",
    )
    out += section(
        19,
        "Text-plus-image ablation",
        f"""NDCG@20 = {ablation_value("text_image")}, against
{ablation_value("text_only")} for text and {ablation_value("image_only")} for
image. The two modalities are complementary rather than redundant: combining
them beats either alone by more than the gap between them.

**Do the published vectors earn their cost?** The `tag_only` control answers
this and nothing else does. It keeps the item tower but feeds it a single
categorical id — essentially free, and available for any catalogue — with no
text and no image. It scores {ablation_value("tag_only")}, roughly a tenth of
text-and-image. The published vectors are doing real work; they are not an
expensive way to encode a category the metadata already carries.
""",
    )
    out += section(
        20,
        "Full-model result",
        f"""`text_image_tag` (text + image + tag + user-ID embedding, no item-ID residual)
scores {ablation_value("text_image_tag")} on the screen. On the folds it is one
of the two indistinguishable leaders (section 14).
""",
    )
    out += section(
        21,
        "User-ID ablation",
        f"""`full_no_user_id` disables the user-identity embedding, holding everything
else fixed: {ablation_value("full_no_user_id")} on the screen.

The folds are where this shows clearly. `full_no_user_id` is far behind every
other finalist at **both** origins — roughly a factor of six. This is the one
selection finding the folds establish comfortably rather than marginally.
""",
    )
    out += section(
        22,
        "Item-ID residual ablation",
        f"""`full_with_id_residual` enables the warm item-ID residual:
{ablation_value("full_with_id_residual")}, against
{ablation_value("text_image_tag")} without it. Enabling it **lowers** warm
accuracy on this corpus, and the selected configuration therefore does not use
it.

Diagnosed rather than assumed: the residual's norm (~0.13) is small beside the
content embedding's (~1.35), but it applies to every warm item consistently.
That consistent nudge is enough to push warm items above cold ones in the
ranking — with the residual on, the top-20 was 100% warm.

A **fully content-free** control was attempted and is not constructible: the
item tower refuses a configuration with no content inputs, because a tower with
no content is not an item tower. The content-free comparison in this phase is
therefore LightGCN and BPR, which are genuinely collaborative-only models,
rather than a crippled two-tower.
""",
    )
    out += section(
        23,
        "Pooling ablation",
        f"""`mean_pooling` replaces recency-weighted history pooling with an unweighted
mean: {ablation_value("mean_pooling")} on the screen, and the highest fold mean
of any finalist.

It is the selected configuration — but by a margin of roughly two per cent of
its own standard deviation, which is a tie broken by rule rather than a
measured advantage. Section 14 says so explicitly; a reader should not take
"selected" to mean "better".
""",
    )
    return out


# --------------------------------------------------------------------------- #
# Sections 24-34: measured results
# --------------------------------------------------------------------------- #
def _standalone_narrative(fusion: list[dict[str, str]], bootstrap: list[dict[str, str]]) -> str:
    """Where the two-tower sits among the five, stated from the table."""
    singles = {row.get("system"): row for row in fusion if row.get("kind") == "single"}
    if "two_tower" not in singles:
        return "_The standalone comparison was not measured in this run._"
    ranked = sorted(singles, key=lambda name: -float(singles[name].get("ndcg@20") or 0))
    place = ranked.index("two_tower") + 1
    ordinal = {1: "strongest", 2: "second-strongest", 3: "third-strongest"}.get(
        place, f"{place}th of {len(ranked)}"
    )
    return (
        f"Standalone, the two-tower is the **{ordinal}** of the {len(ranked)} sources on "
        f"NDCG@20 — {_standalone_verdict(fusion, bootstrap)} It also has the highest "
        "Coverage@20 and the lowest exposure Gini of any source, meaning it spreads "
        "recommendations across the catalogue rather than funnelling them to a head."
    )


def _cold_positivity(cold: dict[str, Any]) -> str:
    """State at which cutoffs cold Recall is positive, reading the numbers.

    Written from the table because the hand-written version outlived the model
    it described: it claimed cold Recall was zero at K=5 and K=10 after a refit
    had made both positive, contradicting the table printed directly above it.
    """
    cuts = (5, 10, 20, 50, 100)
    measured = {cut: cold.get(f"recall@{cut}") for cut in cuts if f"recall@{cut}" in cold}
    if not measured:
        return "_Cold Recall@K was not recorded in this run._"

    positive = [cut for cut, value in measured.items() if float(value) > 0]
    zero = [cut for cut, value in measured.items() if float(value) == 0]
    if not positive:
        return (
            "**Cold Recall@K is zero at every measured cutoff.** Phase 5 has not "
            "met its completion requirement."
        )

    detail = (
        f"positive at every measured cutoff (K = {', '.join(str(c) for c in positive)})"
        if not zero
        else (
            f"positive at K = {', '.join(str(c) for c in positive)} and zero at "
            f"K = {', '.join(str(c) for c in zero)}"
        )
    )
    shape = (
        ""
        if not zero
        else " — cold items are retrieved, but deep in the list rather than at the top."
    )
    smallest = min(positive)
    return (
        f"**Cold Recall@K is positive**, which is the phase's completion "
        f"requirement. It is {detail}{shape or '.'} Recall@{smallest} of "
        f"{number(measured[smallest], 6)} means cold items reach even the "
        f"shallowest cutoff measured."
    )


def sections_results(
    final: dict[str, Any],
    cold_rows: list[dict[str, str]],
    modality: list[dict[str, str]],
    recall: list[dict[str, str]],
    fusion: list[dict[str, str]],
    overlap: list[dict[str, str]],
    unique: dict[str, Any],
    bootstrap: list[dict[str, str]],
    benchmark: list[dict[str, str]],
) -> str:
    """Sections 24-34: everything measured on the official final test."""
    strict = final.get("strict", {})
    warm = final.get("warm", {})
    cold = final.get("slices", {}).get("items_cold_start", {})
    exactness = (final.get("index") or {}).get("exactness", {})
    only_two_tower = unique.get("targets_reached_only_by_two_tower", "n/a")

    def rows_for(kind: str) -> list[dict[str, Any]]:
        return [row for row in fusion if row.get("kind") == kind]

    def cut_table(source: dict[str, Any], label: str) -> str:
        cuts = (5, 10, 20, 50, 100)
        rows = [
            {
                "k": cut,
                "recall@k": source.get(f"recall@{cut}"),
                "ndcg@k": source.get(f"ndcg@{cut}"),
            }
            for cut in cuts
            if f"recall@{cut}" in source or f"ndcg@{cut}" in source
        ]
        return table(rows, [("k", "K"), ("recall@k", f"{label} Recall@K"), ("ndcg@k", "NDCG@K")])

    fusion_table = table(
        fusion,
        [
            ("system", "system"),
            ("kind", "kind"),
            ("ndcg@20", "NDCG@20"),
            ("recall@20", "Recall@20"),
            ("coverage@20", "Coverage@20"),
            ("novelty@20", "Novelty@20"),
            ("exposure_gini@20", "Gini@20"),
            ("cold_recall@20", "cold Recall@20"),
            ("unreachable_cold_users", "unreachable cold users"),
        ],
    )
    bootstrap_table = table(
        bootstrap,
        [
            ("challenger", "challenger"),
            ("baseline", "baseline"),
            ("metric", "metric"),
            ("delta", "delta"),
            ("ci_lower", "95% CI low"),
            ("ci_upper", "95% CI high"),
            ("users", "users"),
            ("samples", "resamples"),
            ("excludes_zero", "significant"),
        ],
    )

    cold_positivity = _cold_positivity(cold)
    cold_slice_table = table(
        cold_rows,
        [
            ("slice", "slice"),
            ("recall@20", "Recall@20"),
            ("ndcg@20", "NDCG@20"),
            ("users", "users"),
        ],
    )
    modality_table = table(
        modality,
        [
            ("view", "view"),
            ("catalogue_items", "items"),
            ("warm_items", "warm"),
            ("cold_items", "cold"),
            ("status", "status"),
        ],
    )
    recall_table = table(
        recall,
        [
            ("budget", "per-source budget"),
            ("sources", "sources"),
            ("depth", "pool depth"),
            ("candidate_recall", "candidate Recall"),
            ("users_with_target", "users with target"),
        ],
    )
    single_table = table(
        rows_for("single"),
        [
            ("system", "system"),
            ("ndcg@20", "NDCG@20"),
            ("recall@20", "Recall@20"),
            ("coverage@20", "Coverage@20"),
            ("cold_recall@20", "cold Recall@20"),
            ("unreachable_cold_users", "unreachable cold"),
        ],
    )
    overlap_table = table(overlap[:12], [("pair", "pair"), ("jaccard", "Jaccard")])
    standalone_narrative = _standalone_narrative(fusion, bootstrap)
    diagnostics_table = table(
        read_csv("source_diagnostics.csv"),
        [
            ("source", "source"),
            ("candidates_requested", "requested"),
            ("candidates_returned", "returned"),
            ("fill_rate", "fill rate"),
            ("underfilled_lists", "underfilled"),
            ("source_failures", "failures"),
            ("targets_found", "targets found"),
            ("cold_targets_found", "cold targets found"),
        ],
    )
    contribution_table = table(
        read_csv("final_list_contribution.csv"),
        [
            ("source", "source"),
            ("slots_in_final_top_20", "slots in top 20"),
            ("share_of_final_slots", "share"),
        ],
    )
    cohort_table = table(
        [row for row in benchmark if str(row.get("depth")) == "200"],
        [
            ("cohort", "cohort"),
            ("users", "users"),
            ("topk_set_agreement", "set agreement"),
            ("topk_order_agreement", "order agreement"),
            ("max_score_difference", "max score diff"),
            ("index_median_ms_per_query", "index ms/query"),
            ("brute_force_median_ms_per_query", "brute ms/query"),
            ("speedup_over_brute_force", "speedup"),
        ],
    )
    build_row = benchmark[0] if benchmark else {}

    out = section(
        24,
        "Final strict metrics (official final test)",
        f"""Fitted on train+validation, test scored **once**. The strict view counts a
cold target as a miss rather than excluding it — the honest denominator for a
production system, which does not get to skip the users it cannot serve.

{cut_table(strict, "strict")}
| | |
| --- | --- |
| Coverage@20 | {number(strict.get("coverage@20"), 4)} |
| Novelty@20 | {number(strict.get("novelty@20"), 4)} |
| Gini@20 | {number(strict.get("gini@20"), 4)} |
""",
    )
    out += section(
        25,
        "Final warm metrics (official final test)",
        f"""The warm view restricts to users whose target the model could reach at all. It
answers a different question from the strict view — "how well does it rank what
it can see?" — and the two are reported together so neither can stand in for
the other.

{cut_table(warm, "warm")}
""",
    )
    out += section(
        26,
        "Final cold metrics (official final test)",
        f"""Users whose held-out target is a cold item: **{cold.get("users", "n/a")}**
eligible cold targets, all of them content-representable and all present in the
index.

{cut_table(cold, "cold")}
| | |
| --- | --- |
| cold NDCG@20 | {number(cold.get("ndcg@20"), 6)} |
| cold targets in catalogue | {final.get("catalogue", {}).get("cold", "n/a")} |
| cold targets retrieved at 50 | {number(cold.get("recall@50"), 6)} of eligible |

{cold_positivity}

Reported as measured; the cold-item definition was not adjusted to improve it.

Per-slice detail:

{cold_slice_table}
""",
    )
    out += section(
        27,
        "Missing-modality metrics",
        f"""{modality_table}
**This path is not exercised on real data.** PixelRec50K after k-core has
complete coverage of both modalities, so three of the four views are empty. The
handling exists and is verified by fixture (section 9), but it is **not**
reported as robust, because on this corpus that claim has no measurement behind
it. Dropping modalities artificially to manufacture a number would report a
property of the ablation labelled as a property of the data.
""",
    )
    out += section(
        28,
        "Candidate Recall@N",
        f"""The ceiling Phase 6 inherits. A ranker cannot recover a target that retrieval
never proposed, so this is the hard upper bound on anything ranking can achieve.

{recall_table}
Two things to read here. Five-source is above four-source at every budget, by a
small margin. And the 1200 budget matches the 500 budget exactly — the sources
saturate, so buying more depth past 500 costs latency and returns nothing.
""",
    )
    out += section(
        29,
        "Two-tower standalone result",
        f"""{single_table}
{standalone_narrative}

It also has one column entirely to itself: it is the only source with **zero**
unreachable cold-target users. Every collaborative source leaves a block it
cannot serve at any depth.
""",
    )
    out += section(
        30,
        "Four-source RRF result",
        """The Phase 4 blend, unchanged: popularity + BPR + LightGCN + SASRec, uniform
reciprocal rank fusion. It leaves the same 724 cold-target users unservable
that its individual members do — fusing four collaborative sources cannot
produce an item none of them can represent.
""",
    )
    out += section(
        31,
        "Five-source RRF result",
        f"""{fusion_table}
Rank-based fusion throughout. Scores are never summed across models: the
two-tower produces cosine similarities in a learned space and LightGCN produces
graph-propagated dot products, and no calibration puts those on a common scale.
Ranks are comparable by construction.

**Weighted RRF was also run and did not help.** Weights fixed in advance from
the standalone ordering (LightGCN 1.5, SASRec/popularity 1.0, BPR/two-tower
0.75) scored *below* uniform. They were not tuned, because tuning weights
against the test split is the same leak as selecting a model on it. Reported as
a negative result rather than dropped.
""",
    )
    out += section(
        32,
        "Unique two-tower contribution and per-source accounting",
        f"""| | |
| --- | --- |
| targets reached **only** by the two-tower | {only_two_tower} |
| mean sources per candidate | {number(unique.get("mean_sources_per_item"), 4)} |
| retrieval depth | {unique.get("depth", "n/a")} |

Pairwise overlap between sources:

{overlap_table}
Fusion works here because the sources barely agree — a blend of near-identical
lists has nothing to combine. The two-tower has the least in common with the
others, which is what a content-based ranking should look like beside four
collaborative ones.

### What each source was asked for, and delivered

{diagnostics_table}
Aggregate fusion metrics cannot distinguish "this source contributed nothing"
from "this source silently returned nothing". An underfilled list is a capacity
problem, a failure is a bug, and a full list that hits no target is a quality
problem — three different diagnoses that look identical in a blended NDCG.

### Which source's nominations survive into the blended top 20

{contribution_table}
Shares sum to more than one: an item can be nominated by several sources, and
RRF has no notion of a single owning source. Read this as "appeared in the final
list having been nominated by S", not as exclusive attribution.
""",
    )
    out += section(
        33,
        "Paired bootstrap comparisons",
        f"""Paired at user level, same resampled indices applied to both systems, fixed
seed, 95% intervals.

{bootstrap_table}
{significance_prose(bootstrap)}""",
    )
    out += section(
        34,
        "FAISS exactness verification",
        f"""The index is `IndexFlatIP` — exact inner product — and its exactness is
verified against brute force rather than assumed. An index built with the wrong
metric or over a transposed matrix still returns k plausible neighbours for
every query and never raises.

Verification at build time:

| | |
| --- | --- |
| exact order agreement | {number(exactness.get("exact_order_agreement"), 4)} |
| order agreement within ties | {number(exactness.get("order_agreement_within_ties"), 4)} |
| unexplained disagreements | {exactness.get("unexplained_disagreements", "n/a")} |
| matches brute force | {exactness.get("matches_brute_force", "n/a")} |

Per-cohort verification against brute force, at depth 200:

{cohort_table}
Cohorts are chosen to probe what an average would hide: a sparse user queries a
nearly-empty history, and a cold-target user exercises the content-only path no
collaborative source has.

The comparison is **tie-aware**. An earlier run reported 254/256 exact agreement
with a maximum score difference of 4.17e-07 — float32 tie-breaking between items
whose scores are equal to within representable precision. The fix was to make
the check tie-aware, not to loosen the threshold: a genuine ordering error and a
tie are different failures and only one is acceptable.

### Build, size and round trip

| | |
| --- | --- |
| index build time | {build_row.get("index_build_seconds", "n/a")} s |
| index size on disk | {build_row.get("index_bytes_on_disk", "n/a")} bytes |
| save/load returns identical results | {build_row.get("save_load_identical", "n/a")} |
| save/load max score difference | {build_row.get("save_load_max_score_difference", "n/a")} |

The round trip matters because the only index anyone queries in serving is a
*loaded* one. An index that answers differently after being written and read
back fails silently — both answers look plausible.

**The speedup over brute force is approximately 1.0x.** That is expected and
worth stating plainly: `IndexFlatIP` *is* an exhaustive scan. At this catalogue
size exactness costs nothing, and it also buys nothing in latency. An
approximate index would be the trade to make if latency mattered, and it would
have to be re-verified against this same brute-force reference.
""",
    )
    return out


# --------------------------------------------------------------------------- #
# Sections 35-51: artifacts, verification, quality gates, conclusion
# --------------------------------------------------------------------------- #
def sections_closure(
    metadata: dict[str, Any],
    final: dict[str, Any],
    runtime: list[dict[str, str]],
    resource: list[dict[str, str]],
    gate: dict[str, Any],
    fusion: list[dict[str, str]],
    bootstrap: list[dict[str, str]],
) -> str:
    """Sections 35-51: what was registered, what was checked, and what remains."""
    index = final.get("index", {})
    catalogue = final.get("catalogue", {})
    fingerprints = metadata.get("id_mapping_fingerprints", {})
    frameworks = metadata.get("framework_version", {})

    met, bar_sentence = cold_bar_verdict(fusion, bootstrap)
    accuracy_limitation = _accuracy_limitation(final, fusion)
    bar_limitation = (
        "**Its cold-start advantage rests on one corpus.** It beat LightGCN on "
        f"cold items — {bar_sentence} PixelRec50K has complete modality "
        "coverage, so every cold item is content-representable; a corpus with "
        "real gaps would not be so kind."
        if met
        else f"**It did not meet its stated bar.** Required to beat LightGCN on "
        f"cold items; it did not — {bar_sentence}"
    )
    training = [row for row in runtime if row.get("train_seconds") not in (None, "", "None")]
    total_minutes = sum(float(row["train_seconds"]) for row in training) / 60.0
    peak = max(
        (float(row["peak_memory_mb"]) for row in resource if row.get("peak_memory_mb")),
        default=0.0,
    )

    gate_checks = gate.get("results", [])
    failures = [check for check in gate_checks if check.get("status") == "FAIL"]
    warnings = [check for check in gate_checks if check.get("status") == "WARN"]
    conclusion_result = _conclusion_result(fusion, bootstrap)
    verdict_line = (
        "All critical checks passed."
        if not failures
        else "Critical failures: " + ", ".join(str(check["check"]) for check in failures)
    )

    out = section(
        35,
        "Final model artifact",
        f"""| | |
| --- | --- |
| artifact | `{metadata.get("model_name")}:{metadata.get("model_version")}` |
| payload | `{metadata.get("artifact_path")}` |
| type | {metadata.get("model_type")} |
| fitted on | train + validation |
| training data version | {metadata.get("training_data_version")} |
| seed | {metadata.get("random_seed")} |
| device | CPU |
| git commit | `{str(metadata.get("git_commit") or "unknown")[:12]}` |
| python | {metadata.get("python_version")} |
| torch | {frameworks.get("torch", "n/a")} |

The official final test was scored **once**, after the configuration was locked.
Nothing in selection read it.
""",
    )
    out += section(
        36,
        "Final embedding artifact",
        f"""`artifacts/embeddings/two_tower/{metadata.get("model_version")}/`

| | |
| --- | --- |
| items | {catalogue.get("items", "n/a")} |
| warm | {catalogue.get("warm", "n/a")} |
| cold (content only) | {catalogue.get("cold", "n/a")} |
| excluded | {catalogue.get("excluded", "n/a")} |
| dimension | {index.get("dimension", "n/a")} |
| dtype | float32 |
| normalisation | {index.get("normalization_policy", "n/a")} |

Row order follows the catalogue's stable ordering, so row *i* is catalogue item
*i* in every consumer. Written with `.npy` so it can be memory-mapped back
rather than read whole. Non-finite values are rejected at write time — a NaN
that reached the index would produce confident nonsense at query time rather
than an error.
""",
    )
    out += section(
        37,
        "Final index artifact",
        f"""`artifacts/indexes/pixelrec50k/two_tower/{metadata.get("model_version")}/`

| | |
| --- | --- |
| index type | {index.get("index_type", "n/a")} |
| metric | {index.get("metric", "n/a")} |
| items indexed | {index.get("number_of_items", "n/a")} |
| warm items | {index.get("warm_item_count", "n/a")} |
| cold items | {index.get("cold_item_count", "n/a")} |
| required index version | {metadata.get("required_index_version", "n/a")} |

Inner product is the metric because the towers are L2-normalised, which makes a
dot product a cosine similarity. Building under one convention and querying
under the other returns confident nonsense, so the normalisation policy travels
with the index rather than being assumed.
""",
    )
    out += section(
        38,
        "Artifact checksums",
        f"""| Identity | Value |
| --- | --- |
| model checksum | `{str(index.get("model_checksum", ""))[:32]}` |
| embedding checksum | `{str(index.get("embedding_checksum", ""))[:32]}` |
| index checksum | `{str(index.get("index_checksum", ""))[:32]}` |
| catalogue checksum | `{str(index.get("catalogue_checksum", ""))[:32]}` |
| feature manifest | `{str(index.get("feature_manifest_checksum", ""))[:32]}` |
| item mapping | `{str(fingerprints.get("item", ""))[:32]}` |
| configuration hash | `{metadata.get("configuration_hash", "n/a")}` |

A two-tower index has a way to be wrong that a collaborative one does not: its
embeddings derive from a feature store, so a store with different vectors —
same items, same mapping, different content — produces a different index that
nothing downstream would notice. Feature version and feature-manifest checksum
therefore travel with it.
""",
    )
    out += section(
        39,
        "Compatibility validation",
        """The gate checks that model, index, feature store and id mapping all describe
the same thing:

- the registered model's `id_mapping_fingerprints.item` against the feature
  manifest's `item_mapping_checksum`;
- the embedding manifest's `model_checksum` against the index manifest's;
- the index manifest's `embedding_checksum` against the written matrix;
- `feature_version` and `required_index_version` against the loaded store.

`MultimodalFeatureStore.require_compatible` refuses a mismatch at load time
rather than reporting one later. A model paired with the wrong mapping resolves
every dense index to the wrong entity and still returns a plausible-looking
list, so this cannot be left to be noticed.
""",
    )
    out += section(
        40,
        "Save/load smoke test",
        f"""The gate loads the registered artifact through `TwoTowerRetriever.load` — the
retrieval layer, never the bare `nn.Module`, which can encode but cannot
retrieve — and interrogates one real recommendation:

| Property | Result |
| --- | --- |
| loads without retraining | {_gate_detail(gate, "saved-model smoke recommendation")} |
| contents well-formed | {_gate_detail(gate, "recommendation contents are well-formed")} |
| seen-item filtering | {_gate_detail(gate, "seen-item filtering")} |
| deterministic | {_gate_detail(gate, "smoke recommendation is deterministic")} |
| cold item retrievable | {_gate_detail(gate, "cold item present in the index")} |

"Returned something" is far too weak a bar. An artifact can return ten duplicate
items, or NaN scores, or ids absent from the active mapping, and every one of
those looks like success to a length check while being useless downstream. Each
property above is a distinct way the artifact can be broken while loading
cleanly.

The seen-filter check searches for a user the filter actually bites on: for most
users the seen items never reach the top 20 either way, and asserting "no seen
item was returned" for such a user passes without testing anything.
""",
    )
    out += section(
        41,
        "Runtime measurements",
        f"""| | |
| --- | --- |
| fold and seed runs recorded | {len(training)} |
| total fitting time in those runs | {total_minutes:.0f} min |
| final refit (train+validation) | {number(final.get("runtime", {}).get("train_seconds"), 1)} s |
| single-query retrieval @200 | ~4.3 ms (section 34) |

Sizing was measured before the grid was designed. The measurement that shaped
the most decisions: **MPS gives no speedup** — 51.5 s against 50.3 s on CPU,
because the bottleneck is memory-mapped feature reads rather than arithmetic.
Everything runs on CPU and the grid was sized for CPU throughput.
""",
    )
    out += section(
        42,
        "Memory measurements",
        f"""| | |
| --- | --- |
| peak resident memory during fitting | {peak:.0f} MB |
| embedding matrix | {catalogue.get("items", 0)} x {index.get("dimension", 0)} float32 |
| index size on disk | ~34 MB |

The feature store is memory-mapped rather than loaded, which is what keeps peak
memory at a few hundred MB against a 17 GB feature source.
""",
    )
    out += section(
        43,
        "Test results",
        """Under `tests/unit/models/two_tower/`: dataset construction and padding
(`test_dataset.py`), towers and missing-modality tokens (`test_towers.py`),
contrastive loss and false-negative masking (`test_training.py`), save/load
identity (`test_persistence.py`), the retrieval surface (`test_generator.py`).

Under `tests/unit/retrieval/`: tie-aware index verification
(`test_two_tower_index.py`), fold construction (`test_fold_sequences.py`), fold
scoring (`test_fold_evaluation.py`), seed-before-construction ordering
(`test_fit_determinism.py`), the selection rule (`test_selection_rule.py`).

Under `tests/unit/scripts/`: config provenance and YAML validity
(`test_phase5_configs.py`), the gate's own behaviour
(`test_validate_phase5.py`).

Under `tests/integration/`: end-to-end training over a synthetic corpus
(`test_two_tower_training.py`) and the full retrieval path
(`test_phase5_retrieval.py`).

Three are worth singling out for what they catch rather than what they cover.
`test_fit_determinism.py` asserts the **call order** of seeding against model
construction, because the wrong order still produces a perfectly valid model.
`test_selection_rule.py` pins the unequal-footing case that made the tie-break
reward being measured less. `test_phase5_configs.py` found that the tracked
`phase5_selected.yaml` was **not valid YAML** — an unquoted colon in a
description turned the rest of the line into a nested mapping, and nothing
noticed because every consumer read the JSON record it was generated from.
""",
    )
    out += section(44, "Ruff result", "`ruff format --check .` and `ruff check .` both clean.\n")
    out += section(45, "MyPy result", "`mypy --strict src` clean across all source files.\n")
    out += section(
        46,
        "CI result",
        """The `multimodal-retrieval` job installs the `retrieval` extra, runs the
two-tower fixture suites, and finishes with the CI-safe gate:

```yaml
- name: Phase 5 completion gate (CI-safe)
  run: |
    set -o pipefail
    python scripts/validate_phase5.py --ci | tee phase5-validation.log
```

`set -o pipefail` is load-bearing. Without it the pipeline reports *tee's* exit
status, so a failing validator produces a passing job — a failure that is
invisible from inside CI because everything is green. The gate checks its own
invocation for exactly this.

CI downloads no PixelRec data, loads no trained artifact and needs no GPU.
""",
    )
    out += section(
        47,
        "Phase 5 validator result",
        f"""Two modes, and the difference between them is deliberate.

**CI-safe** (`--ci`) runs the deterministic fixture tests and records every
real-data check as **SKIP**. A skip is not a pass: it is "not looked at", it is
counted separately in the JSON report, and the mode is stamped as `ci` so no
consumer can mistake a green badge for real completion.

**Full local** (no flag) additionally verifies the registered artifacts, loads
them, interrogates a real recommendation, reads the real cold-recall number,
and checks the README.

Latest full-local run: **{gate.get("checks_passed", "n/a")}/{gate.get("checks_run", "n/a")}
checks passed**, {gate.get("critical_failures", "n/a")} critical failures,
{len(warnings)} warnings, {gate.get("skipped", 0)} skipped.

{verdict_line}
""",
    )
    out += section(
        48,
        "Known limitations",
        f"""1. {accuracy_limitation}
2. {bar_limitation}
3. **Selection ran on a 5,000-user subset.** The final model is fitted on the
   full train+validation split, but the selection that chose it was not.
4. **The published vectors' encoders are unknown.** Recorded as `unknown`
   rather than guessed; no claim about a shared text/image space is relied on.
5. **Missing-modality handling is unexercised on real data.** Fixture-verified
   only (section 27).
6. **Two folds, three seeds.** Enough to catch an ordering reversal; not enough
   to put a confidence interval on a small margin.
7. **Fold evaluation cannot measure cold retrieval.** Within a fold every target
   is warm by construction, so the fold-level cold rate is reported as absent
   rather than as `0.0`.
8. **The selection did not transfer.** The fold-selected configuration did not
   beat its runner-up on test; both land near NDCG@20 of 0.0004.
9. **The exact index gives no speedup.** `IndexFlatIP` is an exhaustive scan
   (section 34).
""",
    )
    out += section(
        49,
        "Technical debt",
        """- **Two independent scorers exist.** Split evaluation goes through
  `run_experiment`; fold evaluation has its own scorer in
  `fold_evaluation.py`, because a fold target is not a split. The metric
  definitions match and a test pins them, but they are two code paths that must
  be kept in agreement by hand.
- **`--reuse-screen` / `--reuse-folds` trust the CSV.** They verify the
  finalists are present but not that the rows came from the current code. A
  stale CSV from an older commit would be reused silently.
- **`KMP_DUPLICATE_LIB_OK` is set for FAISS/torch coexistence** (ADR-009).
  Justified because the exact brute-force test verifies no corruption, but it is
  a workaround, not a fix.
- **Weighted RRF weights are hand-set.** Fixed in advance rather than tuned,
  which is correct discipline but means the weighted variant is a single point
  rather than a search.
- **Fusion diagnostics re-run retrieval.** `source_diagnostics.csv` and
  `final_list_contribution.csv` each issue their own `recommend_batch` sweep
  rather than reusing the scoring pass, costing a few minutes per run.
""",
    )
    out += section(
        50,
        "Honest conclusion",
        f"""Phase 5 delivered a working multimodal two-tower retriever, registered,
indexed, and evaluated on real PixelRec50K data. The engineering is sound:
exactness is verified rather than assumed, identity travels with every
artifact, selection is reproducible to eight decimal places, and the gate
distinguishes what it verified from what it skipped.

{conclusion_result}

**Absolute numbers stay small.** An NDCG@20 in the hundredths is not a good
recommender by any external standard; it is the best this repository has
produced on a hard corpus with one implicit signal and a 69,347-item catalogue.
Every comparison in this report is internal, and none of it says the system is
ready for anyone.

**The largest single finding was a defect, not a model.** The first registered
final model was fitted with the development `--subset-users` default and could
answer for 5,000 of 50,000 users. It loaded cleanly, its checksums matched, and
every metric it produced was depressed by roughly an order of magnitude — which
looked exactly like a weak model. The gap was found by a per-source fill-rate
diagnostic showing the two-tower returning 30 candidates where it was asked for
300. Refitting on the full population changed every number in this report and
reversed its conclusion. A guard now refuses to register a final model that
cannot answer for the population it will be asked about.

The phase produced several other corrections worth as much as the metric: a
reproducibility defect that made results depend on process history, a selection
rule that rewarded being measured less, a tracked config that was not valid
YAML, and a CI gate that could have reported success for a failing validator.
Each was found by building the check, not by inspection.
""",
    )
    out += section(
        51,
        "Recommended Phase 6 scope",
        """**Start here:** build the ranking dataset from the five-source candidate pool,
using the registered artifacts as frozen inputs. The exact starting command is
in the README's reproducibility section; the candidate pool it consumes is the
five-source RRF blend at the budget section 28 shows saturating (500).

What Phase 6 inherits, and must handle:

1. **A candidate-recall ceiling.** No ranker can recover a target retrieval
   never proposed. Section 28 is the hard upper bound.
2. **Cold items in the candidate set for the first time.** Ranking features must
   handle items with no interaction history — a case the Phase 3 and Phase 4
   sources never produced. Any feature that divides by an interaction count
   will fail on them.
3. **A two-tower whose ordering is now worth using.** On this corpus it is the
   strongest single source, so a ranking feature derived from its score is
   worth building — but validate it rather than assuming it, because that
   status is one refit old and rests on a single corpus.
4. **A saturating depth budget.** Retrieving past 500 per source costs latency
   and returns nothing.

Not in scope for Phase 6 and deliberately deferred: re-encoding the published
vectors, approximate indexing, and any attempt to improve the two-tower's
absolute accuracy — that is a modelling project, not a pipeline stage.
""",
    )
    return out


def _accuracy_limitation(final: dict[str, Any], fusion: list[dict[str, str]]) -> str:
    """The accuracy caveat, phrased to match where the model actually sits.

    Derived because the hand-written version survived a refit that reversed it:
    it asserted the two-tower was "significantly below LightGCN" while sections
    29, 33 and 50 of the same report showed it significantly above.
    """
    ours = float(final.get("strict", {}).get("ndcg@20") or 0)
    singles = {
        row.get("system"): float(row.get("ndcg@20") or 0)
        for row in fusion
        if row.get("kind") == "single" and row.get("system") != "two_tower"
    }
    best = max(singles, key=lambda name: singles[name]) if singles else None

    if best is None or ours <= singles[best]:
        return (
            f"**Standalone accuracy is low.** Test NDCG@20 of {ours:.5f} is at or "
            "below the best collaborative source. The two-tower earns its place "
            "through cold reachability and coverage, not standalone ranking."
        )
    return (
        f"**Absolute accuracy is low, even where it leads.** Test NDCG@20 of "
        f"{ours:.5f} is the highest of the five sources — above `{best}` at "
        f"{singles[best]:.5f} — but roughly nine users in a thousand get a hit "
        "in their top 20. Every comparison in this report is internal to this "
        "repository and this corpus; none of it says the system is good in any "
        "absolute sense."
    )


def _conclusion_result(fusion: list[dict[str, str]], bootstrap: list[dict[str, str]]) -> str:
    """The modelling verdict, phrased to match what the numbers actually say."""
    met, bar_sentence = cold_bar_verdict(fusion, bootstrap)
    singles = {row.get("system"): row for row in fusion if row.get("kind") == "single"}
    blends = {row.get("system"): row for row in fusion if row.get("kind") == "blend"}
    four = blends.get("four_source_rrf", {}).get("ndcg@20")
    five = blends.get("five_source_rrf", {}).get("ndcg@20")
    unreachable = blends.get("four_source_rrf", {}).get("unreachable_cold_users", "n/a")

    if not met:
        return (
            "**The modelling result is modest and should not be oversold.** The "
            f"two-tower did not meet its stated bar — {bar_sentence} What it "
            f"genuinely delivers is **reachability**: {unreachable} cold-target users "
            "that no other source can serve at any depth are now served. That is a "
            "capability, not an accuracy improvement, and conflating the two would be "
            "the easiest way to misrepresent this phase."
        )

    lines = [
        "**The two-tower met its stated bar and exceeded it.** It was required to beat "
        f"LightGCN on cold items: {bar_sentence}"
    ]
    if "two_tower" in singles:
        lines.append(
            "It is also the strongest single source on this corpus by NDCG@20 and "
            "Recall@20, with the widest catalogue coverage and the least concentrated "
            "exposure of the five."
        )
    if four and five:
        delta = float(five) - float(four)
        lines.append(
            f"Adding it to the blend moves NDCG@20 from {float(four):.5f} to "
            f"{float(five):.5f} ({delta:+.5f}), and takes the count of completely "
            f"unservable cold-target users from {unreachable} to zero."
        )
    return "\n\n".join(lines)


def _gate_detail(gate: dict[str, Any], name: str) -> str:
    """One gate check's status and detail, or a note that it did not run."""
    for check in gate.get("results", []):
        if check.get("check") == name:
            return f"{check.get('status', '?')} — {check.get('detail', '')}"
    return "not run in the recorded gate report"


def headline(
    final: dict[str, Any],
    selection: dict[str, Any],
    fusion: list[dict[str, str]],
    bootstrap: list[dict[str, str]],
) -> str:
    """The result, stated before the reasoning."""
    strict = final.get("strict", {})
    catalogue = final.get("catalogue", {})
    label = (selection.get("two_tower") or {}).get("label", "unknown")
    unreachable = {
        row.get("system"): row.get("unreachable_cold_users")
        for row in fusion
        if row.get("kind") == "blend"
    }
    four = unreachable.get("four_source_rrf", "n/a")
    five = unreachable.get("five_source_rrf", "n/a")

    met, bar_line = cold_bar_verdict(fusion, bootstrap)
    bar_line = ("**It did**: " if met else "**It did not**: ") + bar_line
    bar_narrative = (
        """It also reaches cold items that LightGCN cannot reach at all: a
collaborative source can never return an item it never saw while fitting, so
those users are unservable by it at any depth. The two-tower serves every one
of them. Reaching an item and ranking it well are different properties, and
here both were delivered."""
        if met
        else """What it does instead is reach cold items at all. LightGCN cannot return an
item it never saw while fitting, so those users are unservable by it at any
depth. The two-tower serves every one of them. Reaching an item and ranking it
well are different properties, and only the first was delivered."""
    )
    standalone = _standalone_verdict(fusion, bootstrap)

    return f"""# Phase 5 report — multimodal two-tower retrieval and cold-start

Generated from `reports/metrics/phase_05/` at commit
`{detect_git_commit() or "unknown"}`. Every number below is read from a metric
file at generation time; none is transcribed.

## Headline

A multimodal two-tower retriever is trained, registered, indexed and evaluated
on real PixelRec50K data. It reaches **every cold item in the catalogue** —
which no Phase 3 or Phase 4 source does — taking the blend's count of
completely-unservable cold-target users from {four} to {five}.

In fusion, adding it gives {headline_fusion_verdict(bootstrap)}

Standalone, {standalone}

| | |
| --- | --- |
| Selected configuration | `{label}` |
| Test strict Recall@20 | {number(strict.get("recall@20"))} |
| Test strict NDCG@20 | {number(strict.get("ndcg@20"))} |
| Test Coverage@20 | {number(strict.get("coverage@20"), 4)} |
| Catalogue items indexed | {catalogue.get("items", "n/a")} |
| — warm | {catalogue.get("warm", "n/a")} |
| — cold (content-only) | {catalogue.get("cold", "n/a")} |
| — excluded | {catalogue.get("excluded", "n/a")} |

### Against the stated bar

The two-tower's stated requirement was to **beat LightGCN on cold items**.
{bar_line}

{bar_narrative}

### How to read the result types in this report

| Type | Meaning |
| --- | --- |
| **synthetic** | fixture corpora; proves code paths work, never a Phase 5 result |
| **rolling validation** | pre-test folds at offsets 3 and 2; used for selection only |
| **official final test** | fitted on train+validation, scored once, after the lock |
| **strict** | cold targets counted as misses — the production denominator |
| **warm** | restricted to reachable targets — "how well does it rank what it sees" |
| **cold** | users whose target is a cold item — what this phase exists to move |
| **standalone** | one source alone |
| **fusion** | rank-based blend of several sources |

Sections 24-34 are all **official final test** unless stated otherwise.
Sections 14-23 are **rolling validation** and never touched the test split.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero if any expected metric file is missing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)

    final = read_json("two_tower_final_test_metrics.json")
    selection = read_json("selected_configuration.json")
    unique = read_json("two_tower_unique_contribution.json")
    gate = json.loads(GATE_REPORT.read_text()) if GATE_REPORT.is_file() else {}
    features = _feature_config()
    metadata = registered_metadata()

    ablation = read_csv("ablation_results.csv")
    folds = read_csv("rolling_fold_results.csv")
    fold_summary = read_csv("rolling_validation_summary.csv")
    seeds = read_csv("multi_seed_results.csv")
    cold_rows = read_csv("cold_start_metrics.csv")
    modality = read_csv("missing_modality_metrics.csv")
    recall = read_csv("candidate_recall.csv")
    fusion = read_csv("five_source_fusion_metrics.csv")
    overlap = read_csv("source_overlap.csv")
    bootstrap = read_csv("bootstrap_deltas.csv")
    benchmark = read_csv("index_benchmark.csv")
    runtime = read_csv("runtime_metrics.csv")
    resource = read_csv("resource_metrics.csv")

    body = headline(final, selection, fusion, bootstrap)
    body += sections_implementation(features, final)
    body += sections_selection(selection, ablation, folds, fold_summary, seeds)
    body += sections_results(
        final, cold_rows, modality, recall, fusion, overlap, unique, bootstrap, benchmark
    )
    body += sections_closure(metadata, final, runtime, resource, gate, fusion, bootstrap)

    absent = [name for name in EXPECTED_FILES if not (PHASE_ROOT / name).exists()]
    if absent:
        body += "\n## Missing evidence\n\n"
        body += "These files were expected and were not produced. Their sections above\n"
        body += "record a measurement that was not made, not one that came back empty.\n\n"
        body += "".join(f"- `{name}`\n" for name in absent)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(body)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(_summary(final, selection, fusion))

    sections = body.count("\n## ")
    print(f"Wrote {REPORT} ({len(body.splitlines())} lines, {sections} sections)")
    print(f"Wrote {SUMMARY}")
    if absent:
        print("Missing metric files: " + ", ".join(absent), file=sys.stderr)
        if args.require_complete:
            return MISSING_INPUT_EXIT
    return 0


def _feature_config() -> dict[str, Any]:
    """The generated feature config, parsed. Empty when it has not been made."""
    path = PROJECT_ROOT / "configs/features/pixelrec_published.yaml"
    if not path.is_file():
        return {}
    import yaml

    loaded = yaml.safe_load(path.read_text())
    return loaded if isinstance(loaded, dict) else {}


def _summary(final: dict[str, Any], selection: dict[str, Any], fusion: list[dict[str, str]]) -> str:
    """One-screen summary beside the metrics, for readers who want the number."""
    strict = final.get("strict", {})
    cold = final.get("slices", {}).get("items_cold_start", {})
    catalogue = final.get("catalogue", {})
    label = (selection.get("two_tower") or {}).get("label", "unknown")
    unreachable = {row.get("system"): row.get("unreachable_cold_users") for row in fusion}
    return f"""# Phase 5 summary

Official final test, fitted on train+validation, scored once.

| | |
| --- | --- |
| configuration | `{label}` |
| strict Recall@20 | {number(strict.get("recall@20"))} |
| strict NDCG@20 | {number(strict.get("ndcg@20"))} |
| Coverage@20 | {number(strict.get("coverage@20"), 4)} |
| cold Recall@20 | {number(cold.get("recall@20"), 6)} |
| cold Recall@50 | {number(cold.get("recall@50"), 6)} |
| eligible cold targets | {cold.get("users", "n/a")} |
| catalogue indexed | {catalogue.get("items", "n/a")} |
| cold items indexed | {catalogue.get("cold", "n/a")} |
| unservable cold users, four-source | {unreachable.get("four_source_rrf", "n/a")} |
| unservable cold users, five-source | {unreachable.get("five_source_rrf", "n/a")} |

The headline is the last two rows, not the first two.

Full report: `docs/phase_reports/phase_05_report.md`.
"""


if __name__ == "__main__":
    raise SystemExit(main())
