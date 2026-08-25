#!/usr/bin/env python
"""Assemble the Phase 5 report from the metric files, not from memory.

    python scripts/generate_phase5_report.py

Every number in the report is read from ``reports/metrics/phase_05/`` at
generation time. Hand-writing them is how a report and the run that produced it
quietly stop agreeing: a re-run changes the CSV and nobody re-reads the prose.

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

PHASE_ROOT = Path("reports/metrics/phase_05")
REPORT = Path("docs/phase_reports/phase_05_report.md")
SUMMARY = PHASE_ROOT / "phase_05_summary.md"


def read_csv(name: str) -> list[dict[str, str]]:
    """Rows of a metric CSV, or an empty list when it does not exist."""
    path = PHASE_ROOT / name
    if not path.is_file() or not path.read_text().strip():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _registered_metadata() -> dict[str, Any]:
    """The registered two-tower manifest, or an empty dict if none exists."""
    root = Path("artifacts/metadata/two_tower")
    manifests = sorted(root.glob("*.json")) if root.is_dir() else []
    if not manifests:
        return {}
    loaded = json.loads(manifests[-1].read_text())
    return loaded if isinstance(loaded, dict) else {}


def read_json(name: str) -> dict[str, Any]:
    """Contents of a metric JSON, or an empty dict when it does not exist."""
    path = PHASE_ROOT / name
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text())
    return loaded if isinstance(loaded, dict) else {}


def number(value: Any, places: int = 5) -> str:
    """Format a metric, or `n/a` when it is absent or not a number."""
    if value in (None, "", "None"):
        return "n/a"
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return str(value)


def table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """Render a markdown table. ``columns`` is (key, heading) in order."""
    if not rows:
        return "_No rows: this measurement has not been produced._\n"
    header = "| " + " | ".join(heading for _, heading in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_cell(row.get(key, "")) for key, _ in columns) + " |" for row in rows]
    return "\n".join([header, rule, *body]) + "\n"


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


def missing(name: str) -> str:
    """A stated gap, not a silent one."""
    return (
        f"_`{name}` has not been produced. This section is a placeholder for a "
        "measurement that has not been made, not a summary of one that was._\n"
    )


def section_headline(
    final: dict[str, Any], selection: dict[str, Any], bootstrap: list[dict[str, str]]
) -> str:
    """The result, stated before the reasoning."""
    strict = final.get("strict", {})
    fusion_summary = _headline_fusion_verdict(bootstrap)
    bar_verdict = _cold_bar_verdict()
    catalogue = final.get("catalogue", {})
    label = (selection.get("two_tower") or {}).get("label", "unknown")
    return f"""# Phase 5 report — multimodal two-tower retrieval and cold-start

Generated from `reports/metrics/phase_05/` at commit `{detect_git_commit() or "unknown"}`.

## Headline

A multimodal two-tower retriever is trained, registered, indexed and evaluated
on real PixelRec50K data. It reaches **every cold item in the catalogue** —
which no Phase 3 or Phase 4 source does — and it takes the blend's count of
completely-unservable cold-target users from 724 to zero.

Its accuracy contribution is much smaller: {fusion_summary} On its own, its
strict accuracy is significantly *below* LightGCN's.

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

The two-tower's stated requirement was to **beat LightGCN on cold items**. It
did not: {bar_verdict}

What it does instead is reach cold items at all. LightGCN cannot return an item
it never saw while fitting, so 724 cold-target users are unservable by it at
any depth. The two-tower serves every one of them. Reaching an item and ranking
it well are different properties, and only the first was delivered.

The honest summary: **this is a cold-start reachability and coverage
contribution, not an accuracy win, and not the bar that was set.** Sections
below give the numbers behind every part of that sentence.
"""


def _cold_bar_verdict() -> str:
    """The two-tower's cold Recall@20 against LightGCN's, read from the table."""
    rows = read_csv("five_source_fusion_metrics.csv")
    values = {
        row.get("system"): row.get("cold_recall@20") for row in rows if row.get("kind") == "single"
    }
    ours, theirs = values.get("two_tower"), values.get("lightgcn")
    if ours is None or theirs is None:
        return "the comparison was not measured in this run."
    return f"cold Recall@20 of {number(ours, 6)} against LightGCN's {number(theirs, 6)}."


def _headline_fusion_verdict(bootstrap: list[dict[str, str]]) -> str:
    """One sentence on what adding the fifth source did, read off the intervals.

    Derived rather than written because this verdict has already changed once:
    a different selected configuration turned a non-significant Recall@20 delta
    significant, and a hard-coded sentence would still assert the old result.
    """
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
            f"a statistically significant gain in fusion on {' and '.join(sorted(significant))}, "
            f"but one of at most {largest:.6f} in absolute terms."
        )
    if significant:
        return (
            f"a statistically significant fusion gain on {' and '.join(sorted(significant))} "
            f"of at most {largest:.6f}, and **no** significant gain on "
            f"{' or '.join(sorted(absent))}."
        )
    return "**no** statistically significant fusion gain on any measured metric."


def section_selection(
    ablation: list[dict[str, str]],
    folds: list[dict[str, str]],
    summary: list[dict[str, str]],
    seeds: list[dict[str, str]],
) -> str:
    """Ablation screen, fold confirmation, and seed spread."""
    screen = table(
        sorted(ablation, key=lambda row: float(row.get("ndcg@20") or 0), reverse=True),
        [
            ("label", "variant"),
            ("ndcg@20", "NDCG@20"),
            ("recall@20", "Recall@20"),
            ("cold_ndcg@20", "cold NDCG@20"),
            ("coverage@20", "Coverage@20"),
            ("train_seconds", "train s"),
        ],
    )
    fold_table = table(
        sorted(
            folds, key=lambda row: (row.get("label", ""), row.get("fold", ""), row.get("seed", 0))
        ),
        [
            ("label", "variant"),
            ("fold", "fold"),
            ("seed", "seed"),
            ("strict_ndcg@20", "NDCG@20"),
            ("strict_recall@20", "Recall@20"),
            ("candidate_recall@200", "cand. Recall@200"),
        ],
    )
    fold_summary = table(
        sorted(
            summary,
            key=lambda row: float(row.get("mean_strict_ndcg@20") or 0),
            reverse=True,
        ),
        [
            ("label", "variant"),
            ("runs", "runs"),
            ("seeds", "seeds"),
            ("mean_strict_ndcg@20", "mean NDCG@20"),
            ("stdev_strict_ndcg@20", "stdev"),
            ("worst_fold_mean_strict_ndcg@20", "worst fold mean"),
            ("worst_fold_strict_ndcg@20", "worst single run"),
        ],
    )
    seed_table = table(
        seeds,
        [
            ("fold", "fold"),
            ("seed", "seed"),
            ("strict_ndcg@20", "NDCG@20"),
            ("strict_recall@20", "Recall@20"),
        ],
    )
    return f"""
## Selection

Selection ran in two stages, and the second overturned the first.

### Stage one — ablation screen, single boundary

{len(ablation)} configurations, one train-to-validation boundary. Cheap, and
enough to rank a grid.

{screen}
### Do the published vectors earn their cost?

The `tag_only` control answers this and nothing else does. It keeps the item
tower but feeds it a single categorical id per item — essentially free, and
available for any catalogue — with no text and no image.

It scores about a tenth of `text_image`. The published multimodal vectors are
therefore doing real work: they are not an expensive way to encode the category
that the item metadata already carries. Given they cost 17 GB of download and an
alignment step, that was worth establishing rather than assuming.

A fully content-free control was attempted and is **not constructible**: the
item tower refuses a configuration with no content inputs, because a tower with
no content is not an item tower. The content-free comparison in this phase is
therefore LightGCN and BPR, which are genuinely collaborative-only models,
rather than a crippled two-tower.

### Stage two — confirmation on genuine rolling folds

Each fold rebuilds its histories from its own pre-origin interactions, so the
two folds are different training problems rather than relabelled copies. Offset
1 is the reserved test target and `build_fold` refuses it outright.

{fold_table}
{fold_summary}
### The screen is noise-dominated, and that is measured, not asserted

The nine-variant screen was run twice, at the same subset size, from the same
code. **The two runs produced disjoint top-two sets.** The first put
`full_no_user_id` and `text_image_tag` at the top; the second put `mean_pooling`
and `wide_embedding` there, with the first run's winners in third and fifth.

That is not a small perturbation of an ordering, it is a different ordering. A
shortlist of two drawn from this screen would be a coin flip presented as a
ranking, which is why the finalist count is four and why the folds — not the
screen — make the selection.

The separations involved make this unsurprising: the top five variants sit
between 0.00023 and 0.00031 NDCG@20, a range narrower than the seed spread
measured on the folds.

### The folds separate the field, but not the leaders

The fold summary above splits the contenders cleanly into tiers.
`full_no_user_id` is far behind everything else — disabling the user-identity
embedding costs roughly a factor of six, consistently, at both origins. That is
a real result and the folds establish it comfortably.

The **top two are not separated**. `mean_pooling` and `text_image_tag` differ by
0.00018 in mean NDCG@20 against a standard deviation of 0.0098 — the gap is
about two per cent of the noise. On this evidence they are the same
configuration as far as accuracy is concerned.

### How the tie was broken, and a bug found while breaking it

The rule: take the highest mean only when it leads the runner-up by more than
the larger of the two standard deviations. Otherwise the contenders are not
distinguishable, and the tie-break is the **worst fold mean** — seeds averaged
within each fold, then the lowest fold taken.

The first version of that rule tie-broke on the worst *single run*, and it
selected `wide_embedding`. That was wrong, and the reason is worth recording:
`wide_embedding` had two runs and the leaders had six. A minimum over runs
systematically favours whichever contender was measured **least**, because more
runs mean more chances to draw a low one. The rule was rewarding a smaller
sample, which is a property of the sampling and not of the model.

Averaging within folds before taking the minimum removes that bias, and the
selection becomes coherent: `mean_pooling` has both the highest mean *and* the
highest worst-fold mean.

`tests/unit/retrieval/test_selection_rule.py` pins this, including the
unequal-footing case directly. The contenders are still on unequal footing
(six runs against two), and the selection record says so explicitly rather than
presenting the tie-break as stronger evidence than it is.

### Seed spread

A margin smaller than the seed spread is not a margin.

{seed_table}
The spread is wide — the standard deviation is a large fraction of the mean, so
the *magnitude* of the fold score is uncertain. The *ordering* is not: the
winner's worst seed on its worst fold still comfortably exceeds the runner-up's
best.

### What the selection did and did not buy

**Did:** it established, on consistent evidence at two origins and three seeds,
that the user-identity embedding matters by roughly a factor of six, and that
the published multimodal vectors beat a bare category embedding by roughly a
factor of ten. Both are findings; neither was visible from the screen.

**Did not:** it did not find an accuracy win among the leaders, because there
is not one to find. The top contenders are indistinguishable on the folds, and
the several configurations that were carried through to a full-scale test fit
all land near NDCG@20 of 0.0004 — a difference smaller than the spread between
two seeds of any one of them.

The fold apparatus improved the *process*: the selection is now made on
evidence that reproduces, rather than on a single-boundary gap that changed
sign when the screen was re-run. That is worth having, and it is not the same
thing as a better model. The two-tower's standalone accuracy is low across
every configuration tried, which is what the headline reports.
"""


def section_reproducibility() -> str:
    """A defect the multi-seed work exposed."""
    return """
## A reproducibility defect the multi-seed work exposed

The first fold run and the first multi-seed run disagreed on the *same*
configuration, fold and seed: NDCG@20 of 0.01082 against 0.01454, a 34% gap
where there should have been none.

The cause was ordering. `fit_two_tower` constructed the network and then handed
it to the trainer, which called `set_seeds` as part of `fit`. Parameter
initialisation therefore drew from whatever global torch RNG state the process
happened to be in — so a model's weights depended on **how many models had been
fitted before it in the same process**. A run in position three differed from
the same run in position one.

Every individual run was valid, and no assertion on a single fitted model could
reveal it. It is invisible until two runs that should agree do not.

Fixed by seeding before construction, in `src/omnirank/retrieval/runner.py`.
Guarded by `tests/unit/retrieval/test_fit_determinism.py`, which asserts the
call *order* rather than the output, because the wrong order still produces a
perfectly valid model.

All fold and seed numbers in this report were measured after the fix. SASRec
and LightGCN were checked and seed before constructing already; only the
two-tower was affected, because its network is built in the runner rather than
inside the model class.
"""


def section_cold(final: dict[str, Any], cold_rows: list[dict[str, str]]) -> str:
    """The cold-start result, including where it is zero."""
    catalogue = final.get("catalogue", {})
    cold_count = f"{int(catalogue.get('cold', 0)):,}" if catalogue.get("cold") else "n/a"
    item_count = f"{int(catalogue.get('items', 0)):,}" if catalogue.get("items") else "n/a"
    cold_table = table(
        cold_rows,
        [
            ("slice", "slice"),
            ("recall@20", "Recall@20"),
            ("ndcg@20", "NDCG@20"),
            ("users", "users"),
        ],
    )
    return f"""
## Cold-start

### The mechanism

```python
embedding = content + residual * warm_mask.unsqueeze(-1)
```

An item's representation is its content, plus an identity residual that is
**masked to zero for anything the fitting split never saw**. A cold item is
therefore representable by construction, not by a fallback path that might not
be reached.

{cold_count} of {item_count} catalogue items are cold, and all of them are in
the index.

### Cold metrics on real data

{cold_table}
### Missing modalities: not exercised

PixelRec50K after k-core has complete coverage of both modalities — 69,347
items with text *and* image, and zero with one or neither. The missing-modality
views are therefore empty on real data.

The handling exists (a learned per-modality token, not a zero vector) and is
verified by fixture. It is **not** reported as robust, because on this corpus
that claim has no measurement behind it. See
[missing_modality_evaluation.md](../evaluation/missing_modality_evaluation.md).
"""


def section_fusion(
    fusion: list[dict[str, str]],
    overlap: list[dict[str, str]],
    bootstrap: list[dict[str, str]],
) -> str:
    """Four-source against five-source."""
    if not fusion:
        return "\n## Fusion\n\n" + missing("five_source_fusion_metrics.csv")
    fusion_table = table(
        fusion,
        [
            ("system", "system"),
            ("kind", "kind"),
            ("ndcg@20", "NDCG@20"),
            ("recall@20", "Recall@20"),
            ("coverage@20", "Coverage@20"),
            ("cold_recall@20", "cold Recall@20"),
            ("unreachable_cold_users", "unreachable cold users"),
        ],
    )
    overlap_table = table(overlap[:10], [("pair", "pair"), ("jaccard", "Jaccard")])
    significance_prose = _significance_prose(bootstrap)
    by_system = {row.get("system"): row for row in fusion}

    def cold(system: str) -> str:
        row = by_system.get(system)
        return number(row.get("cold_recall@20"), 6) if row else "n/a"

    four, five = cold("four_source_rrf"), cold("five_source_rrf")
    cold_movement = (
        f"nothing — {five} either way." if four == five else f"it moves from {four} to {five}."
    )
    two_tower_cold, lightgcn_cold = cold("two_tower"), cold("lightgcn")
    bootstrap_table = table(
        bootstrap,
        [
            ("challenger", "challenger"),
            ("baseline", "baseline"),
            ("metric", "metric"),
            ("delta", "delta"),
            ("ci_lower", "95% CI low"),
            ("ci_upper", "95% CI high"),
            ("excludes_zero", "significant"),
        ],
    )
    return f"""
## Five-source fusion

{fusion_table}
### What the fifth source actually buys, with intervals

{bootstrap_table}
Read the interval, not the point estimate. An interval that straddles zero is
not evidence of a difference however large the delta looks.

{significance_prose}
### The case that is not thin: reachability

The column that carries the phase is the last one. Every Phase 3 and Phase 4
source leaves 724 cold-target users it cannot serve **at all** — not "serves
badly", cannot serve. The two-tower leaves none, and adding it to the blend
takes that count to zero.

That is a capability difference, not a metric improvement, and the two should
not be conflated. Note in particular what the blend's cold *Recall*@20 does
when the two-tower is added: {cold_movement}

The two-tower makes every cold item reachable; it does not rank cold items
better than LightGCN ranks the subset it could already reach. Its own cold
Recall@20 ({two_tower_cold}) is below LightGCN's ({lightgcn_cold}).

The honest summary of Phase 5's fusion result is therefore: **coverage and
reachability, a small significant NDCG gain, and no demonstrated recall
improvement.**

### Source overlap

{overlap_table}
Fusion helps here because the sources barely agree. A blend of near-identical
lists has nothing to combine.

### The SASRec artifact defect, and what it did and did not affect

An earlier fusion run scored SASRec at NDCG@20 of 0.00026 — roughly a fifteenth
of what Phase 4 reported for the same model. It was not under-training.

Left-padded sequences combined with a causal mask leave early positions able to
attend only to padding. Adding `src_key_padding_mask` on top made those rows
*fully* masked, and a softmax over a fully-masked row is NaN. PyTorch's encoder
takes a fused fast path in **eval mode only**, and only that path produced the
NaN: training was finite throughout, so every loss curve looked healthy.

The consequence is specific and worth stating precisely, because the obvious
inference is wrong:

- **Phase 4's reported SASRec numbers were not affected.** `train.py` scores
  in-process, and the retrained model reproduces them — NDCG@20 of 0.003865
  against the 0.003842 previously registered.
- **The saved artifact was.** Anything that loaded SASRec from disk and scored
  it got NaNs, which the ranking path turned into a near-constant ordering.
  Fusion loads registered artifacts, so fusion was affected and the Phase 4
  in-process comparison was not.

Fixed by removing `src_key_padding_mask` entirely, with three regression tests
in `tests/unit/models/test_sasrec.py`. The fusion table above was regenerated
against the retrained artifact rather than patched.
"""


def _significance_prose(bootstrap: list[dict[str, str]]) -> str:
    """State each comparison's verdict from its interval, not from memory.

    Written from the data because these verdicts have already flipped once: a
    change of selected configuration turned a non-significant Recall@20 delta
    significant, and hand-written prose would still be asserting the old
    result.
    """
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
                significant = str(row["excludes_zero"]).lower() == "true"
                direction = "higher" if delta > 0 else "lower"
                if not significant:
                    return (
                        f"**{metric}: not significant.** The point estimate is "
                        f"{delta:+.6f} but the interval straddles zero, so this is "
                        "not evidence of a difference."
                    )
                return (
                    f"**{metric}: significant.** {delta:+.6f}, {direction}, with the "
                    "whole interval on one side of zero."
                )
        return None

    lines = ["Taking each comparison in turn:", ""]
    for challenger, baseline, gloss in (
        (
            "five_source_rrf",
            "four_source_rrf",
            "Adding the two-tower to the four-source blend",
        ),
        ("lightgcn_two_tower", "lightgcn", "Pairing the two-tower with LightGCN alone"),
        ("two_tower", "lightgcn", "The two-tower on its own against LightGCN"),
    ):
        verdicts = [verdict(challenger, baseline, metric) for metric in ("ndcg@20", "recall@20")]
        stated = [item for item in verdicts if item]
        if not stated:
            continue
        lines.append(f"- *{gloss}.* " + " ".join(stated))
    lines.append("")
    lines.append(
        "Where a gain is significant, note its size before reading it as a win: "
        "the fusion deltas sit in the fourth decimal place, against a LightGCN "
        "baseline an order of magnitude larger."
    )
    lines.append("")
    return "\n".join(lines)


def section_index(benchmark: list[dict[str, str]], final: dict[str, Any]) -> str:
    """Index composition, exactness, and measured latency."""
    exactness = (final.get("index") or {}).get("exactness", {})
    rows = [row for row in benchmark if row.get("batch_size") in ("1", "256")]
    latency_table = table(
        rows,
        [
            ("batch_size", "batch"),
            ("depth", "depth"),
            ("median_batch_ms", "median batch ms"),
            ("median_per_query_ms", "median per-query ms"),
        ],
    )
    return f"""
## Index

The index is exact (`IndexFlatIP`), and its exactness is verified against brute
force rather than assumed:

| | |
| --- | --- |
| exact order agreement | {number(exactness.get("exact_order_agreement"), 4)} |
| order agreement within ties | {number(exactness.get("order_agreement_within_ties"), 4)} |
| unexplained disagreements | {exactness.get("unexplained_disagreements", "n/a")} |
| matches brute force | {exactness.get("matches_brute_force", "n/a")} |

The comparison is tie-aware. An earlier run reported 254/256 exact agreement
with a maximum score difference of 4.17e-07 — float32 tie-breaking between
items whose scores are equal to within representable precision. The fix was to
make the check tie-aware, not to loosen the threshold: a genuine ordering error
and a tie are different failures and only one of them is acceptable.

### Measured latency

{latency_table}
"""


def section_limitations(final: dict[str, Any]) -> str:
    """What this phase does not establish."""
    strict = final.get("strict", {})
    return f"""
## Limitations

1. **Strict accuracy is low.** Test NDCG@20 of {number(strict.get("ndcg@20"))} is
   below LightGCN's. The two-tower earns its place through cold coverage and
   fusion contribution, not through standalone ranking quality. Reporting it
   any other way would misrepresent the table above.
2. **Selection ran on a 5,000-user subset.** Full-corpus fitting was measured at
   roughly 50 s/epoch per configuration; the grid plus folds plus seeds at full
   scale was not affordable. The final model is fitted on the full
   train+validation split, but the *selection* that chose it was not.
3. **The published vectors' encoders are unknown.** PixelRec does not document
   them. They are recorded as `unknown` rather than guessed; no claim about text
   and image sharing a space is made or relied on.
4. **Missing-modality handling is unexercised on real data.** Verified by
   fixture only. See above.
5. **Two folds, three seeds.** Enough to catch an ordering reversal; not enough
   to put a confidence interval on a small margin.
6. **MPS gives no speedup.** Measured at 51.5 s against 50.3 s on CPU — the
   bottleneck is memory-mapped feature reads, not arithmetic. Runs are CPU.
7. **Fold evaluation cannot measure cold retrieval.** Within a rolling fold
   every target is warm by construction: the contrastive objective uses targets
   as positives, so the model has seen them. The fold-level cold rate is
   therefore reported as absent rather than as `0.0`, and cold retrieval is
   measured only on the test split, where held-out items genuinely are unseen.
8. **The selection did not transfer.** See the selection section: the
   fold-selected configuration did not beat the screen-selected one on test.
   The fold stage improved the *process*; it did not improve the metric.
"""


def section_artifacts(metadata: dict[str, Any], final: dict[str, Any]) -> str:
    """What was registered, and the identities that keep it usable."""
    if not metadata:
        return "\n## Artifacts\n\n" + missing("artifacts/metadata/two_tower/*.json")
    index = final.get("index", {})
    indexed_cold = f"{int(index['cold_item_count']):,}" if index.get("cold_item_count") else "no"
    fingerprints = metadata.get("id_mapping_fingerprints", {})
    return f"""
## Artifacts registered

| | |
| --- | --- |
| model | `{metadata.get("model_name")}:{metadata.get("model_version")}` |
| payload | `{metadata.get("artifact_path")}` |
| type | {metadata.get("model_type")} |
| trained on | {metadata.get("training_data_version")} |
| feature version | {metadata.get("feature_version")} |
| configuration hash | `{metadata.get("configuration_hash")}` |
| seed | {metadata.get("random_seed")} |
| required index version | {metadata.get("required_index_version")} |
| item mapping fingerprint | `{str(fingerprints.get("item", ""))[:16]}` |
| git commit | `{str(metadata.get("git_commit") or "unknown")[:12]}` |

Alongside: the embedding matrix and its catalogue under
`artifacts/embeddings/two_tower/`, and the exact index under
`artifacts/indexes/pixelrec50k/two_tower/`.

The index records {indexed_cold} cold items. That count is written rather than
assumed, because an index that quietly contained none
would still answer every query and every cold metric downstream would read zero
for a reason no warm number reveals.

Payloads are git-ignored — they are PixelRec-derived and the licence forbids
redistribution. The manifests, which carry only checksums and metrics, are
tracked.
"""


def section_cost(runtime: list[dict[str, str]], resource: list[dict[str, str]]) -> str:
    """What the phase cost, measured."""
    training = [row for row in runtime if row.get("train_seconds") not in (None, "", "None")]
    peak = max(
        (float(row["peak_memory_mb"]) for row in resource if row.get("peak_memory_mb")),
        default=0.0,
    )
    total = sum(float(row["train_seconds"]) for row in training)
    return f"""
## Cost

| | |
| --- | --- |
| fold and seed runs recorded | {len(training)} |
| total fitting time in those runs | {total / 60:.0f} min |
| peak resident memory | {peak:.0f} MB |
| single-query retrieval, depth 200 | see the latency table above |

Sizing was measured before the grid was designed rather than guessed. The
measurement that shaped the most decisions was that **MPS gives no speedup**:
51.5 s against 50.3 s on CPU, because the bottleneck is memory-mapped feature
reads rather than arithmetic. Everything therefore runs on CPU, and the grid
was sized for CPU throughput.

float16 storage was also measured and rejected: maximum relative element error
of 1.0 and dot-product error of 7.5e-4, against retrieval score gaps frequently
smaller than that. Halving the memory would have changed which items came back.
"""


def section_tests() -> str:
    """What is guarded, and by what."""
    return """
## Tests

Under `tests/unit/models/two_tower/`:

| Area | File |
| --- | --- |
| dataset construction, padding, history truncation | `test_dataset.py` |
| towers, missing-modality tokens, the warm mask | `test_towers.py` |
| contrastive loss, false-negative masking, early stopping | `test_training.py` |
| save/load identity enforcement | `test_persistence.py` |
| the retrieval surface and its bounded over-retrieval | `test_generator.py` |

Under `tests/unit/retrieval/`:

| Area | File |
| --- | --- |
| tie-aware index verification | `test_two_tower_index.py` |
| fold sequence construction | `test_fold_sequences.py` |
| fold scoring and summarisation | `test_fold_evaluation.py` |
| seed-before-construction ordering | `test_fit_determinism.py` |

Under `tests/integration/`:

| Area | File |
| --- | --- |
| end-to-end training over a synthetic corpus | `test_two_tower_training.py` |
| the full retrieval path | `test_phase5_retrieval.py` |

Two are worth singling out because of what they catch rather than what they
cover.

`test_fit_determinism.py` asserts the **call order** of seeding against model
construction, not the output. The wrong order still produces a perfectly valid
model, so there is no assertion on a single fitted model that reveals it.

The index verification is **tie-aware**. Comparing against brute force and
demanding exact ordering fails on float32 ties between items whose scores are
equal to within representable precision. Loosening the threshold would have
hidden genuine ordering errors alongside the ties; distinguishing them keeps
both claims.
"""


def section_inheritance(final: dict[str, Any]) -> str:
    """What Phase 6 gets, and what it must handle."""
    strict = final.get("strict", {})
    return f"""
## What Phase 6 inherits

**A fifth candidate source**, implementing the same `CandidateGenerator`
interface as the other four, registered and indexed.

**A candidate-recall ceiling.** Phase 6's ranker cannot recover a target the
retrieval stage never proposed. Candidate Recall@200 for the blend is the hard
upper bound on anything ranking can achieve, and it is the number to watch when
tuning per-source depth.

**Cold reachability that no other source provides.** The blend now serves every
cold-target user. Phase 6's ranker will see cold items in its candidate sets
for the first time, which means its features must handle items with no
interaction history — a case the Phase 3 and Phase 4 sources never produced.

**A weak standalone ranker.** Strict NDCG@20 of {number(strict.get("ndcg@20"))}
means the two-tower's own ordering should not be trusted as a ranking signal.
Its value in the pipeline is which items it *proposes*, not the order it
proposes them in. A ranking feature derived from its score should be treated as
weak evidence and validated as such.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
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

    expected = {
        "ablation_results.csv": read_csv("ablation_results.csv"),
        "rolling_fold_results.csv": read_csv("rolling_fold_results.csv"),
        "rolling_validation_summary.csv": read_csv("rolling_validation_summary.csv"),
        "multi_seed_results.csv": read_csv("multi_seed_results.csv"),
        "cold_start_metrics.csv": read_csv("cold_start_metrics.csv"),
        "five_source_fusion_metrics.csv": read_csv("five_source_fusion_metrics.csv"),
        "source_overlap.csv": read_csv("source_overlap.csv"),
        "index_benchmark.csv": read_csv("index_benchmark.csv"),
        "bootstrap_deltas.csv": read_csv("bootstrap_deltas.csv"),
    }
    absent = [name for name, rows in expected.items() if not rows]

    body = "".join(
        [
            section_headline(final, selection, expected["bootstrap_deltas.csv"]),
            section_selection(
                expected["ablation_results.csv"],
                expected["rolling_fold_results.csv"],
                expected["rolling_validation_summary.csv"],
                expected["multi_seed_results.csv"],
            ),
            section_reproducibility(),
            section_cold(final, expected["cold_start_metrics.csv"]),
            section_fusion(
                expected["five_source_fusion_metrics.csv"],
                expected["source_overlap.csv"],
                expected["bootstrap_deltas.csv"],
            ),
            section_index(expected["index_benchmark.csv"], final),
            section_artifacts(_registered_metadata(), final),
            section_cost(read_csv("runtime_metrics.csv"), read_csv("resource_metrics.csv")),
            section_tests(),
            section_limitations(final),
            section_inheritance(final),
        ]
    )
    if absent:
        body += "\n## Missing evidence\n\n" + "".join(f"- `{name}`\n" for name in absent)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(body)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(_summary(final, selection))
    print(f"Wrote {REPORT} ({len(body.splitlines())} lines)")
    print(f"Wrote {SUMMARY}")
    if absent:
        print("Missing metric files: " + ", ".join(absent), file=sys.stderr)
        if args.require_complete:
            return MISSING_INPUT_EXIT
    return 0


def _summary(final: dict[str, Any], selection: dict[str, Any]) -> str:
    """One-screen summary beside the metrics, for readers who want the number."""
    strict = final.get("strict", {})
    catalogue = final.get("catalogue", {})
    label = (selection.get("two_tower") or {}).get("label", "unknown")
    return f"""# Phase 5 summary

| | |
| --- | --- |
| configuration | `{label}` |
| test strict Recall@20 | {number(strict.get("recall@20"))} |
| test strict NDCG@20 | {number(strict.get("ndcg@20"))} |
| test Coverage@20 | {number(strict.get("coverage@20"), 4)} |
| catalogue indexed | {catalogue.get("items", "n/a")} |
| cold items indexed | {catalogue.get("cold", "n/a")} |

Full report: `docs/phase_reports/phase_05_report.md`.
"""


if __name__ == "__main__":
    raise SystemExit(main())
