#!/usr/bin/env python
"""Regenerate `configs/models/phase4_selected.yaml` from the selection record.

    python scripts/generate_phase4_config.py [--check]

Same discipline as `generate_selected_config.py` does for Phase 3: the YAML is
*derived*, never hand-written. The authority is
`reports/metrics/phase_04/selected_configuration.json`, locked before any test
metric was read. Keeping the file generated means the config and the experiment
record cannot drift apart, and `--check` fails when they have.

Like the Phase 3 generator, `--check` is a local gate rather than a CI one:
`reports/metrics/` is gitignored, so the record it reads is not in the
repository for CI to compare against.

Kept separate from the Phase 3 generator rather than merged behind a flag: the
two phases select different models with different hyperparameter sets and
different provenance caveats, so a shared renderer would be a chain of
per-phase branches wearing a single function's name.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SELECTION_FILE = Path("reports/metrics/phase_04/selected_configuration.json")
OUTPUT_FILE = Path("configs/models/phase4_selected.yaml")

MISSING_SELECTION_EXIT = 2
DRIFT_EXIT = 3

#: Emitted in this order so the file is stable across regenerations.
LIGHTGCN_KEYS = (
    "embedding_dim",
    "num_layers",
    "learning_rate",
    "regularization",
    "batch_size",
    "negatives_per_positive",
    "max_epochs",
    "early_stopping_patience",
    "evaluation_user_batch_size",
    "seed",
)
SASREC_KEYS = (
    "maximum_sequence_length",
    "embedding_dim",
    "num_blocks",
    "num_heads",
    "dropout",
    "learning_rate",
    "batch_size",
    "negatives_per_positive",
    "max_epochs",
    "early_stopping_patience",
    "evaluation_user_batch_size",
    "seed",
)


def _strip_metrics(block: dict[str, Any]) -> dict[str, Any]:
    """Drop the provenance stored beside the hyperparameters."""
    return {
        key: value
        for key, value in block.items()
        if not key.startswith("validation_") and key != "inherited_from"
    }


def _emit(block: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    """Render one model's hyperparameters, in a fixed order."""
    lines = []
    for key in keys:
        if key not in block:
            continue
        value = block[key]
        lines.append(
            f"      {key}: {value:g}" if isinstance(value, float) else f"      {key}: {value}"
        )
    return lines


def render(selection: dict[str, Any]) -> str:
    """Render the YAML from a selection record."""
    lightgcn = selection["lightgcn"]
    sasrec = selection["sasrec"]

    lines = [
        "# ---------------------------------------------------------------------------",
        "# Phase 4 SELECTED configuration.",
        "#",
        "# Generated from reports/metrics/phase_04/selected_configuration.json, which",
        "# was locked before any test metric was read. This file records what was",
        "# actually chosen; the per-model files hold the search spaces. The layers are:",
        "#",
        "#   configs/models/retrieval.yaml        generic defaults - small and fast",
        "#   configs/models/phase3_selected.yaml  what Phase 3 selected on validation",
        "#   configs/models/{lightgcn,sasrec}.yaml    Phase 4 search spaces",
        "#   configs/models/phase4_selected.yaml  what Phase 4 selected on validation",
        "#",
        "# Consumed by `scripts/train.py --from-selection` and",
        "# `scripts/compare_aggregation.py`.",
        "#",
        "# DO NOT HAND-EDIT. Re-select, then run:",
        "#     python scripts/generate_phase4_config.py",
        "# ---------------------------------------------------------------------------",
        "models:",
        "  candidate_generators:",
        "    lightgcn:",
        "      enabled: false",
        "      phase: 4",
        "      top_k: 300",
        *_emit(_strip_metrics(lightgcn), LIGHTGCN_KEYS),
        "",
        "    sasrec:",
        "      enabled: false",
        "      phase: 4",
        "      top_k: 200",
        *_emit(_strip_metrics(sasrec), SASREC_KEYS),
        "",
        "# Validation metrics recorded at selection time, for provenance only. These",
        "# are NOT test metrics.",
        "#",
        f"#   lightgcn  validation ndcg@20 = {lightgcn['validation_ndcg@20']:.5f}",
        f"#   sasrec    validation ndcg@20 = {sasrec['validation_ndcg@20']:.5f}",
        "#",
        "# Both figures are the best of a search that was cut short by compute, not a",
        "# demonstrated optimum. LightGCN's NDCG was still rising at num_layers=3, and",
        "# SASRec's training loss was still falling at its epoch budget. See",
        "# docs/phase_reports/phase_04_report.md.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the tracked file differs from what the record implies.",
    )
    args = parser.parse_args(argv)

    if not SELECTION_FILE.is_file():
        print(
            f"Selection record not found: {SELECTION_FILE}\n"
            "Run `python scripts/compare_retrievers.py --stage selection --stage lock` first.",
            file=sys.stderr,
        )
        return MISSING_SELECTION_EXIT

    rendered = render(json.loads(SELECTION_FILE.read_text()))

    if args.check:
        current = OUTPUT_FILE.read_text() if OUTPUT_FILE.is_file() else ""
        if current != rendered:
            print(
                f"{OUTPUT_FILE} is out of date with {SELECTION_FILE}.\n"
                "Regenerate it with: python scripts/generate_phase4_config.py",
                file=sys.stderr,
            )
            return DRIFT_EXIT
        print(f"{OUTPUT_FILE} matches the selection record.")
        return 0

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(rendered)
    print(f"Wrote {OUTPUT_FILE} from {SELECTION_FILE}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
