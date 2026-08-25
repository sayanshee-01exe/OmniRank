#!/usr/bin/env python
"""Regenerate `configs/models/phase3_selected.yaml` from the selection record.

    python scripts/generate_selected_config.py [--check]

The YAML is *derived*, never hand-written: the authority is
`reports/metrics/phase_03/selected_configuration.json`, which was locked before
any test metric was read. Keeping the file generated means the config and the
experiment record cannot drift apart, and `--check` fails when they have.

`--check` is a local gate, not a CI one. `reports/metrics/` is gitignored, so
the selection record it reads is not in the repository and CI has nothing to
compare against. Run it before committing a configuration change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SELECTION_FILE = Path("reports/metrics/phase_03/selected_configuration.json")
OUTPUT_FILE = Path("configs/models/phase3_selected.yaml")

MISSING_SELECTION_EXIT = 2
DRIFT_EXIT = 3

#: Emitted in this order so the file is stable across regenerations.
MF_KEYS = (
    "embedding_dim",
    "learning_rate",
    "regularization",
    "batch_size",
    "epochs",
    "negatives_per_positive",
    "evaluation_user_batch_size",
    "seed",
)


def _strip_metrics(block: dict[str, Any]) -> dict[str, Any]:
    """Drop the validation metrics stored beside the hyperparameters."""
    return {key: value for key, value in block.items() if not key.startswith("validation_")}


def render(selection: dict[str, Any]) -> str:
    """Render the YAML from a selection record."""
    popularity = selection["popularity"]
    matrix_factorization = selection["matrix_factorization"]
    pop = _strip_metrics(popularity)
    mf = _strip_metrics(matrix_factorization)

    lines = [
        "# ---------------------------------------------------------------------------",
        "# Phase 3 SELECTED configuration.",
        "#",
        "# Generated from reports/metrics/phase_03/selected_configuration.json, which",
        "# was locked before any test metric was read. This file records what was",
        "# actually chosen; `retrieval.yaml` holds generic development defaults and is",
        "# a different thing entirely. The three layers are:",
        "#",
        "#   configs/models/retrieval.yaml        generic defaults - small and fast",
        "#   configs/models/phase3_selected.yaml  what Phase 3 selected on validation",
        "#   configs/models/{lightgcn,sasrec,aggregation,faiss}.yaml",
        "#                                        Phase 4 search space and defaults",
        "#",
        "# Consumed by `scripts/train.py --from-selection` and",
        "# `scripts/compare_retrievers.py`.",
        "#",
        "# DO NOT HAND-EDIT. Re-select, then run:",
        "#     python scripts/generate_selected_config.py",
        "# ---------------------------------------------------------------------------",
        "models:",
        "  candidate_generators:",
        "    popularity:",
        "      enabled: false",
        "      phase: 3",
        "      top_k: 200",
        f"      variant: {pop['variant']}",
        f"      half_life_days: {pop['half_life_days']:g}",
        "",
        "    matrix_factorization:",
        "      enabled: false",
        "      phase: 3",
        "      top_k: 200",
        "      implementation: bpr",
    ]
    for key in MF_KEYS:
        if key not in mf:
            continue
        value = mf[key]
        lines.append(
            f"      {key}: {value:g}" if isinstance(value, float) else f"      {key}: {value}"
        )

    lines += [
        "",
        "# Validation metrics recorded at selection time, for provenance only. These",
        "# are NOT test metrics: Phase 3's test benchmark reversed this ranking. See",
        "# docs/phase_reports/phase_03_report.md section 23.",
        "#",
        f"#   popularity            validation ndcg@20 = {popularity['validation_ndcg@20']:.5f}",
        "#   matrix_factorization  validation ndcg@20 = "
        f"{matrix_factorization['validation_ndcg@20']:.5f}",
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
            "Run `python scripts/compare_baselines.py --stage selection` first.",
            file=sys.stderr,
        )
        return MISSING_SELECTION_EXIT

    rendered = render(json.loads(SELECTION_FILE.read_text()))

    if args.check:
        current = OUTPUT_FILE.read_text() if OUTPUT_FILE.is_file() else ""
        if current != rendered:
            print(
                f"{OUTPUT_FILE} is out of date with {SELECTION_FILE}.\n"
                "Regenerate it with: python scripts/generate_selected_config.py",
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
