#!/usr/bin/env python
"""Write a tracked provenance record for a phase's locked configuration.

    python scripts/record_selection_provenance.py --phase 4
    python scripts/record_selection_provenance.py --phase 4 --check

`reports/metrics/` is gitignored, so the selection records that
`generate_selected_config.py --check` and `generate_phase4_config.py --check`
compare against are not in the repository. That made those gates unrunnable in
CI: the generated YAML was tracked while its authority was not, so drift between
them could not be detected by anything except a developer remembering to look.

This writes the small, licence-safe half of the record to a tracked path, so the
drift check has something to verify against in CI. It contains hyperparameters,
aggregate metrics and identity hashes only -- no per-user recommendations, no
per-item rows, nothing derived from PixelRec content. See the dataset licence in
`docs/data/pixelrec50k_overview.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnirank.artifacts.metadata import detect_git_commit

MISSING_SOURCE_EXIT = 2
DRIFT_EXIT = 3

PROVENANCE_DIR = Path("provenance")

#: Keys copied verbatim from a model's selection block. Anything not listed is
#: dropped, which is what keeps dataset-derived rows out by construction rather
#: than by review.
METRIC_PREFIX = "validation_"


def _source_for(phase: int) -> Path:
    return Path(f"reports/metrics/phase_{phase:02d}/selected_configuration.json")


def build_record(phase: int, selection: dict[str, Any], *, timestamp: str) -> dict[str, Any]:
    """Reduce a selection record to its trackable core."""
    models: dict[str, Any] = {}
    for name, block in sorted(selection.items()):
        if not isinstance(block, dict):
            continue
        models[name] = {
            "hyperparameters": {
                key: value
                for key, value in sorted(block.items())
                if not key.startswith(METRIC_PREFIX) and key != "inherited_from"
            },
            "validation_metrics": {
                key: value for key, value in sorted(block.items()) if key.startswith(METRIC_PREFIX)
            },
            **({"inherited_from": block["inherited_from"]} if "inherited_from" in block else {}),
        }

    identity = selection.get("dataset_identity", {})
    return {
        "phase": phase,
        "selection_timestamp": timestamp,
        "git_commit": detect_git_commit() or "unknown",
        "models": models,
        "dataset_identity": {
            key: identity.get(key)
            for key in (
                "dataset_name",
                "dataset_version",
                "split_strategy",
                "split_version",
                "mapping_version",
                "schema_version",
                "pipeline_version",
                "ordering_field",
                "dataset_manifest_sha256",
                "dataset_configuration_hash",
                "data_git_commit",
            )
        },
        "selected_by": selection.get("selected_by"),
        "fit_splits": selection.get("fit_splits"),
        "target_split": selection.get("target_split"),
        "search_notes": selection.get("search_notes"),
        "contains": (
            "Hyperparameters, aggregate validation metrics and identity hashes only. "
            "No per-user or per-item PixelRec-derived rows."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, required=True, choices=(3, 4, 5))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the tracked record differs from the selection record.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    source, destination = (
        _source_for(args.phase),
        PROVENANCE_DIR / f"phase_{args.phase:02d}_selection.json",
    )

    if args.check:
        if not destination.is_file():
            print(f"Missing tracked provenance record: {destination}", file=sys.stderr)
            return DRIFT_EXIT
        tracked = json.loads(destination.read_text())
        if not source.is_file():
            # The usual CI case: the record is gitignored and absent. The tracked
            # file is then the only authority and there is nothing to compare it
            # against, so its presence and shape are what get verified.
            required = {"phase", "models", "dataset_identity", "git_commit"}
            missing = sorted(required - set(tracked))
            if missing:
                print(f"{destination} is missing required keys: {missing}", file=sys.stderr)
                return DRIFT_EXIT
            print(f"{destination} is present and well-formed ({source} not available to diff).")
            return 0
        rebuilt = build_record(
            args.phase,
            json.loads(source.read_text()),
            timestamp=tracked.get("selection_timestamp", ""),
        )
        rebuilt["git_commit"] = tracked.get("git_commit", rebuilt["git_commit"])
        if rebuilt != tracked:
            print(
                f"{destination} is out of date with {source}.\n"
                f"Regenerate it with: python scripts/record_selection_provenance.py "
                f"--phase {args.phase}",
                file=sys.stderr,
            )
            return DRIFT_EXIT
        print(f"{destination} matches {source}.")
        return 0

    if not source.is_file():
        print(f"Selection record not found: {source}", file=sys.stderr)
        return MISSING_SOURCE_EXIT

    record = build_record(
        args.phase,
        json.loads(source.read_text()),
        timestamp=datetime.now(UTC).isoformat(),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {destination} from {source}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
