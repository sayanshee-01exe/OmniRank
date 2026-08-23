#!/usr/bin/env python
"""Evaluate a trained model offline.

    python scripts/evaluate.py --model popularity --version v1

**Phase 1 status: no metrics are implemented and no model exists to score.**
This script does one genuinely useful thing today: it looks the requested
artifact up in the real registry and reports whether it exists and whether it is
loadable on this host. That is the check that catches a device or index-version
mismatch (ADR-006) before a training run wastes an hour.

It never prints a metric. Fabricated benchmark numbers are worse than no numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnirank.artifacts.registry import ArtifactRegistry
from omnirank.core.config import load_config
from omnirank.core.exceptions import ArtifactError, ConfigurationError
from omnirank.core.logging import configure_logging, get_logger, run_context

NOT_IMPLEMENTED_EXIT = 3
ARTIFACT_MISSING_EXIT = 4


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Evaluate an OmniRank model offline.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--model", required=True, help="Registered model name.")
    parser.add_argument("--version", default=None, help="Model version; defaults to latest.")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config_dir)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging(config.logging, force=True)
    logger = get_logger("omnirank.evaluate")

    paths = config.paths.resolved(Path.cwd())
    registry = ArtifactRegistry(paths["metadata_dir"], artifact_root=paths["artifact_root"])

    with run_context(stage="evaluate", model=args.model) as run_id:
        try:
            metadata = (
                registry.get(args.model, args.version)
                if args.version
                else registry.latest(args.model)
            )
        except ArtifactError as exc:
            logger.error("evaluate.artifact_unavailable", run_id=run_id, reason=str(exc))
            return ARTIFACT_MISSING_EXIT

        device = config.device.resolve()
        compatible = metadata.is_compatible_with(
            device=device, index_version=config.models.index.index_version
        )
        logger.info(
            "evaluate.artifact_found",
            run_id=run_id,
            artifact=metadata.key,
            artifact_type=metadata.model_type.value,
            trained_on=metadata.training_data_version,
            feature_version=metadata.feature_version,
            host_device=device,
            compatible=compatible,
            recorded_metrics=sorted(metadata.metrics),
        )
        logger.error(
            "evaluate.not_implemented",
            run_id=run_id,
            planned_phase=2,
            detail=(
                "Evaluation metrics are a Phase 2 deliverable. The Evaluator contract "
                "is defined in omnirank/evaluation/base.py; no metric is computed here, "
                "and none is invented."
            ),
        )
    return NOT_IMPLEMENTED_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
