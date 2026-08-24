#!/usr/bin/env python
"""Evaluate a registered OmniRank model against a held-out split.

    python scripts/evaluate.py --model matrix_factorization \
        --version phase3-mf-selection --split validation --protocol full

    python scripts/evaluate.py --model popularity \
        --version phase3-popularity-final --split test --protocol full

Loads the registered artifact, verifies it is compatible with the current
dataset, reconstructs the correct fit-history boundary for the requested split,
generates recommendations, evaluates, and writes structured reports.

**The fit boundary follows the split, not a flag.** Evaluating the validation
split uses a train-only history; evaluating test uses train+validation. Allowing
those to be mixed on the command line would make it possible to produce an
optimistic number by accident.

Never prints a placeholder metric. Exit codes: 0 success · 2 configuration or
artifact error · 3 evaluation failure.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typing import Any

from omnirank.artifacts.registry import ArtifactRegistry
from omnirank.core.config import load_config
from omnirank.core.exceptions import ConfigurationError, OmniRankError
from omnirank.core.logging import configure_logging, get_logger, run_context
from omnirank.data.processed import load_processed_dataset
from omnirank.evaluation.reporting import REPORT_ROOT, write_json
from omnirank.models.baselines.popularity import PopularityRecommender
from omnirank.models.baselines.runner import (
    MATRIX_FACTORIZATION,
    POPULARITY,
    run_experiment,
)

CONFIG_ERROR_EXIT = 2
EVALUATION_ERROR_EXIT = 3

#: Evaluating a split implies the history a model may legitimately have seen.
BOUNDARY_FOR_SPLIT = {
    "validation": ("train",),
    "test": ("train", "validation"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a registered OmniRank model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True, choices=(POPULARITY, MATRIX_FACTORIZATION))
    parser.add_argument("--version", required=True)
    parser.add_argument("--split", default="validation", choices=tuple(BOUNDARY_FOR_SPLIT))
    parser.add_argument(
        "--protocol",
        default="full",
        choices=("full",),
        help="Only full-catalogue evaluation is supported for reported results.",
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--data-config", default="configs/data/pixelrec50k.yaml")
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "mps"))
    parser.add_argument("--output", default=None, help="Where to write the JSON report.")
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    config_dir = Path(args.config_dir)
    profile = Path(args.data_config)
    # A path outside config_dir is passed through unchanged.
    with contextlib.suppress(ValueError):
        profile = profile.resolve().relative_to(config_dir.resolve())

    try:
        config = load_config(config_dir, data_profile=profile)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR_EXIT

    configure_logging(config.logging, force=True)
    logger = get_logger("omnirank.evaluate")
    dataset_config = config.data.dataset
    if dataset_config is None:
        print("The selected profile declares no data.dataset block.", file=sys.stderr)
        return CONFIG_ERROR_EXIT

    with run_context(stage="evaluate", model=args.model, version=args.version) as run_id:
        registry = ArtifactRegistry(
            Path(config.paths.metadata_dir), artifact_root=Path(config.paths.artifact_root)
        )
        try:
            metadata = registry.get(args.model, args.version)
        except OmniRankError as exc:
            logger.error("evaluate.artifact_not_found", run_id=run_id, reason=str(exc))
            return CONFIG_ERROR_EXIT

        try:
            dataset = load_processed_dataset(
                Path(dataset_config.processed_dir),
                Path(config.paths.mappings_dir) / config.data.dataset_name,
                verify_checksums=not args.skip_checksums,
            )
        except OmniRankError as exc:
            logger.error("evaluate.dataset_unavailable", run_id=run_id, reason=str(exc))
            return CONFIG_ERROR_EXIT

        artifact_dir = Path(metadata.artifact_path or "")
        model: Any
        try:
            if args.model == POPULARITY:
                model = PopularityRecommender.load(artifact_dir)
            else:
                from omnirank.models.baselines.bpr import BPRMatrixFactorization

                model = BPRMatrixFactorization.load(artifact_dir, device=args.device)
            # A model paired with the wrong mapping resolves every recommended
            # id to a different item and fails silently. Check before scoring.
            model.require_mapping(dataset.mapping_metadata.get("item_mapping_checksum", ""))
        except OmniRankError as exc:
            logger.error("evaluate.artifact_unusable", run_id=run_id, reason=str(exc))
            return CONFIG_ERROR_EXIT

        fit_splits = BOUNDARY_FOR_SPLIT[args.split]
        try:
            result = run_experiment(
                model,
                dataset,
                config,
                model_name=args.model,
                model_version=args.version,
                fit_splits=fit_splits,
                target_split=args.split,
                fit_measurement=None,
                configuration=metadata.model_dump(mode="json")["metrics"],
            )
        except OmniRankError as exc:
            logger.error("evaluate.failed", run_id=run_id, reason=str(exc))
            return EVALUATION_ERROR_EXIT

        output = Path(
            args.output or REPORT_ROOT / f"{args.model}_{args.version}_{args.split}_metrics.json"
        )
        write_json(result.to_dict(), output)

        strict = result.strict.flat()
        logger.info(
            "evaluate.completed",
            run_id=run_id,
            model=args.model,
            version=args.version,
            split=args.split,
            protocol="full_catalogue",
            fit_splits="+".join(fit_splits),
            report=str(output),
            **{key: round(strict[key], 6) for key in ("recall@20", "ndcg@20") if key in strict},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
