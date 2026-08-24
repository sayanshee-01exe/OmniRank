#!/usr/bin/env python
"""Prepare a dataset for OmniRank training.

    python scripts/prepare_data.py --config configs/data/pixelrec50k.yaml

Runs the full pipeline: inspect, validate, profile, canonicalize, clean, filter,
map ids, split, derive collaborative/graph/sequential/metadata/feature datasets,
build evaluation slices, check leakage, write reports and the manifest.

Exit codes:

===  =======================================================================
0    success
2    configuration error, or a source file is missing
3    a pipeline stage failed (including a critical leakage check)
===  =======================================================================

Options that change what is produced are recorded in the dataset manifest, so a
subset run can never be mistaken for a full one.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnirank.core.config import load_config
from omnirank.core.exceptions import ConfigurationError, DataError, OmniRankError
from omnirank.core.logging import configure_logging, get_logger
from omnirank.data.pipeline import PipelineOptions, run_pipeline

CONFIG_ERROR_EXIT = 2
PIPELINE_ERROR_EXIT = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare a dataset for OmniRank training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/prepare_data.py --config configs/data/pixelrec50k.yaml\n"
            "  python scripts/prepare_data.py --config configs/data/pixelrec50k.yaml "
            "--subset-users 500\n"
            "  python scripts/prepare_data.py --config configs/data/pixelrec50k.yaml "
            "--validate-only\n"
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/data/pixelrec50k.yaml",
        help="Domain profile to use. Replaces the profile named in base.yaml.",
    )
    parser.add_argument("--config-dir", default="configs", help="Directory holding base.yaml.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check that source files exist and have the expected schema, then stop.",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Profile the raw data and write its reports, then stop before cleaning.",
    )
    parser.add_argument(
        "--subset-users",
        type=int,
        default=None,
        metavar="N",
        help="Development run over the first N users (whole histories, deterministic).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing processed outputs. Off by default.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)

    config_dir = Path(args.config_dir)
    profile = Path(args.config)
    # Accept both `configs/data/x.yaml` and `data/x.yaml`.
    if profile.is_absolute() or profile.exists():
        # A path outside config_dir passes through unchanged.
        with contextlib.suppress(ValueError):
            profile = profile.resolve().relative_to(config_dir.resolve())

    try:
        config = load_config(config_dir, data_profile=profile)
    except ConfigurationError as exc:
        # Printed rather than logged: logging is configured from the config that
        # just failed to load, and a stack trace would bury the actual problem.
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR_EXIT

    configure_logging(config.logging, force=True)
    logger = get_logger("omnirank.prepare_data")

    options = PipelineOptions(
        overwrite=args.overwrite,
        validate_only=args.validate_only,
        profile_only=args.profile_only,
        subset_users=args.subset_users,
    )

    try:
        result = run_pipeline(config, options)
    except DataError as exc:
        logger.error("prepare_data.pipeline_failed", reason=str(exc))
        return PIPELINE_ERROR_EXIT
    except OmniRankError as exc:
        logger.error("prepare_data.failed", error_code=exc.code, reason=str(exc))
        return CONFIG_ERROR_EXIT

    if args.validate_only:
        logger.info("prepare_data.validated", detail="source files present and well-formed")
        return 0
    if args.profile_only:
        logger.info("prepare_data.profiled", **result.counts)
        return 0

    logger.info(
        "prepare_data.completed",
        manifest=str(result.manifest_path),
        outputs=len(result.outputs),
        leakage_passed=result.leakage_passed,
        **result.counts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
