#!/usr/bin/env python
"""Prepare a raw dataset for training.

    python scripts/prepare_data.py --config-dir configs

**Phase 1 status: the pipeline this drives is not built.** What this script does
today is real and useful on its own - it loads and validates the configuration,
resolves and reports every path the pipeline will read and write, and confirms
the domain profile is coherent - then exits non-zero rather than pretending to
have produced a dataset.

Planned steps (Phase 2), in order:

1. Load raw records via ``omnirank.data.loaders.DatasetLoader``.
2. Validate with ``omnirank.data.validation.validate_batch``.
3. Clean, k-core filter, and build id mappings (``data.preprocessing``).
4. Split temporally (``data.splitting``), then assert ``check_split_integrity``.
5. Generate features and sequences (``omnirank.features``).
6. Write to ``data/processed/`` and register the mappings as artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnirank.core.config import load_config
from omnirank.core.exceptions import ConfigurationError
from omnirank.core.logging import configure_logging, get_logger, run_context

NOT_IMPLEMENTED_EXIT = 3


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Prepare a dataset for OmniRank training.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate configuration and report resolved paths, then exit 0.",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config_dir)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging(config.logging, force=True)
    logger = get_logger("omnirank.prepare_data")

    with run_context(stage="prepare_data") as run_id:
        paths = config.paths.resolved(Path.cwd())
        logger.info(
            "prepare_data.configuration_ok",
            run_id=run_id,
            domain=config.data.domain,
            dataset=f"{config.data.dataset_name}@{config.data.dataset_version}",
            event_types=sorted(config.data.event_types),
            positive_events=list(config.data.positive_event_types),
            split_strategy=config.data.splitting.strategy,
            raw_dir=str(paths["raw_dir"]),
            processed_dir=str(paths["processed_dir"]),
            mappings_dir=str(paths["mappings_dir"]),
            training_config_hash=config.training_config_hash[:16],
        )

        if args.check_only:
            return 0

        logger.error(
            "prepare_data.not_implemented",
            run_id=run_id,
            planned_phase=2,
            detail=(
                "The data pipeline is a Phase 2 deliverable. Configuration and paths "
                "above are valid. Re-run with --check-only to exit successfully."
            ),
        )
    return NOT_IMPLEMENTED_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
