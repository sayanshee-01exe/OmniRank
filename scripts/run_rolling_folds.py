#!/usr/bin/env python
"""Rolling-fold and multi-seed verification for the two-tower configuration.

    python scripts/run_rolling_folds.py --stage folds
    python scripts/run_rolling_folds.py --stage multi-seed --seeds 42,43,44

Two properties a single train-to-validation boundary cannot establish:

**Fold stability.** One boundary measures one week. A configuration that wins at
both pre-test origins has a property; one that wins at a single origin may just
suit that origin. Each fold here builds its histories from its own pre-origin
interactions, so the folds are genuinely different training problems.

**Seed stability.** A margin smaller than the seed spread is not a margin. The
selected configuration is re-run at several seeds so its lead can be compared
against its own noise.

Offset 1 is the official test target; ``build_fold`` refuses it outright, so a
selection run cannot reach it even by mistake. The measurement itself lives in
:mod:`omnirank.retrieval.fold_evaluation`, shared with the ablation driver so
the two cannot drift into scoring the same model differently.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from omnirank.core.config import load_config
from omnirank.core.exceptions import ConfigurationError, OmniRankError
from omnirank.core.logging import configure_logging, get_logger, run_context
from omnirank.data.processed import load_processed_dataset
from omnirank.data.rolling import (
    build_rolling_validation,
    check_fold_integrity,
    check_no_reserved_offset_used,
)
from omnirank.evaluation.reporting import REPORT_ROOT, append_jsonl, write_csv, write_json
from omnirank.retrieval.fold_evaluation import evaluate_on_fold, summarise_folds

CONFIG_ERROR_EXIT = 2
RUN_ERROR_EXIT = 3

PHASE_ROOT = REPORT_ROOT.parent / "phase_05"
RUNS_FILE = PHASE_ROOT / "rolling_validation_runs.jsonl"
PROCESSED = Path("data/processed/pixelrec50k")
BASE_CONFIG = Path("configs/models/two_tower.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--data-config", default="configs/data/pixelrec50k.yaml")
    parser.add_argument("--stage", default="folds", choices=("folds", "multi-seed"))
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "mps"))
    parser.add_argument("--subset-users", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--finalists",
        default="full_no_user_id,text_image_tag",
        help="Ablation labels to confirm across folds. Multi-seed uses the first.",
    )
    parser.add_argument(
        "--seeds",
        default="42,43,44",
        help="Comma-separated seeds. Used by --stage multi-seed.",
    )
    parser.add_argument(
        "--fold-offsets",
        default="3,2",
        type=lambda value: [int(part) for part in value.split(",")],
        help="Rolling-fold target offsets. Offset 1 is the reserved test target.",
    )
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args(argv)


def build_folds(dataset: Any, offsets: list[int], logger: Any, run_id: str) -> Any:
    """Construct and check the pre-test rolling folds.

    Both checks are cheap and both failures are invisible at runtime: a fold
    that leaked future events trains a better-looking model, and a fold that
    reached offset 1 reports a test number as a validation one.
    """
    validation = build_rolling_validation(
        dataset.fit_interactions(("train", "validation")),
        target_offsets=tuple(offsets),
        dataset_identity=dataset.identity.to_dict(),
    )
    for fold in validation.folds:
        check_fold_integrity(fold)
    check_no_reserved_offset_used(validation)
    write_json(validation.manifest(), PHASE_ROOT / "rolling_fold_manifest.json")
    for fold in validation.folds:
        logger.info("rolling.fold_ready", run_id=run_id, **fold.describe())
    return validation


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    config_dir = Path(args.config_dir)
    profile = Path(args.data_config)
    with contextlib.suppress(ValueError):
        profile = profile.resolve().relative_to(config_dir.resolve())

    try:
        config = load_config(config_dir, data_profile=profile)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR_EXIT

    configure_logging(config.logging, force=True)
    logger = get_logger("omnirank.rolling")
    PHASE_ROOT.mkdir(parents=True, exist_ok=True)
    base_config = yaml.safe_load(BASE_CONFIG.read_text())["two_tower"]

    with run_context(stage="rolling_folds") as run_id:
        try:
            dataset = load_processed_dataset(
                Path(config.data.dataset.processed_dir),
                Path(config.paths.mappings_dir) / config.data.dataset_name,
                verify_checksums=not args.skip_checksums,
            )
            validation = build_folds(dataset, args.fold_offsets, logger, run_id)
        except OmniRankError as exc:
            logger.error("rolling.setup_failed", run_id=run_id, reason=str(exc))
            return RUN_ERROR_EXIT

        labels = [name.strip() for name in args.finalists.split(",") if name.strip()]
        multi_seed = args.stage == "multi-seed"
        seeds = [int(value) for value in args.seeds.split(",")] if multi_seed else [config.seed]
        if multi_seed:
            # One configuration, many seeds -- the point is the spread of a
            # single configuration, not a comparison between several.
            labels = labels[:1]

        rows: list[dict[str, Any]] = []
        for label in labels:
            for fold in validation.folds:
                for seed in seeds:
                    try:
                        record = evaluate_on_fold(
                            label,
                            fold,
                            seed,
                            dataset=dataset,
                            processed_root=PROCESSED,
                            base_config=base_config,
                            epochs=args.epochs,
                            device=args.device,
                            subset_users=args.subset_users,
                        )
                    except OmniRankError as exc:
                        logger.error(
                            "rolling.run_failed",
                            run_id=run_id,
                            label=label,
                            fold=fold.name,
                            seed=seed,
                            reason=str(exc),
                        )
                        return RUN_ERROR_EXIT
                    record |= {
                        "run_id": run_id,
                        "stage": args.stage,
                        "subset_users": args.subset_users,
                    }
                    append_jsonl(record, RUNS_FILE)
                    rows.append(record)
                    # Written every iteration: a run interrupted at hour two
                    # should still leave the folds it finished.
                    write_csv(
                        rows,
                        PHASE_ROOT
                        / ("multi_seed_results.csv" if multi_seed else "rolling_fold_results.csv"),
                    )

        if not rows:
            logger.error("rolling.no_runs", run_id=run_id)
            return RUN_ERROR_EXIT

        summary = summarise_folds(rows)
        write_csv(
            summary,
            PHASE_ROOT
            / ("multi_seed_summary.csv" if multi_seed else "rolling_validation_summary.csv"),
        )
        for entry in summary:
            logger.info("rolling.summary", run_id=run_id, **entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
