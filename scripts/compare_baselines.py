#!/usr/bin/env python
"""Run the full Phase 3 baseline comparison and write the experiment reports.

    python scripts/compare_baselines.py --config-dir configs \
        --data-config configs/data/pixelrec50k.yaml

Stages, in the order the discipline requires:

1. **Selection** - fit every candidate configuration on training only, evaluate
   on validation, log each trial to ``validation_runs.jsonl``.
2. **Lock** - write ``selected_configuration.json`` *before* any test data is
   read. The file is the commitment.
3. **Final** - refit the locked configurations from a clean initialisation on
   train + validation, and evaluate the test split **once**.
4. **Report** - comparison table, slice metrics, runtimes, confidence intervals,
   paired deltas, and a markdown summary.

``--stage`` runs one part; the default runs all of them in order. ``--stage final``
refuses to proceed without an existing selection file, so a test number can never
be produced before a configuration was locked.

This script does not retrain anything it was not asked to.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnirank.core.config import load_config
from omnirank.core.exceptions import ConfigurationError, OmniRankError
from omnirank.core.logging import configure_logging, get_logger, run_context
from omnirank.data.processed import load_processed_dataset
from omnirank.evaluation.bootstrap import paired_bootstrap_delta
from omnirank.evaluation.evaluator import PRIMARY_METRICS
from omnirank.evaluation.reporting import (
    REPORT_ROOT,
    append_jsonl,
    comparison_table,
    render_markdown_summary,
    runtime_table,
    slice_table,
    write_csv,
    write_json,
    write_text,
)
from omnirank.models.baselines.popularity import (
    GLOBAL_COUNT,
    TIME_DECAY,
    PopularityConfig,
)
from omnirank.models.baselines.runner import (
    MATRIX_FACTORIZATION,
    POPULARITY,
    boundary_for_stage,
    fit_bpr,
    fit_popularity,
    run_experiment,
)

CONFIG_ERROR_EXIT = 2
RUN_ERROR_EXIT = 3

SELECTION_FILE = REPORT_ROOT / "selected_configuration.json"
VALIDATION_RUNS_FILE = REPORT_ROOT / "validation_runs.jsonl"

#: Popularity grid. The four prescribed half-lives plus two longer ones, because
#: PixelRec50K training interactions have a median age of 485 days and span
#: 3,788 - a grid that stopped at 365 would not have bracketed the corpus.
POPULARITY_GRID = [
    PopularityConfig(GLOBAL_COUNT),
    *(
        PopularityConfig(TIME_DECAY, half_life_days=days)
        for days in (7.0, 30.0, 45.0, 90.0, 365.0, 730.0, 1825.0)
    ),
]


def bpr_grid(epochs: int) -> list[dict[str, Any]]:
    """A small, justified BPR grid rather than a full cartesian sweep.

    Six configurations covering both embedding sizes, both learning rates, both
    regularisation strengths, and both negative counts, without multiplying them
    out to sixteen runs that would take an hour to say the same thing.
    """
    return [
        {
            "embedding_dim": 32,
            "learning_rate": 0.005,
            "regularization": 1e-4,
            "negatives_per_positive": 1,
            "epochs": epochs,
        },
        {
            "embedding_dim": 64,
            "learning_rate": 0.005,
            "regularization": 1e-4,
            "negatives_per_positive": 1,
            "epochs": epochs,
        },
        {
            "embedding_dim": 64,
            "learning_rate": 0.001,
            "regularization": 1e-4,
            "negatives_per_positive": 1,
            "epochs": epochs,
        },
        {
            "embedding_dim": 64,
            "learning_rate": 0.005,
            "regularization": 1e-5,
            "negatives_per_positive": 1,
            "epochs": epochs,
        },
        {
            "embedding_dim": 64,
            "learning_rate": 0.005,
            "regularization": 1e-4,
            "negatives_per_positive": 3,
            "epochs": epochs,
        },
        {
            "embedding_dim": 128,
            "learning_rate": 0.005,
            "regularization": 1e-4,
            "negatives_per_positive": 1,
            "epochs": epochs,
        },
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Phase 3 baseline comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--data-config", default="configs/data/pixelrec50k.yaml")
    parser.add_argument("--stage", default="all", choices=("all", "selection", "final"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--epochs", type=int, default=30, help="BPR epoch budget.")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds for the selected BPR config.")
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument(
        "--skip-bpr", action="store_true", help="Popularity only. For a quick check without torch."
    )
    return parser.parse_args(argv)


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
    logger = get_logger("omnirank.compare")
    dataset_config = config.data.dataset
    if dataset_config is None:
        print("The selected profile declares no data.dataset block.", file=sys.stderr)
        return CONFIG_ERROR_EXIT

    with run_context(stage="compare_baselines") as run_id:
        try:
            dataset = load_processed_dataset(
                Path(dataset_config.processed_dir),
                Path(config.paths.mappings_dir) / config.data.dataset_name,
                verify_checksums=not args.skip_checksums,
            )
        except OmniRankError as exc:
            logger.error("compare.dataset_unavailable", run_id=run_id, reason=str(exc))
            return CONFIG_ERROR_EXIT

        if args.stage in ("all", "selection") and not _run_selection(
            dataset, config, args, logger, run_id
        ):
            return RUN_ERROR_EXIT

        if args.stage in ("all", "final"):
            if not SELECTION_FILE.is_file():
                logger.error(
                    "compare.no_selection",
                    run_id=run_id,
                    detail=(
                        "No locked configuration found. Run --stage selection first; "
                        "test data must never be read before a configuration is locked."
                    ),
                    expected=str(SELECTION_FILE),
                )
                return RUN_ERROR_EXIT
            if not _run_final(dataset, config, args, logger, run_id):
                return RUN_ERROR_EXIT

    return 0


def _run_selection(
    dataset: Any, config: Any, args: argparse.Namespace, logger: Any, run_id: str
) -> bool:
    """Fit every candidate on train, score validation, and lock the winners."""
    fit_splits, target = boundary_for_stage("selection")
    VALIDATION_RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_RUNS_FILE.unlink(missing_ok=True)

    popularity_results = []
    for candidate in POPULARITY_GRID:
        model, fit_measurement = fit_popularity(dataset, fit_splits, candidate)
        result = run_experiment(
            model,
            dataset,
            config,
            model_name=POPULARITY,
            model_version=f"sel-{candidate.variant}-{candidate.half_life_days:g}",
            fit_splits=fit_splits,
            target_split=target,
            fit_measurement=fit_measurement,
            configuration=candidate.to_dict(),
            bootstrap=False,
        )
        append_jsonl(
            {
                "model": POPULARITY,
                "stage": "selection",
                **candidate.to_dict(),
                **{
                    k: round(v, 8)
                    for k, v in result.strict.flat().items()
                    if k in ("recall@20", "ndcg@20", "coverage@20")
                },
            },
            VALIDATION_RUNS_FILE,
        )
        popularity_results.append((candidate, result))
        logger.info(
            "compare.popularity_trial",
            run_id=run_id,
            variant=candidate.variant,
            half_life_days=candidate.half_life_days,
            **{k: round(result.strict.flat()[k], 6) for k in PRIMARY_METRICS},
        )

    best_popularity = max(popularity_results, key=lambda item: item[1].strict.flat()["ndcg@20"])
    selection: dict[str, Any] = {
        "selected_by": "validation ndcg@20, tie-broken by recall@20 then runtime",
        "target_split": target,
        "fit_splits": list(fit_splits),
        "dataset_identity": dataset.identity.to_dict(),
        "popularity": {
            **best_popularity[0].to_dict(),
            "validation_ndcg@20": round(best_popularity[1].strict.flat()["ndcg@20"], 8),
            "validation_recall@20": round(best_popularity[1].strict.flat()["recall@20"], 8),
        },
    }

    if not args.skip_bpr:
        from omnirank.models.baselines.bpr import BPRConfig

        bpr_results = []
        for bpr_candidate in bpr_grid(args.epochs):
            bpr_config = BPRConfig(
                **bpr_candidate,
                batch_size=8192,
                seed=config.seed,
                evaluation_user_batch_size=1024,
            )
            model, fit_measurement = fit_bpr(dataset, fit_splits, bpr_config, device=args.device)
            result = run_experiment(
                model,
                dataset,
                config,
                model_name=MATRIX_FACTORIZATION,
                model_version=f"sel-{bpr_config.label}",
                fit_splits=fit_splits,
                target_split=target,
                fit_measurement=fit_measurement,
                configuration=bpr_config.to_dict(),
                bootstrap=False,
            )
            append_jsonl(
                {
                    "model": MATRIX_FACTORIZATION,
                    "stage": "selection",
                    **bpr_config.to_dict(),
                    "final_loss": model.loss_history[-1],
                    **{
                        k: round(v, 8)
                        for k, v in result.strict.flat().items()
                        if k in ("recall@20", "ndcg@20", "coverage@20")
                    },
                },
                VALIDATION_RUNS_FILE,
            )
            bpr_results.append((bpr_config, result))
            logger.info(
                "compare.bpr_trial",
                run_id=run_id,
                config=bpr_config.label,
                **{k: round(result.strict.flat()[k], 6) for k in PRIMARY_METRICS},
            )

        best_bpr = max(bpr_results, key=lambda item: item[1].strict.flat()["ndcg@20"])
        selection["matrix_factorization"] = {
            **best_bpr[0].to_dict(),
            "validation_ndcg@20": round(best_bpr[1].strict.flat()["ndcg@20"], 8),
            "validation_recall@20": round(best_bpr[1].strict.flat()["recall@20"], 8),
        }

    write_json(selection, SELECTION_FILE)
    write_json(
        {
            "trials": [
                {
                    "configuration": candidate.to_dict(),
                    "metrics": {k: round(v, 8) for k, v in result.strict.flat().items()},
                }
                for candidate, result in popularity_results
            ]
        },
        REPORT_ROOT / "popularity_validation_metrics.json",
    )
    logger.info("compare.selection_locked", run_id=run_id, path=str(SELECTION_FILE))
    return True


def _run_final(
    dataset: Any, config: Any, args: argparse.Namespace, logger: Any, run_id: str
) -> bool:
    """Refit the locked configurations on train+validation and score test once."""
    selection = json.loads(SELECTION_FILE.read_text())
    fit_splits, target = boundary_for_stage("final")
    results = []

    locked_popularity: dict[str, Any] = selection["popularity"]
    popularity_config = PopularityConfig(
        variant=locked_popularity["variant"],
        half_life_days=locked_popularity["half_life_days"],
    )
    model, fit_measurement = fit_popularity(dataset, fit_splits, popularity_config)
    results.append(
        run_experiment(
            model,
            dataset,
            config,
            model_name=POPULARITY,
            model_version="phase3-final",
            fit_splits=fit_splits,
            target_split=target,
            fit_measurement=fit_measurement,
            configuration=popularity_config.to_dict(),
        )
    )

    if not args.skip_bpr and "matrix_factorization" in selection:
        from omnirank.models.baselines.bpr import BPRConfig

        locked = dict(selection["matrix_factorization"])
        for key in ("validation_ndcg@20", "validation_recall@20"):
            locked.pop(key, None)
        bpr_config = BPRConfig(**locked)
        model, fit_measurement = fit_bpr(dataset, fit_splits, bpr_config, device=args.device)
        results.append(
            run_experiment(
                model,
                dataset,
                config,
                model_name=MATRIX_FACTORIZATION,
                model_version="phase3-final",
                fit_splits=fit_splits,
                target_split=target,
                fit_measurement=fit_measurement,
                configuration=bpr_config.to_dict(),
            )
        )

    # Deterministic, anonymised examples plus explicit failures (never a
    # highlight reel): see runner.recommendation_examples.
    examples = {result.model_name: result.extra.get("examples", {}) for result in results}

    deltas = []
    if len(results) == 2:
        deltas = [
            paired_bootstrap_delta(
                results[1].strict.per_user,
                results[0].strict.per_user,
                metric,
                samples=config.evaluation.bootstrap.samples,
                confidence_level=config.evaluation.bootstrap.confidence_level,
                seed=config.evaluation.bootstrap.seed,
            )
            for metric in PRIMARY_METRICS
        ]

    write_json(
        {
            "results": [result.to_dict() for result in results],
            "paired_deltas": [interval.to_dict() for interval in deltas],
        },
        REPORT_ROOT / "final_test_metrics.json",
    )
    write_json(examples, REPORT_ROOT / "recommendation_examples.json")
    write_csv(comparison_table(results), REPORT_ROOT / "model_comparison.csv")
    write_csv(slice_table(results), REPORT_ROOT / "slice_metrics.csv")
    write_csv(runtime_table(results), REPORT_ROOT / "runtime_metrics.csv")
    write_text(
        render_markdown_summary(
            results,
            title="Phase 3 - final test results",
            deltas=deltas,
            notes=[
                "Full-catalogue evaluation; seen items excluded; users with no "
                "recommendations scored zero.",
                "Test data was evaluated once, after the configuration was locked in "
                "selected_configuration.json.",
                "Under one held-out item per user, recall@k == hit_rate@k, "
                "map@k == mrr@k, and precision@k == recall@k / k. They are not "
                "independent evidence.",
                "Embedding-based intra-list diversity is unavailable in this phase; "
                "category_diversity@k is reported instead.",
            ],
        ),
        REPORT_ROOT / "phase_03_summary.md",
    )
    logger.info(
        "compare.final_completed",
        run_id=run_id,
        models=[result.model_name for result in results],
        reports=str(REPORT_ROOT),
    )
    return True


if __name__ == "__main__":
    raise SystemExit(main())
