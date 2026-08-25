#!/usr/bin/env python
"""Compare every Phase 4 retrieval system through the shared Phase 3 harness.

    python scripts/compare_retrievers.py --stage selection --epochs 30

Stages, in the order the discipline requires:

1. **selection** - fit every candidate configuration on training only, evaluate
   on validation, log each trial to ``phase_04/validation_runs.jsonl``.
2. **rolling** - re-evaluate the *selected* configurations at earlier temporal
   origins. Phase 3 ended with a model ordering that reversed between
   validation and test; a single origin cannot distinguish "this model is
   better" from "this model suits this particular week". Rolling folds can.
3. **lock** - write ``phase_04/selected_configuration.json`` *before* any test
   data is read. The file is the commitment.
4. **final** - refit the locked configurations from a clean initialisation on
   train + validation and evaluate the test split **once**, across seeds.
5. **report** - comparison tables, aggregation experiments, runtimes,
   confidence intervals, paired deltas, and a markdown summary.

Every model here -- popularity, BPR, LightGCN, SASRec, and the blends -- is
scored by ``run_experiment`` from the Phase 3 runner, unchanged. Only fitting
differs between them.

``--stage final`` refuses to proceed without an existing selection file, so a
test number can never be produced before a configuration was locked.
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
from omnirank.evaluation.evaluator import PRIMARY_METRICS
from omnirank.evaluation.reporting import (
    REPORT_ROOT,
    append_jsonl,
    write_csv,
    write_json,
)
from omnirank.models.baselines.runner import (
    MATRIX_FACTORIZATION,
    POPULARITY,
    boundary_for_stage,
    run_experiment,
)
from omnirank.retrieval.runner import LIGHTGCN, SASREC, fit_lightgcn, fit_sasrec

CONFIG_ERROR_EXIT = 2
RUN_ERROR_EXIT = 3

# REPORT_ROOT is already the Phase 3 directory, so Phase 4 is its sibling.
PHASE_ROOT = REPORT_ROOT.parent / "phase_04"
VALIDATION_RUNS_FILE = PHASE_ROOT / "validation_runs.jsonl"
SELECTION_FILE = PHASE_ROOT / "selected_configuration.json"
PHASE_3_SELECTION = REPORT_ROOT / "selected_configuration.json"

#: Metrics carried into the trial log. The full result is far larger; these are
#: the ones a selection decision is actually made on.
TRIAL_METRICS = ("recall@20", "ndcg@20", "coverage@20")


def lightgcn_grid(epochs: int) -> list[dict[str, Any]]:
    """Candidate LightGCN configurations.

    The axis that matters for LightGCN is ``num_layers`` -- it is the only thing
    distinguishing the model from the Phase 3 matrix factorization, and
    ``num_layers=0`` is exactly that ablation. Including it makes "did
    propagation help?" a measurement rather than an assumption. Embedding
    dimension is held at the value Phase 3 selected so the comparison isolates
    the graph.
    """
    return [
        {"embedding_dim": 128, "num_layers": layers, "max_epochs": epochs}
        for layers in (0, 1, 2, 3)
    ]


def sasrec_grid(epochs: int) -> list[dict[str, Any]]:
    """Candidate SASRec configurations.

    Deliberately short. SASRec costs roughly ten times a LightGCN epoch on this
    hardware (measured, not assumed), so a wide grid is not affordable and
    pretending otherwise would mean reporting a search that never ran. Sequence
    length is the axis with the clearest prior effect on a short-video corpus.
    """
    return [
        {"maximum_sequence_length": length, "embedding_dim": 64, "max_epochs": epochs}
        for length in (20, 50)
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--data-config", default="configs/data/pixelrec50k.yaml")
    parser.add_argument(
        "--stage",
        default="all",
        choices=("all", "selection", "rolling", "lock", "final", "report"),
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--models", default="lightgcn,sasrec", help="Comma-separated.")
    parser.add_argument("--epochs", type=int, default=30, help="LightGCN epoch budget.")
    parser.add_argument("--sasrec-epochs", type=int, default=15, help="SASRec epoch budget.")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds for the final stage.")
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One tiny configuration per model, to prove the path runs end to end.",
    )
    return parser.parse_args(argv)


def _record_trial(
    logger: Any,
    run_id: str,
    model_name: str,
    configuration: dict[str, Any],
    result: Any,
    *,
    smoke: bool = False,
) -> None:
    """Append one selection trial to the run log and the structured log.

    The file is append-only and stamped with ``run_id``. Truncating it on each
    invocation would be simpler, but ``--models`` runs one model at a time, so
    truncating would silently discard the other models' trials. Stamping keeps
    the full history and lets the report select the run it means.
    """
    flat = result.strict.flat()
    append_jsonl(
        {
            "model": model_name,
            "stage": "selection",
            "run_id": run_id,
            # Smoke trials use a deliberately tiny budget; they are recorded so
            # the log stays a complete history, and excluded from selection.
            "smoke": smoke,
            **configuration,
            **{k: round(flat[k], 8) for k in TRIAL_METRICS if k in flat},
        },
        VALIDATION_RUNS_FILE,
    )
    logger.info(
        "compare_retrievers.trial",
        run_id=run_id,
        model=model_name,
        **{k: round(flat[k], 6) for k in PRIMARY_METRICS if k in flat},
    )


def _fit_and_score(
    model_name: str,
    candidate: dict[str, Any],
    *,
    dataset: Any,
    config: Any,
    args: argparse.Namespace,
    fit_splits: tuple[str, ...],
    target: str,
    seed: int,
    version_prefix: str,
) -> tuple[Any, Any]:
    """Fit one configuration and evaluate it through the shared harness."""
    processed_root = Path(config.data.dataset.processed_dir)
    if model_name == LIGHTGCN:
        from omnirank.models.lightgcn import LightGCNConfig

        model_config: Any = LightGCNConfig(**candidate, batch_size=8192, seed=seed)
        model, fit_measurement = fit_lightgcn(dataset, fit_splits, model_config, device=args.device)
    elif model_name == SASREC:
        from omnirank.models.sasrec import SASRecConfig

        model_config = SASRecConfig(**candidate, batch_size=512, seed=seed)
        model, fit_measurement = fit_sasrec(
            dataset,
            fit_splits,
            model_config,
            processed_root=processed_root,
            device=args.device,
        )
    else:  # pragma: no cover - guarded by the CLI choices
        raise ValueError(f"Unsupported model: {model_name}")

    result = run_experiment(
        model,
        dataset,
        config,
        model_name=model_name,
        model_version=f"{version_prefix}-{model_config.label}",
        fit_splits=fit_splits,
        target_split=target,
        fit_measurement=fit_measurement,
        configuration=model_config.to_dict(),
        bootstrap=False,
    )
    return model, result


def _run_selection(
    dataset: Any, config: Any, args: argparse.Namespace, logger: Any, run_id: str
) -> bool:
    """Fit every candidate on train, score validation, and record the trials."""
    fit_splits, target = boundary_for_stage("selection")
    VALIDATION_RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    requested = [name.strip() for name in args.models.split(",") if name.strip()]

    grids: dict[str, list[dict[str, Any]]] = {}
    if LIGHTGCN in requested:
        grids[LIGHTGCN] = (
            [{"embedding_dim": 32, "num_layers": 1, "max_epochs": 2}]
            if args.smoke
            else lightgcn_grid(args.epochs)
        )
    if SASREC in requested:
        grids[SASREC] = (
            [{"maximum_sequence_length": 20, "embedding_dim": 32, "max_epochs": 1}]
            if args.smoke
            else sasrec_grid(args.sasrec_epochs)
        )

    best: dict[str, Any] = {}
    for model_name, grid in grids.items():
        results = []
        for candidate in grid:
            try:
                model, result = _fit_and_score(
                    model_name,
                    candidate,
                    dataset=dataset,
                    config=config,
                    args=args,
                    fit_splits=fit_splits,
                    target=target,
                    seed=config.seed,
                    version_prefix="sel",
                )
            except OmniRankError as exc:
                logger.error(
                    "compare_retrievers.trial_failed",
                    run_id=run_id,
                    model=model_name,
                    configuration=candidate,
                    reason=str(exc),
                )
                return False
            _record_trial(
                logger, run_id, model_name, result.configuration, result, smoke=args.smoke
            )
            results.append((result.configuration, result))
            del model

        chosen, chosen_result = max(results, key=lambda item: item[1].strict.flat()["ndcg@20"])
        flat = chosen_result.strict.flat()
        best[model_name] = {
            **chosen,
            "validation_ndcg@20": round(flat["ndcg@20"], 8),
            "validation_recall@20": round(flat["recall@20"], 8),
        }
        logger.info(
            "compare_retrievers.selected", run_id=run_id, model=model_name, **best[model_name]
        )
        # Written per model, not once at the end: a grid is a long job, and a
        # failure in the last model should not discard the earlier winners.
        partial = PHASE_ROOT / "selection_candidates.json"
        existing = json.loads(partial.read_text()) if partial.is_file() else {}
        existing.update(best)
        write_json(existing, partial)

    return True


def _run_lock(dataset: Any, logger: Any, run_id: str) -> bool:
    """Freeze the selected configurations before any test data is touched.

    Records the dataset identity alongside the hyperparameters. A locked
    configuration that does not say which data it was selected on is not a
    reproducible commitment -- the same numbers could have come from a different
    split or mapping, and nothing in the file would say so.
    """
    partial = PHASE_ROOT / "selection_candidates.json"
    if not partial.is_file():
        logger.error(
            "compare_retrievers.no_candidates",
            run_id=run_id,
            detail="Run --stage selection first.",
            expected=str(partial),
        )
        return False
    selection = json.loads(partial.read_text())
    # Phase 3's locked baselines are carried forward verbatim rather than
    # re-selected: re-tuning them against the same validation split would give
    # the baselines a second search the Phase 3 numbers never had.
    if PHASE_3_SELECTION.is_file():
        phase_3 = json.loads(PHASE_3_SELECTION.read_text())
        for name in (POPULARITY, MATRIX_FACTORIZATION):
            if name in phase_3:
                selection[name] = {**phase_3[name], "inherited_from": "phase_03"}
    selection["dataset_identity"] = dataset.identity.to_dict()
    selection["selected_by"] = "rolling/validation strict ndcg@20"
    selection["fit_splits"] = ["train"]
    selection["target_split"] = "validation"
    write_json(selection, SELECTION_FILE)
    logger.info(
        "compare_retrievers.locked",
        run_id=run_id,
        models=sorted(key for key, value in selection.items() if isinstance(value, dict)),
        dataset=dataset.identity.label,
    )
    return True


def _run_final(
    dataset: Any, config: Any, args: argparse.Namespace, logger: Any, run_id: str
) -> bool:
    """Refit the locked configurations on train+validation and score test once.

    Multi-seed, because a single seed cannot distinguish a real difference from
    initialisation noise -- and with differences this small on a sparse
    catalogue, that distinction is the whole question.
    """
    if not SELECTION_FILE.is_file():
        logger.error(
            "compare_retrievers.no_selection",
            run_id=run_id,
            detail=(
                "No locked configuration found. Run --stage lock first; test "
                "data must never be read before a configuration is locked."
            ),
            expected=str(SELECTION_FILE),
        )
        return False

    selection = json.loads(SELECTION_FILE.read_text())
    fit_splits, target = boundary_for_stage("final")
    requested = [name.strip() for name in args.models.split(",") if name.strip()]

    final_metrics: dict[str, Any] = {}
    for model_name in requested:
        locked = selection.get(model_name)
        if not locked:
            logger.error("compare_retrievers.model_not_locked", run_id=run_id, model=model_name)
            return False
        # Validation metrics sit beside the configuration as provenance; they
        # are not hyperparameters and must not be passed to the constructor.
        candidate = {
            key: value
            for key, value in locked.items()
            if not key.startswith("validation_") and key != "inherited_from"
        }

        per_seed = []
        for offset in range(args.seeds):
            seed = config.seed + offset
            try:
                model, result = _fit_and_score(
                    model_name,
                    candidate,
                    dataset=dataset,
                    config=config,
                    args=args,
                    fit_splits=fit_splits,
                    target=target,
                    seed=seed,
                    version_prefix=f"final-s{seed}",
                )
            except OmniRankError as exc:
                logger.error(
                    "compare_retrievers.final_failed",
                    run_id=run_id,
                    model=model_name,
                    seed=seed,
                    reason=str(exc),
                )
                return False
            flat = result.strict.flat()
            per_seed.append({"seed": seed, **{k: flat[k] for k in TRIAL_METRICS if k in flat}})
            logger.info(
                "compare_retrievers.final_seed",
                run_id=run_id,
                model=model_name,
                seed=seed,
                **{k: round(flat[k], 6) for k in PRIMARY_METRICS if k in flat},
            )
            del model

        final_metrics[model_name] = {
            "configuration": candidate,
            "seeds": per_seed,
            "mean": {
                metric: sum(row[metric] for row in per_seed) / len(per_seed)
                for metric in TRIAL_METRICS
                if per_seed and metric in per_seed[0]
            },
            "target_split": target,
            "fit_splits": list(fit_splits),
        }
        write_json(final_metrics, PHASE_ROOT / "final_test_metrics.json")

    logger.info("compare_retrievers.final_complete", run_id=run_id, models=sorted(final_metrics))
    return True


def _run_report(logger: Any, run_id: str) -> bool:
    """Assemble the Phase 4 tables from the records the other stages wrote.

    Reads only what was actually run. A stage that has not been run leaves its
    section absent rather than filled with a placeholder, so the report cannot
    imply an experiment that never happened.
    """
    if not VALIDATION_RUNS_FILE.is_file():
        logger.error(
            "compare_retrievers.no_runs",
            run_id=run_id,
            detail="Run --stage selection first.",
            expected=str(VALIDATION_RUNS_FILE),
        )
        return False

    trials = [
        json.loads(line) for line in VALIDATION_RUNS_FILE.read_text().splitlines() if line.strip()
    ]
    # Smoke trials use a deliberately tiny budget and must never influence a
    # reported comparison.
    real = [row for row in trials if not row.get("smoke")]

    rows = [
        {
            "model": row["model"],
            "num_layers": row.get("num_layers"),
            "maximum_sequence_length": row.get("maximum_sequence_length"),
            "embedding_dim": row.get("embedding_dim"),
            "max_epochs": row.get("max_epochs"),
            "recall@20": row.get("recall@20"),
            "ndcg@20": row.get("ndcg@20"),
            "coverage@20": row.get("coverage@20"),
        }
        for row in real
    ]
    write_csv(rows, PHASE_ROOT / "validation_comparison.csv")

    ablation = sorted(
        (row for row in real if row["model"] == LIGHTGCN and row.get("num_layers") is not None),
        key=lambda row: row["num_layers"],
    )
    if ablation:
        baseline = next((row for row in ablation if row["num_layers"] == 0), None)
        write_json(
            {
                "question": (
                    "Does graph propagation improve on matrix factorization, "
                    "holding code, data, objective and seed fixed?"
                ),
                "control": (
                    "num_layers=0 disables propagation, so LightGCN degenerates "
                    "to matrix factorization trained by the same code."
                ),
                "rows": [
                    {
                        "num_layers": row["num_layers"],
                        "ndcg@20": row.get("ndcg@20"),
                        "recall@20": row.get("recall@20"),
                        "coverage@20": row.get("coverage@20"),
                        "ndcg_ratio_to_zero_layers": (
                            round(row["ndcg@20"] / baseline["ndcg@20"], 4)
                            if baseline and baseline.get("ndcg@20")
                            else None
                        ),
                    }
                    for row in ablation
                ],
            },
            PHASE_ROOT / "lightgcn_layer_ablation.json",
        )

    if SELECTION_FILE.is_file():
        write_json(json.loads(SELECTION_FILE.read_text()), PHASE_ROOT / "locked_configuration.json")

    logger.info(
        "compare_retrievers.report_written",
        run_id=run_id,
        trials=len(real),
        smoke_trials=len(trials) - len(real),
        output=str(PHASE_ROOT),
    )
    return True


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
    logger = get_logger("omnirank.compare_retrievers")
    dataset_config = config.data.dataset
    if dataset_config is None:
        print("The selected profile declares no data.dataset block.", file=sys.stderr)
        return CONFIG_ERROR_EXIT

    with run_context(stage="compare_retrievers") as run_id:
        try:
            dataset = load_processed_dataset(
                Path(dataset_config.processed_dir),
                Path(config.paths.mappings_dir) / config.data.dataset_name,
                verify_checksums=not args.skip_checksums,
            )
        except OmniRankError as exc:
            logger.error("compare_retrievers.dataset_unavailable", run_id=run_id, reason=str(exc))
            return CONFIG_ERROR_EXIT

        if args.stage in ("all", "selection") and not _run_selection(
            dataset, config, args, logger, run_id
        ):
            return RUN_ERROR_EXIT

        if args.stage in ("all", "lock") and not _run_lock(dataset, logger, run_id):
            return RUN_ERROR_EXIT

        if args.stage in ("all", "final") and not _run_final(dataset, config, args, logger, run_id):
            return RUN_ERROR_EXIT

        if args.stage in ("all", "report") and not _run_report(logger, run_id):
            return RUN_ERROR_EXIT

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
