#!/usr/bin/env python
"""Phase 5 multimodal retrieval experiments.

    python scripts/compare_multimodal_retrievers.py --stage preflight
    python scripts/compare_multimodal_retrievers.py --stage rolling-selection
    python scripts/compare_multimodal_retrievers.py --stage lock
    python scripts/compare_multimodal_retrievers.py --stage final
    python scripts/compare_multimodal_retrievers.py --stage report

Stages, in the order the discipline requires:

1. **preflight** - features aligned, store loads, identities agree.
2. **rolling-selection** - modality ablations and configuration search on
   validation and the pre-test rolling folds. Never touches the test split.
3. **lock** - freeze the selected configuration *before* any test metric is read.
4. **final** - refit on train+validation, encode the catalogue, build the exact
   index, evaluate test once.
5. **report** - assemble the tables.

Every model here is scored by ``run_experiment`` from the Phase 3 runner,
unchanged, so a two-tower number and a LightGCN number are comparable. What
differs between them is only how they are fitted.

``--stage final`` refuses to run without a locked configuration.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from omnirank.core.config import load_config
from omnirank.core.exceptions import ConfigurationError, OmniRankError
from omnirank.core.logging import configure_logging, get_logger, run_context
from omnirank.data.processed import load_processed_dataset
from omnirank.evaluation.evaluator import PRIMARY_METRICS
from omnirank.evaluation.reporting import REPORT_ROOT, append_jsonl, write_csv, write_json
from omnirank.models.baselines.runner import boundary_for_stage, run_experiment
from omnirank.retrieval.fold_evaluation import ABLATION_OVERRIDES, summarise_folds

CONFIG_ERROR_EXIT = 2
RUN_ERROR_EXIT = 3

PHASE_ROOT = REPORT_ROOT.parent / "phase_05"

#: Screened variants carried into the fold confirmation.
#:
#: Four, not two. Two independent runs of the same nine-variant screen produced
#: *disjoint* top-two sets, which says the screen's ordering is noise-dominated
#: at this subset size. A shortlist of two would therefore be a coin flip
#: dressed as a ranking. Four is what the fold budget affords while making it
#: unlikely that the fold-best variant was screened out.
FINALIST_COUNT = 4
RUNS_FILE = PHASE_ROOT / "rolling_validation_runs.jsonl"
SELECTION_FILE = PHASE_ROOT / "selected_configuration.json"
CANDIDATES_FILE = PHASE_ROOT / "selection_candidates.json"
PROCESSED = "data/processed/pixelrec50k"

TRIAL_METRICS = ("recall@20", "ndcg@20", "coverage@20")


def ablation_grid() -> list[dict[str, Any]]:
    """The modality ablations, from the single definition both drivers share.

    Each is a configuration *override*, not a separate model class, so every
    variant goes through identical code and a difference between them is
    attributable to the input rather than to the implementation. Sourced from
    :data:`ABLATION_OVERRIDES` so the screen and the fold confirmation cannot
    end up running different grids under the same labels.
    """
    return [{"label": label, **overrides} for label, overrides in ABLATION_OVERRIDES.items()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--data-config", default="configs/data/pixelrec50k.yaml")
    parser.add_argument(
        "--stage",
        default="preflight",
        choices=("preflight", "rolling-selection", "lock", "final", "report"),
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--subset-users", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated seeds for multi-seed verification of the selection.",
    )
    parser.add_argument(
        "--fold-offsets",
        default="3,2",
        type=lambda value: [int(part) for part in value.split(",")],
        help="Rolling-fold target offsets. Offset 1 is the reserved test target.",
    )
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an already-registered artifact version of the same name.",
    )
    parser.add_argument("--only", default=None, help="Comma-separated ablation labels to run.")
    parser.add_argument(
        "--reuse-screen",
        action="store_true",
        help=(
            "Load ablation_results.csv instead of re-running the screen. Only "
            "valid when that file was produced by this code at this commit."
        ),
    )
    parser.add_argument(
        "--reuse-folds",
        action="store_true",
        help="Load rolling_fold_results.csv instead of re-running the folds.",
    )
    return parser.parse_args(argv)


def _build_retriever(
    model: Any,
    store: Any,
    dataset: Any,
    fit_splits: tuple[str, ...],
    subset_users: int | None,
    device: str,
) -> Any:
    """Wrap a trained network in its retrieval surface."""
    from omnirank.models.two_tower import TwoTowerRetriever
    from omnirank.retrieval.runner import load_item_tags, load_sequences

    tags, _ = load_item_tags(PROCESSED, dataset.num_items)
    sequences = load_sequences(PROCESSED, fit_splits)
    if subset_users is not None:
        sequences = sequences[sequences["internal_user_id"] < subset_users]

    histories: dict[int, list[int]] = {}
    warm = np.zeros(dataset.num_items, dtype=bool)
    for user, history, target in zip(
        sequences["internal_user_id"],
        sequences["item_sequence"],
        sequences["target_item"],
        strict=True,
    ):
        combined = [*list(history), int(target)]
        key = int(user)
        if key not in histories or len(combined) > len(histories[key]):
            histories[key] = combined
        warm[list(history)] = True
        warm[int(target)] = True

    return TwoTowerRetriever.from_trained(
        model, store, dataset, histories, warm, tags, device=device
    )


def _run_variant(
    label: str,
    overrides: dict[str, Any],
    *,
    dataset: Any,
    config: Any,
    args: argparse.Namespace,
    fit_splits: tuple[str, ...],
    target: str,
    seed: int,
    subset_users: int | None = -1,
) -> tuple[Any, dict[str, Any]]:
    """Train one configuration and score it through the shared harness.

    ``subset_users`` defaults to the sentinel ``-1`` meaning "use the CLI
    value". Pass ``None`` to fit on every user regardless of the flag, which is
    what the final stage does: selection may subset for affordability, but a
    model registered as final must be fitted on the whole population it will be
    asked to serve.
    """
    import yaml

    from omnirank.models.two_tower import TwoTowerConfig
    from omnirank.retrieval.runner import fit_two_tower

    raw = yaml.safe_load(Path("configs/models/two_tower.yaml").read_text())["two_tower"]
    raw.update({k: v for k, v in overrides.items() if k != "label"})
    raw.update({"max_epochs": args.epochs, "seed": seed, "device": args.device})
    model_config = TwoTowerConfig(**raw)
    users = args.subset_users if subset_users == -1 else subset_users

    started = time.perf_counter()
    (model, store, history), fit_measurement = fit_two_tower(
        dataset,
        fit_splits,
        model_config,
        processed_root=PROCESSED,
        device=args.device,
        subset_users=users,
    )
    train_seconds = time.perf_counter() - started

    retriever = _build_retriever(model, store, dataset, fit_splits, users, args.device)
    started = time.perf_counter()
    retriever.export_item_embeddings()
    export_seconds = time.perf_counter() - started

    result = run_experiment(
        retriever,
        dataset,
        config,
        model_name="two_tower",
        model_version=f"abl-{label}-s{seed}",
        fit_splits=fit_splits,
        target_split=target,
        fit_measurement=fit_measurement,
        configuration=model_config.to_dict(),
        bootstrap=False,
    )
    flat = result.strict.flat()
    cold = {
        slice_result.slice_name: slice_result.metrics.get("ndcg@20")
        for slice_result in result.slices
    }
    record = {
        # Identity first, so a row in the CSV can be traced back to the run
        # that produced it without joining against the JSONL log.
        "experiment_id": f"{label}@{target}@seed{seed}",
        "label": label,
        # The screen has no fold: it is a single train-to-validation boundary,
        # and recording it as one would make the column a lie the moment
        # somebody concatenated these rows with the fold results.
        "fold": f"boundary:{'+'.join(fit_splits)}->{target}",
        "seed": seed,
        "configuration_hash": model_config.label,
        **{key: round(flat[key], 8) for key in TRIAL_METRICS if key in flat},
        "cold_ndcg@20": cold.get("items_cold_start"),
        "catalogue_items": len(retriever.catalogue),
        "warm_items": retriever.catalogue.warm_count,
        "cold_items": retriever.catalogue.cold_count,
        "train_seconds": round(train_seconds, 1),
        "embedding_export_seconds": round(export_seconds, 1),
        "device": history.device,
        "best_epoch": history.best_epoch,
        "final_train_loss": round(history.train_loss[-1], 6) if history.train_loss else None,
        **{k: v for k, v in overrides.items() if k != "label"},
    }
    return (retriever, result), record


def _run_preflight(dataset: Any, logger: Any, run_id: str) -> bool:
    """Verify the feature foundation before anything expensive runs."""
    from omnirank.features.multimodal_store import MultimodalFeatureStore

    try:
        store = MultimodalFeatureStore(f"{PROCESSED}/features")
    except OmniRankError as exc:
        logger.error("phase5.features_unavailable", run_id=run_id, reason=str(exc))
        return False

    described = store.describe()
    mapping = dataset.mapping_metadata.get("item_mapping_checksum", "")
    if mapping and described["item_mapping_checksum"] != mapping:
        logger.error(
            "phase5.mapping_mismatch",
            run_id=run_id,
            detail="The feature store describes a different item mapping.",
            store=described["item_mapping_checksum"][:16],
            dataset=mapping[:16],
        )
        return False

    write_json(described, PHASE_ROOT / "feature_coverage.json")
    logger.info(
        "phase5.preflight_passed",
        run_id=run_id,
        **{
            "catalogue_items": described["catalogue_items"],
            "items_with_any_modality": described["items_with_any_modality"],
            "items_with_no_modality": described["items_with_no_modality"],
        },
    )
    return True


def _build_folds(dataset: Any, args: argparse.Namespace, logger: Any, run_id: str) -> Any:
    """Construct the genuine pre-test rolling folds.

    A single train-to-validation boundary is not rolling selection: it measures
    one week and cannot distinguish "this configuration is better" from "this
    configuration suits this particular boundary". Phase 3 ended with exactly
    that ambiguity, and Phase 4 built the fold machinery to resolve it.

    Offset 1 is the official test target and is refused by ``build_fold``
    itself, so a selection run cannot reach it even by mistake.
    """
    from omnirank.data.rolling import (
        build_rolling_validation,
        check_fold_integrity,
        check_no_reserved_offset_used,
    )

    interactions = dataset.fit_interactions(("train", "validation"))
    validation = build_rolling_validation(
        interactions,
        target_offsets=tuple(args.fold_offsets),
        dataset_identity=dataset.identity.to_dict(),
    )
    # Both assertions are cheap and both failures are invisible at runtime: a
    # fold that leaked future events trains a better-looking model, and a fold
    # that reached offset 1 reports a test number as a validation one.
    for fold in validation.folds:
        check_fold_integrity(fold)
    check_no_reserved_offset_used(validation)

    write_json(validation.manifest(), PHASE_ROOT / "rolling_fold_manifest.json")
    for fold in validation.folds:
        logger.info("phase5.fold_built", run_id=run_id, **fold.describe())
    return validation


def _run_selection(
    dataset: Any, config: Any, args: argparse.Namespace, logger: Any, run_id: str
) -> bool:
    """Screen the ablation grid, then confirm the finalists across folds.

    Never touches test. The screen uses the train-to-validation boundary
    because nine configurations at two origins each is compute nobody needs to
    spend to rank a grid; the finalists then pay for the folds.
    """
    fit_splits, target = boundary_for_stage("selection")
    PHASE_ROOT.mkdir(parents=True, exist_ok=True)
    wanted = {name.strip() for name in args.only.split(",")} if args.only else None

    rows: list[dict[str, Any]] = []
    if wanted and not args.reuse_screen:
        # A partial re-run extends the screen rather than replacing it.
        # Writing only the requested variants would silently drop every other
        # row from ablation_results.csv, and the file would still look valid.
        rows = [
            row
            for row in _load_rows(PHASE_ROOT / "ablation_results.csv", TRIAL_METRICS)
            if row.get("label") not in wanted
        ]
        logger.info("phase5.screen_extended", run_id=run_id, existing=len(rows), adding=len(wanted))

    if args.reuse_screen:
        rows = _load_rows(PHASE_ROOT / "ablation_results.csv", ("ndcg@20", "recall@20"))
        if not rows:
            logger.error(
                "phase5.no_screen_to_reuse",
                run_id=run_id,
                expected=str(PHASE_ROOT / "ablation_results.csv"),
            )
            return False
        logger.info("phase5.screen_reused", run_id=run_id, variants=len(rows))

    for variant in [] if args.reuse_screen else ablation_grid():
        if wanted and variant["label"] not in wanted:
            continue
        try:
            (_, _), record = _run_variant(
                variant["label"],
                variant,
                dataset=dataset,
                config=config,
                args=args,
                fit_splits=fit_splits,
                target=target,
                seed=config.seed,
            )
        except OmniRankError as exc:
            logger.error(
                "phase5.variant_failed",
                run_id=run_id,
                label=variant["label"],
                reason=str(exc),
            )
            return False
        record["run_id"] = run_id
        record["stage"] = "ablation-screen"
        record["subset_users"] = args.subset_users
        append_jsonl(record, RUNS_FILE)
        rows.append(record)
        write_csv(rows, PHASE_ROOT / "ablation_results.csv")
        logger.info(
            "phase5.ablation",
            run_id=run_id,
            label=variant["label"],
            **{k: record[k] for k in PRIMARY_METRICS if k in record},
            cold_ndcg=record.get("cold_ndcg@20"),
        )

    if not rows:
        logger.error("phase5.no_variants_ran", run_id=run_id)
        return False

    # Phase one screened every variant at one origin. That ranks the grid
    # cheaply; it does not establish that the ranking holds. The finalists are
    # re-fitted on each pre-test fold's own history before anything is locked,
    # so the selected configuration is one that won more than one week.
    finalists = [
        row["label"]
        for row in sorted(rows, key=lambda row: row.get("ndcg@20") or 0.0, reverse=True)
    ][:FINALIST_COUNT]
    if args.reuse_folds:
        fold_rows = _load_rows(
            PHASE_ROOT / "rolling_fold_results.csv",
            (
                "strict_ndcg@20",
                "strict_recall@20",
                "cold_recall@20",
                "candidate_recall@200",
                "train_seconds",
                "peak_memory_mb",
            ),
        )
        # Reuse is only meaningful if the finalists are actually in the file.
        # Silently summarising somebody else's runs would be worse than paying
        # to re-run these ones.
        measured = {row["label"] for row in fold_rows}
        if not fold_rows or not set(finalists) <= measured:
            logger.error(
                "phase5.no_folds_to_reuse",
                run_id=run_id,
                finalists=finalists,
                measured=sorted(measured),
            )
            return False
        fold_rows = [row for row in fold_rows if row["label"] in set(finalists)]
        logger.info("phase5.folds_reused", run_id=run_id, runs=len(fold_rows))
    else:
        fold_rows = _confirm_across_folds(finalists, dataset, config, args, logger, run_id)
    if fold_rows is None:
        return False

    summary = {entry["label"]: entry for entry in summarise_folds(fold_rows)}
    write_csv(list(summary.values()), PHASE_ROOT / "rolling_validation_summary.csv")

    chosen = _choose(list(summary.values()), logger, run_id)
    best = next(row for row in rows if row["label"] == chosen["label"])
    # Record the *resolved* configuration, not just the overrides. A lock that
    # stores "use_tag: true" and nothing else leaves every other hyperparameter
    # to whatever configs/models/two_tower.yaml happens to say when the final
    # refit runs -- which is a different model under the same name.
    best = {
        **best,
        **_resolve_configuration(str(chosen["label"]), args),
        "rolling_validation": chosen,
    }
    write_json({"two_tower": best}, CANDIDATES_FILE)
    logger.info(
        "phase5.selected",
        run_id=run_id,
        label=best["label"],
        screen_ndcg=best.get("ndcg@20"),
        mean_fold_ndcg=chosen["mean_strict_ndcg@20"],
        worst_fold_ndcg=chosen["worst_fold_strict_ndcg@20"],
        folds=chosen["folds"],
    )
    return True


def _choose(summary: list[dict[str, Any]], logger: Any, run_id: str) -> dict[str, Any]:
    """Pick a configuration from the fold summary, noise-aware.

    Ranking purely on the mean picks a winner even when the gap between the top
    two is smaller than their own run-to-run spread -- which is choosing on
    noise while producing a number that looks like a decision. Measured spreads
    here are frequently larger than the gaps.

    So: take the highest mean only when it leads the runner-up by more than the
    larger of the two standard deviations. Otherwise the two are not
    distinguishable on this evidence, and the tie-break is the **worst single
    run**, which prefers the configuration that fails less badly rather than the
    one that happened to spike.
    """
    ranked = sorted(summary, key=lambda entry: entry["mean_strict_ndcg@20"], reverse=True)
    best = ranked[0]
    if len(ranked) < 2:
        return best

    runner_up = ranked[1]
    gap = best["mean_strict_ndcg@20"] - runner_up["mean_strict_ndcg@20"]
    noise = max(best["stdev_strict_ndcg@20"], runner_up["stdev_strict_ndcg@20"])
    if gap > noise:
        logger.info(
            "phase5.selection_decisive",
            run_id=run_id,
            winner=best["label"],
            gap=round(gap, 8),
            noise=round(noise, 8),
        )
        return best

    contenders = [
        entry
        for entry in ranked
        if best["mean_strict_ndcg@20"] - entry["mean_strict_ndcg@20"] <= noise
    ]
    tie_broken = max(contenders, key=lambda entry: entry["worst_fold_mean_strict_ndcg@20"])
    counts = {entry["label"]: entry["runs"] for entry in contenders}
    logger.info(
        "phase5.selection_within_noise",
        run_id=run_id,
        winner=tie_broken["label"],
        highest_mean=best["label"],
        gap=round(gap, 8),
        noise=round(noise, 8),
        contenders=[entry["label"] for entry in contenders],
        runs_per_contender=counts,
        # Surfaced rather than silently tolerated: a tie-break across unequal
        # footing is weaker evidence, and a reader should be able to see it.
        equal_footing=len(set(counts.values())) == 1,
        rule="means within one stdev; broken on worst fold mean",
    )
    return tie_broken


def _resolve_configuration(label: str, args: argparse.Namespace) -> dict[str, Any]:
    """The full hyperparameter set a label expands to, as the trainer sees it."""
    import yaml

    from omnirank.models.two_tower import TwoTowerConfig
    from omnirank.retrieval.fold_evaluation import overrides_for

    raw = yaml.safe_load(Path("configs/models/two_tower.yaml").read_text())["two_tower"]
    raw.update(overrides_for(label))
    raw.update({"max_epochs": args.epochs, "device": args.device})
    return TwoTowerConfig(**raw).to_dict()


def _confirm_across_folds(
    labels: list[str],
    dataset: Any,
    config: Any,
    args: argparse.Namespace,
    logger: Any,
    run_id: str,
) -> list[dict[str, Any]] | None:
    """Re-fit the finalists on each rolling fold. ``None`` on failure."""
    import yaml

    from omnirank.retrieval.fold_evaluation import evaluate_on_fold

    try:
        validation = _build_folds(dataset, args, logger, run_id)
    except OmniRankError as exc:
        logger.error("phase5.folds_failed", run_id=run_id, reason=str(exc))
        return None

    base_config = yaml.safe_load(Path("configs/models/two_tower.yaml").read_text())["two_tower"]
    seeds = [int(value) for value in str(args.seeds).split(",")]
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
                        processed_root=Path(config.data.dataset.processed_dir),
                        base_config=base_config,
                        epochs=args.epochs,
                        device=args.device,
                        subset_users=args.subset_users,
                    )
                except OmniRankError as exc:
                    logger.error(
                        "phase5.fold_run_failed",
                        run_id=run_id,
                        label=label,
                        fold=fold.name,
                        seed=seed,
                        reason=str(exc),
                    )
                    return None
                record |= {"run_id": run_id, "stage": "rolling-fold"}
                append_jsonl(record, PHASE_ROOT / "rolling_validation_runs.jsonl")
                rows.append(record)
                write_csv(rows, PHASE_ROOT / "rolling_fold_results.csv")
    return rows


def _load_rows(path: Path, numeric: tuple[str, ...]) -> list[dict[str, Any]]:
    """Read a metric CSV back, restoring the numeric columns.

    ``csv.DictReader`` returns strings, and a summariser comparing "0.01" with
    "0.009" as strings would rank them wrongly and never say so.
    """
    if not path.is_file() or not path.read_text().strip():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for column in numeric:
            value = row.get(column)
            if value in (None, "", "None"):
                row[column] = None
                continue
            with contextlib.suppress(ValueError):
                row[column] = float(value)
    return rows


def _run_lock(dataset: Any, logger: Any, run_id: str) -> bool:
    """Freeze the selection before any test data is read."""
    if not CANDIDATES_FILE.is_file():
        logger.error(
            "phase5.no_candidates",
            run_id=run_id,
            detail="Run --stage rolling-selection first.",
            expected=str(CANDIDATES_FILE),
        )
        return False
    selection = json.loads(CANDIDATES_FILE.read_text())
    selection["dataset_identity"] = dataset.identity.to_dict()
    selection["selected_by"] = (
        "two stage: ablation screen on the train->validation boundary, then "
        "confirmation across rolling folds; chosen on mean fold strict ndcg@20 "
        "with the worst fold as tie-breaker"
    )
    selection["fit_splits"] = ["train"]
    selection["target_split"] = "validation"
    write_json(selection, SELECTION_FILE)
    logger.info(
        "phase5.locked",
        run_id=run_id,
        models=sorted(key for key, value in selection.items() if isinstance(value, dict)),
    )
    return True


def _run_final(
    dataset: Any, config: Any, args: argparse.Namespace, logger: Any, run_id: str
) -> bool:
    """Refit the locked configuration on train+validation and score test once.

    Also builds and verifies the artifacts the phase is judged on: the item
    catalogue, the embedding matrix, and an exact FAISS index containing cold
    items. An index that quietly contained none would still answer every query,
    so its composition is recorded rather than assumed.
    """
    from omnirank.retrieval.two_tower_index import (
        build_two_tower_index,
        verify_index_against_brute_force,
        write_item_embeddings,
    )

    locked = json.loads(SELECTION_FILE.read_text())["two_tower"]
    # Every hyperparameter the lock recorded, filtered by what the config
    # accepts rather than by a hand-maintained allow-list. An allow-list drops
    # silently: add a field to TwoTowerConfig, forget to list it here, and the
    # final refit uses the YAML default while the lock says otherwise. The
    # record also carries metrics and bookkeeping, which are not fields.
    from omnirank.models.two_tower import TwoTowerConfig

    accepted = {field.name for field in dataclasses.fields(TwoTowerConfig)}
    overrides = {key: value for key, value in locked.items() if key in accepted}
    fit_splits, target = boundary_for_stage("final")
    logger.info(
        "phase5.final_started",
        run_id=run_id,
        label=locked.get("label"),
        fit_splits=list(fit_splits),
        target=target,
    )

    try:
        (retriever, result), record = _run_variant(
            str(locked.get("label", "final")),
            overrides,
            dataset=dataset,
            config=config,
            args=args,
            fit_splits=fit_splits,
            target=target,
            seed=config.seed,
            # Every user, never the development subset. A "final" model fitted
            # on 10% of the population can answer for 10% of the population,
            # and returns an empty list for everyone else -- which depresses
            # every metric it is measured on while looking like a modelling
            # result rather than a missing flag.
            subset_users=None,
        )
    except OmniRankError as exc:
        logger.error("phase5.final_failed", run_id=run_id, reason=str(exc))
        return False

    shortfall = population_shortfall(retriever, dataset.num_users)
    if shortfall is not None:
        logger.error("phase5.final_incomplete_population", run_id=run_id, **shortfall)
        return False

    catalogue = retriever.catalogue
    embeddings = retriever.item_embeddings_matrix
    version = "phase5-two-tower-final"
    destination = Path(config.paths.artifact_root) / "embeddings" / "two_tower" / version
    identity = {
        "model_version": version,
        "model_checksum": retriever.catalogue.checksum(),
        "mapping_checksum": retriever.mapping_checksum,
        "feature_version": retriever.store.feature_version,
        "feature_manifest_checksum": retriever.store.manifest_checksum(),
        "normalization": "l2" if retriever.config.l2_normalize else "none",
    }
    write_item_embeddings(destination, embeddings, catalogue, **identity)

    index, index_metadata = build_two_tower_index(embeddings, catalogue, **identity)
    queries = retriever.build_query_embedding(
        list(retriever._histories.values())[:256],
        list(retriever._histories.keys())[:256],
    )
    exactness = verify_index_against_brute_force(index, embeddings, queries, k=20)
    index_path = Path(config.paths.indexes_dir) / config.data.dataset_name / "two_tower" / version
    index.save(index_path)
    write_json({**index_metadata, "exactness": exactness}, destination / "index_manifest.json")

    _register_final_model(
        retriever,
        result,
        config,
        version=version,
        index_version=index.index_version,
        fit_splits=fit_splits,
        target=target,
        seed=config.seed,
        overwrite=args.overwrite,
        logger=logger,
        run_id=run_id,
    )

    # Cold metrics, reported independently of the aggregate.
    slices = {item.slice_name: item.to_dict() for item in result.slices}
    cold_rows = [
        {"slice": name, **{k: v for k, v in payload.items() if k != "slice_name"}}
        for name, payload in sorted(slices.items())
        if "cold" in name
    ]
    write_csv(cold_rows, PHASE_ROOT / "cold_start_metrics.csv")

    flat = result.strict.flat()
    write_json(
        {
            "stage": "final",
            "target_split": target,
            "fit_splits": list(fit_splits),
            "configuration": locked,
            "strict": {key: flat[key] for key in sorted(flat)},
            "warm": dict(sorted(result.warm.flat().items())),
            "slices": slices,
            "catalogue": {
                "items": len(catalogue),
                "warm": catalogue.warm_count,
                "cold": catalogue.cold_count,
                "excluded": catalogue.excluded_count,
            },
            "index": {**index_metadata, "exactness": exactness},
            "runtime": record,
        },
        PHASE_ROOT / "two_tower_final_test_metrics.json",
    )
    logger.info(
        "phase5.final_complete",
        run_id=run_id,
        **{key: round(flat[key], 8) for key in PRIMARY_METRICS if key in flat},
        cold_items_indexed=catalogue.cold_count,
        index_matches_brute_force=exactness["matches_brute_force"],
    )
    return True


#: Minimum share of the user population a final model must be able to answer
#: for. Not 100%: a few users legitimately have no fittable history -- their
#: entire log is the single interaction that became the held-out target, so
#: there is nothing left to build a query from, and the fold builder excludes
#: them for the same reason. A development subset removes an *order of
#: magnitude*, so any threshold between the two separates the cases cleanly.
MINIMUM_SERVABLE_SHARE = 0.95


def population_shortfall(retriever: Any, expected_users: int) -> dict[str, Any] | None:
    """Report whether a retriever can answer for far fewer users than it should.

    Returns ``None`` when the population is essentially complete, otherwise the
    numbers to log. A final model that serves 10% of users returns an empty
    list for the other 90%, which depresses every metric it is measured on
    while looking like a modelling result rather than a missing
    ``--subset-users`` override.

    This is not hypothetical: the first registered Phase 5 final model was
    fitted with the development default and could answer for 5,000 of 50,000
    users. Nothing in the artifact, the manifest or the metrics said so.
    """
    servable = sum(1 for history in retriever._histories.values() if history)
    if servable >= expected_users * MINIMUM_SERVABLE_SHARE:
        return None
    return {
        "servable_users": servable,
        "expected_users": expected_users,
        "share": round(servable / max(expected_users, 1), 4),
        "detail": (
            "The final model cannot answer for every user. Refit without "
            "--subset-users before registering."
        ),
    }


def _register_final_model(
    retriever: Any,
    result: Any,
    config: Any,
    *,
    version: str,
    index_version: int,
    fit_splits: tuple[str, ...],
    target: str,
    seed: int,
    overwrite: bool,
    logger: Any,
    run_id: str,
) -> None:
    """Save the final retriever and write its registry manifest.

    Registered here rather than through ``register_baseline`` because that
    helper pins ``feature_version`` to the Phase 3 constant and declares
    ``required_index_version=1``. Both are wrong for this model: its features
    come from the multimodal store and it is paired with a real FAISS build, so
    a manifest carrying the defaults would assert compatibility it cannot back.
    """
    from omnirank.artifacts.metadata import (
        ArtifactType,
        SupportedDevice,
        build_metadata,
    )
    from omnirank.artifacts.registry import ArtifactRegistry

    artifact_root = Path(config.paths.artifact_root)
    artifact_dir = Path(config.paths.models_dir) / config.data.dataset_name / "two_tower" / version
    retriever.save(artifact_dir)

    # Project-root relative, matching every other registered model. The
    # registry would happily resolve an artifact-root-relative path, but the
    # gate and the serving loader both join against the project root, so a
    # different convention here produces a manifest that points nowhere.
    relative = artifact_dir

    flat = result.strict.flat()
    cold = {item.slice_name: item for item in result.slices if "cold" in item.slice_name}
    metrics = {
        f"{target}_{key}": float(value) for key, value in flat.items() if key in PRIMARY_METRICS
    }
    for name, item in sorted(cold.items()):
        payload = item.to_dict()
        for key in ("recall@20", "ndcg@20"):
            if key in payload:
                metrics[f"{target}_{name}_{key}"] = float(payload[key])

    metadata = build_metadata(
        model_name="two_tower",
        model_version=version,
        model_type=ArtifactType.RETRIEVAL_MODEL,
        training_data_version=(
            f"{retriever.dataset_identity.get('dataset_name')}@"
            f"{retriever.dataset_identity.get('dataset_version')}"
        ),
        feature_version=str(retriever.store.feature_version),
        configuration_hash=retriever.config.label,
        random_seed=seed,
        supported_device=SupportedDevice.ANY,
        required_index_version=index_version,
        metrics=metrics,
        artifact_path=str(relative),
        id_mapping_fingerprints={"item": retriever.mapping_checksum},
        notes=(
            f"stage=final | fit_splits={'+'.join(fit_splits)} | target={target} | "
            f"protocol=full_catalogue | warm_items={retriever.catalogue.warm_count} | "
            f"cold_items={retriever.catalogue.cold_count} | "
            f"feature_manifest={retriever.store.manifest_checksum()[:16]} | "
            f"trained_on_device={retriever.device}"
        ),
    )
    registry = ArtifactRegistry(Path(config.paths.metadata_dir), artifact_root=artifact_root)
    manifest_path = registry.register(metadata, overwrite=overwrite)
    logger.info(
        "phase5.final_registered",
        run_id=run_id,
        artifact=metadata.key,
        manifest=str(manifest_path),
        payload=str(artifact_dir),
        index_version=index_version,
    )


def _run_report(logger: Any, run_id: str) -> bool:
    """Assemble the Phase 5 tables from what actually ran."""
    if not (PHASE_ROOT / "ablation_results.csv").is_file():
        logger.error(
            "phase5.no_ablations",
            run_id=run_id,
            detail="Run --stage rolling-selection first.",
        )
        return False
    # Delegated rather than duplicated: the generator reads every metric file
    # in this directory, and a second assembler here would be a second place
    # for the report and the numbers to disagree.
    generator = Path(__file__).resolve().parent / "generate_phase5_report.py"
    completed = subprocess.run(  # noqa: S603 - fixed path, no shell, no user input
        [sys.executable, str(generator)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        logger.error(
            "phase5.report_failed",
            run_id=run_id,
            returncode=completed.returncode,
            stderr=completed.stderr.strip()[:400],
        )
        return False
    logger.info(
        "phase5.report_written",
        run_id=run_id,
        output=str(PHASE_ROOT),
        detail=completed.stdout.strip().replace("\n", "; "),
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
    logger = get_logger("omnirank.phase5")
    if config.data.dataset is None:
        print("The selected profile declares no data.dataset block.", file=sys.stderr)
        return CONFIG_ERROR_EXIT

    with run_context(stage="compare_multimodal_retrievers") as run_id:
        try:
            dataset = load_processed_dataset(
                Path(config.data.dataset.processed_dir),
                Path(config.paths.mappings_dir) / config.data.dataset_name,
                verify_checksums=not args.skip_checksums,
            )
        except OmniRankError as exc:
            logger.error("phase5.dataset_unavailable", run_id=run_id, reason=str(exc))
            return CONFIG_ERROR_EXIT

        if args.stage == "preflight":
            return 0 if _run_preflight(dataset, logger, run_id) else RUN_ERROR_EXIT
        if args.stage == "rolling-selection":
            return 0 if _run_selection(dataset, config, args, logger, run_id) else RUN_ERROR_EXIT
        if args.stage == "lock":
            return 0 if _run_lock(dataset, logger, run_id) else RUN_ERROR_EXIT
        if args.stage == "final":
            if not SELECTION_FILE.is_file():
                logger.error(
                    "phase5.no_selection",
                    run_id=run_id,
                    detail=(
                        "No locked configuration. Run --stage lock first; test "
                        "data must never be read before a configuration is locked."
                    ),
                    expected=str(SELECTION_FILE),
                )
                return RUN_ERROR_EXIT
            return 0 if _run_final(dataset, config, args, logger, run_id) else RUN_ERROR_EXIT
        if args.stage == "report":
            return 0 if _run_report(logger, run_id) else RUN_ERROR_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
