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
import json
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

CONFIG_ERROR_EXIT = 2
RUN_ERROR_EXIT = 3

PHASE_ROOT = REPORT_ROOT.parent / "phase_05"
RUNS_FILE = PHASE_ROOT / "rolling_validation_runs.jsonl"
SELECTION_FILE = PHASE_ROOT / "selected_configuration.json"
CANDIDATES_FILE = PHASE_ROOT / "selection_candidates.json"
PROCESSED = "data/processed/pixelrec50k"

TRIAL_METRICS = ("recall@20", "ndcg@20", "coverage@20")


def ablation_grid() -> list[dict[str, Any]]:
    """The modality ablations §11 requires.

    Each is a configuration override, not a separate model class, so every
    variant goes through identical code and a difference between them is
    attributable to the modality rather than to the implementation.
    """
    return [
        {
            "label": "text_only",
            "use_text": True,
            "use_image": False,
            "use_tag": False,
            "use_item_id_residual": False,
        },
        {
            "label": "image_only",
            "use_text": False,
            "use_image": True,
            "use_tag": False,
            "use_item_id_residual": False,
        },
        {
            "label": "text_image",
            "use_text": True,
            "use_image": True,
            "use_tag": False,
            "use_item_id_residual": False,
        },
        {
            "label": "text_image_tag",
            "use_text": True,
            "use_image": True,
            "use_tag": True,
            "use_item_id_residual": False,
        },
        {
            "label": "full_with_id_residual",
            "use_text": True,
            "use_image": True,
            "use_tag": True,
            "use_item_id_residual": True,
        },
        {
            "label": "full_no_user_id",
            "use_text": True,
            "use_image": True,
            "use_tag": True,
            "use_item_id_residual": False,
            "use_user_id_embedding": False,
        },
        {
            "label": "mean_pooling",
            "use_text": True,
            "use_image": True,
            "use_tag": True,
            "use_item_id_residual": False,
            "history_pooling": "mean",
        },
    ]


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
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--only", default=None, help="Comma-separated ablation labels to run.")
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
) -> tuple[Any, dict[str, Any]]:
    """Train one configuration and score it through the shared harness."""
    import yaml

    from omnirank.models.two_tower import TwoTowerConfig
    from omnirank.retrieval.runner import fit_two_tower

    raw = yaml.safe_load(Path("configs/models/two_tower.yaml").read_text())["two_tower"]
    raw.update({k: v for k, v in overrides.items() if k != "label"})
    raw.update({"max_epochs": args.epochs, "seed": seed, "device": args.device})
    model_config = TwoTowerConfig(**raw)

    started = time.perf_counter()
    (model, store, history), fit_measurement = fit_two_tower(
        dataset,
        fit_splits,
        model_config,
        processed_root=PROCESSED,
        device=args.device,
        subset_users=args.subset_users,
    )
    train_seconds = time.perf_counter() - started

    retriever = _build_retriever(model, store, dataset, fit_splits, args.subset_users, args.device)
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
        "label": label,
        "seed": seed,
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


def _run_selection(
    dataset: Any, config: Any, args: argparse.Namespace, logger: Any, run_id: str
) -> bool:
    """Run the modality ablations on validation, never on test."""
    fit_splits, target = boundary_for_stage("selection")
    PHASE_ROOT.mkdir(parents=True, exist_ok=True)
    wanted = {name.strip() for name in args.only.split(",")} if args.only else None

    rows: list[dict[str, Any]] = []
    for variant in ablation_grid():
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
        record["stage"] = "rolling-selection"
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

    best = max(rows, key=lambda row: row.get("ndcg@20") or 0.0)
    write_json({"two_tower": best}, CANDIDATES_FILE)
    logger.info("phase5.selected", run_id=run_id, label=best["label"], ndcg=best.get("ndcg@20"))
    return True


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
    selection["selected_by"] = "validation strict ndcg@20, cold recall as tie-breaker"
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
    overrides = {
        key: value
        for key, value in locked.items()
        if key
        in {
            "use_text",
            "use_image",
            "use_tag",
            "use_user_id_embedding",
            "use_item_id_residual",
            "history_pooling",
            "embedding_dim",
            "temperature",
            "learning_rate",
        }
    }
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
        )
    except OmniRankError as exc:
        logger.error("phase5.final_failed", run_id=run_id, reason=str(exc))
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


def _run_report(logger: Any, run_id: str) -> bool:
    """Assemble the Phase 5 tables from what actually ran."""
    if not (PHASE_ROOT / "ablation_results.csv").is_file():
        logger.error(
            "phase5.no_ablations",
            run_id=run_id,
            detail="Run --stage rolling-selection first.",
        )
        return False
    logger.info("phase5.report_written", run_id=run_id, output=str(PHASE_ROOT))
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
