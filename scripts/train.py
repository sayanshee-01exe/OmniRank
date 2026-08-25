#!/usr/bin/env python
"""Train and register an OmniRank baseline model.

    python scripts/train.py --model popularity \
        --data-config configs/data/pixelrec50k.yaml \
        --stage selection --version phase3-popularity-selection

    python scripts/train.py --model matrix_factorization \
        --data-config configs/data/pixelrec50k.yaml \
        --stage final --version phase3-mf-final

Stages set the fit boundary, and getting it right is the whole discipline of
this phase:

===========  ==============================  ===================
stage        fit splits                      evaluation target
===========  ==============================  ===================
selection    train                           validation
final        train + validation              test
===========  ==============================  ===================

Hyperparameters come from the model's block in ``configs/models/retrieval.yaml``.
CLI flags override **only** what is explicitly passed, so there is one source of
truth and one documented way to deviate from it.

Exit codes: 0 success · 2 configuration or data error · 3 training failure.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typing import Any

from omnirank.artifacts.registry import ArtifactRegistry
from omnirank.core.config import load_config
from omnirank.core.exceptions import ConfigurationError, OmniRankError
from omnirank.core.logging import configure_logging, get_logger, run_context
from omnirank.data.processed import load_processed_dataset
from omnirank.models.baselines.popularity import PopularityConfig
from omnirank.models.baselines.registry_support import register_baseline
from omnirank.models.baselines.runner import (
    MATRIX_FACTORIZATION,
    POPULARITY,
    boundary_for_stage,
    fit_bpr,
    fit_popularity,
    run_experiment,
)

CONFIG_ERROR_EXIT = 2
TRAINING_ERROR_EXIT = 3

LIGHTGCN = "lightgcn"
SASREC = "sasrec"
TWO_TOWER = "two_tower"
MODELS = (POPULARITY, MATRIX_FACTORIZATION, LIGHTGCN, SASREC, TWO_TOWER)
#: Models whose hyperparameters are locked by the Phase 4 selection record.
PHASE_4_MODELS = (LIGHTGCN, SASREC)
#: Models whose hyperparameters are locked by the Phase 5 selection record.
PHASE_5_MODELS = (TWO_TOWER,)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and register an OmniRank baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--data-config", default="configs/data/pixelrec50k.yaml")
    parser.add_argument(
        "--stage",
        default="selection",
        choices=("selection", "final", "development"),
        help=(
            "selection fits train and scores validation; final fits "
            "train+validation; development fits train only and skips evaluation "
            "and registration, for smoke runs."
        ),
    )
    parser.add_argument("--version", required=True, help="Artifact version label.")
    parser.add_argument("--seed", type=int, default=None, help="Override the configured seed.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps"),
        help="Compute device for the torch models. Never selects CUDA.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing version.")
    parser.add_argument(
        "--from-selection",
        action="store_true",
        help=(
            "Take hyperparameters from reports/metrics/phase_03/selected_configuration.json "
            "instead of YAML. Required for a final-stage run, so the registered model is "
            "provably the locked configuration rather than whatever YAML happens to say."
        ),
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Fit and evaluate without writing an artifact. For exploration.",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip dataset checksum verification (faster; less safe).",
    )
    # Explicit hyperparameter overrides. Absent flags fall back to YAML.
    parser.add_argument("--variant", choices=("global_count", "time_decay"))
    parser.add_argument("--half-life-days", type=float)
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--regularization", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--negatives-per-positive", type=int)
    # Phase 4 axes. Ignored by the Phase 3 models.
    parser.add_argument("--num-layers", type=int, help="LightGCN propagation depth.")
    parser.add_argument("--maximum-sequence-length", type=int, help="SASRec context window.")
    parser.add_argument("--num-blocks", type=int, help="SASRec transformer blocks.")
    parser.add_argument("--num-heads", type=int, help="SASRec attention heads.")
    parser.add_argument("--dropout", type=float, help="SASRec dropout.")
    # Phase 5 axes and development controls.
    parser.add_argument(
        "--model-config",
        default=None,
        help="YAML holding this model's hyperparameters (two_tower only).",
    )
    parser.add_argument(
        "--subset-users",
        type=int,
        default=None,
        help=(
            "Train on the first N internal user ids only. The development path: "
            "the full corpus is not something to start by accident."
        ),
    )
    parser.add_argument("--artifact-output", default=None, help="Override the artifact directory.")
    parser.add_argument(
        "--max-batches-per-epoch",
        type=int,
        default=None,
        help="Cap batches per epoch, for smoke runs.",
    )
    return parser.parse_args(argv)


def _finish_two_tower_development(
    model: Any,
    feature_store: Any,
    training_history: Any,
    dataset: Any,
    args: argparse.Namespace,
    config: Any,
    logger: Any,
    run_id: str,
    fit_measurement: Any,
) -> int:
    """Persist a two-tower model and report the run.

    Separate from the shared path because this milestone has no full-catalogue
    retrieval yet, so `run_experiment` has nothing to score. Emitting a metric
    here would mean inventing one.
    """
    from omnirank.models.two_tower import build_metadata, save

    destination = (
        Path(args.artifact_output)
        if getattr(args, "artifact_output", None)
        else Path(config.paths.models_dir) / config.data.dataset_name / args.model / args.version
    )
    if destination.exists() and not args.overwrite:
        logger.error(
            "train.version_exists",
            run_id=run_id,
            path=str(destination),
            detail="Pass --overwrite to replace an existing artifact version.",
        )
        return CONFIG_ERROR_EXIT

    history = training_history.to_dict()
    if not args.no_register:
        metadata = build_metadata(
            model,
            feature_version=feature_store.feature_version,
            feature_manifest_checksum=feature_store.manifest_checksum(),
            mapping_checksum=dataset.mapping_metadata.get("item_mapping_checksum", ""),
            dataset_identity=dataset.identity.to_dict(),
            fit_splits=("train",),
            training_history=history,
        )
        save(model, destination, metadata=metadata, training_history=history)

    logger.info(
        "train.two_tower_completed",
        run_id=run_id,
        version=args.version,
        registered=not args.no_register,
        path=str(destination) if not args.no_register else None,
        subset_users=args.subset_users,
        examples=fit_measurement.items_processed if fit_measurement else None,
        seconds=round(fit_measurement.seconds, 2) if fit_measurement else None,
        peak_memory_mb=round(fit_measurement.peak_memory_mb, 1) if fit_measurement else None,
        device=history.get("device"),
        epochs_run=history.get("epochs_run"),
        best_epoch=history.get("best_epoch"),
        first_loss=round(history["train_loss"][0], 6) if history.get("train_loss") else None,
        final_loss=round(history["train_loss"][-1], 6) if history.get("train_loss") else None,
        validation_loss=(
            round(history["validation_loss"][-1], 6) if history.get("validation_loss") else None
        ),
        note=(
            "Model core only. Full-catalogue retrieval and evaluation land in "
            "the next Phase 5 milestone."
        ),
    )
    return 0


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
    logger = get_logger("omnirank.train")
    seed = args.seed if args.seed is not None else config.seed
    # A development run fits on training only and never reads validation as a
    # target, so a smoke run cannot consume a held-out split by accident.
    fit_splits, target_split = (
        (("train",), "validation")
        if args.stage == "development"
        else boundary_for_stage(args.stage)
    )
    dataset_config = config.data.dataset
    if dataset_config is None:
        print("The selected profile declares no data.dataset block.", file=sys.stderr)
        return CONFIG_ERROR_EXIT

    with run_context(stage="train", model=args.model, version=args.version) as run_id:
        try:
            dataset = load_processed_dataset(
                Path(dataset_config.processed_dir),
                Path(config.paths.mappings_dir) / config.data.dataset_name,
                verify_checksums=not args.skip_checksums,
            )
        except OmniRankError as exc:
            logger.error("train.dataset_unavailable", run_id=run_id, reason=str(exc))
            return CONFIG_ERROR_EXIT

        declared = config.models.candidate_generators[args.model].model_dump()
        if args.from_selection:
            phase = "phase_04" if args.model in PHASE_4_MODELS else "phase_03"
            selection_file = Path(f"reports/metrics/{phase}/selected_configuration.json")
            if not selection_file.is_file():
                logger.error(
                    "train.no_selection",
                    run_id=run_id,
                    detail=(
                        "--from-selection needs a locked configuration. Run the "
                        "matching comparison script's selection stage first "
                        "(compare_baselines.py for Phase 3 models, "
                        "compare_retrievers.py for Phase 4 models)."
                    ),
                    expected=str(selection_file),
                )
                return CONFIG_ERROR_EXIT
            locked = json.loads(selection_file.read_text()).get(args.model)
            if not locked:
                logger.error("train.model_not_in_selection", run_id=run_id, model=args.model)
                return CONFIG_ERROR_EXIT
            # The selection file records validation metrics beside the config;
            # they are provenance, not hyperparameters.
            declared = {
                key: value for key, value in locked.items() if not key.startswith("validation_")
            }
            logger.info(
                "train.using_locked_configuration",
                run_id=run_id,
                model=args.model,
                configuration=declared,
            )
        # Both branches assign a different concrete type; the registry and the
        # runner accept either through their protocol surfaces.
        model: Any
        model_config: Any
        try:
            if args.model == POPULARITY:
                model_config = PopularityConfig(
                    variant=args.variant or declared.get("variant", "time_decay"),
                    half_life_days=(
                        args.half_life_days
                        if args.half_life_days is not None
                        else float(declared.get("half_life_days", 365.0))
                    ),
                )
                model, fit_measurement = fit_popularity(dataset, fit_splits, model_config)
                configuration = model_config.to_dict()
                device = "cpu"
            elif args.model == LIGHTGCN:
                from omnirank.models.lightgcn import LightGCNConfig
                from omnirank.retrieval.runner import fit_lightgcn

                model_config = LightGCNConfig(
                    embedding_dim=args.embedding_dim or int(declared.get("embedding_dim", 64)),
                    num_layers=(
                        args.num_layers
                        if args.num_layers is not None
                        else int(declared.get("num_layers", 2))
                    ),
                    learning_rate=(
                        args.learning_rate
                        if args.learning_rate is not None
                        else float(declared.get("learning_rate", 0.005))
                    ),
                    regularization=(
                        args.regularization
                        if args.regularization is not None
                        else float(declared.get("regularization", 1e-4))
                    ),
                    batch_size=args.batch_size or int(declared.get("batch_size", 8192)),
                    max_epochs=args.epochs or int(declared.get("max_epochs", 30)),
                    negatives_per_positive=(
                        args.negatives_per_positive
                        or int(declared.get("negatives_per_positive", 3))
                    ),
                    evaluation_user_batch_size=int(declared.get("evaluation_user_batch_size", 256)),
                    seed=seed,
                )
                model, fit_measurement = fit_lightgcn(
                    dataset, fit_splits, model_config, device=args.device
                )
                configuration = model_config.to_dict()
                device = model.device
            elif args.model == TWO_TOWER:
                import yaml

                from omnirank.models.two_tower import TwoTowerConfig
                from omnirank.retrieval.runner import fit_two_tower

                # Read from a dedicated YAML rather than the AppConfig tree:
                # the two-tower search space is Phase 5 experimental surface and
                # adding an axis should not require a schema change.
                source = Path(args.model_config or "configs/models/two_tower.yaml")
                if not source.is_file():
                    logger.error("train.model_config_missing", run_id=run_id, expected=str(source))
                    return CONFIG_ERROR_EXIT
                raw = yaml.safe_load(source.read_text()).get("two_tower", {})
                raw["seed"] = seed
                if args.device != "auto":
                    raw["device"] = args.device
                if args.embedding_dim:
                    raw["embedding_dim"] = args.embedding_dim
                if args.epochs:
                    raw["max_epochs"] = args.epochs
                if args.batch_size:
                    raw["batch_size"] = args.batch_size
                if args.learning_rate is not None:
                    raw["learning_rate"] = args.learning_rate
                model_config = TwoTowerConfig(**raw)

                bundle, fit_measurement = fit_two_tower(
                    dataset,
                    fit_splits,
                    model_config,
                    processed_root=Path(dataset_config.processed_dir),
                    device=args.device,
                    subset_users=args.subset_users,
                    max_batches_per_epoch=args.max_batches_per_epoch,
                    validation_splits=("validation",) if args.stage == "development" else (),
                )
                model, feature_store, training_history = bundle
                configuration = model_config.to_dict()
                device = str(next(model.parameters()).device)

                # This milestone delivers the model core only: there is no
                # full-catalogue retrieval path yet, so the shared evaluation
                # harness cannot score it. Persisting and reporting stops here
                # rather than pretending an evaluation happened.
                return _finish_two_tower_development(
                    model,
                    feature_store,
                    training_history,
                    dataset,
                    args,
                    config,
                    logger,
                    run_id,
                    fit_measurement,
                )
            elif args.model == SASREC:
                from omnirank.models.sasrec import SASRecConfig
                from omnirank.retrieval.runner import fit_sasrec

                model_config = SASRecConfig(
                    maximum_sequence_length=(
                        args.maximum_sequence_length
                        or int(declared.get("maximum_sequence_length", 50))
                    ),
                    embedding_dim=args.embedding_dim or int(declared.get("embedding_dim", 64)),
                    num_blocks=args.num_blocks or int(declared.get("num_blocks", 2)),
                    num_heads=args.num_heads or int(declared.get("num_heads", 2)),
                    dropout=(
                        args.dropout
                        if args.dropout is not None
                        else float(declared.get("dropout", 0.2))
                    ),
                    learning_rate=(
                        args.learning_rate
                        if args.learning_rate is not None
                        else float(declared.get("learning_rate", 1e-3))
                    ),
                    batch_size=args.batch_size or int(declared.get("batch_size", 512)),
                    max_epochs=args.epochs or int(declared.get("max_epochs", 15)),
                    negatives_per_positive=(
                        args.negatives_per_positive
                        or int(declared.get("negatives_per_positive", 1))
                    ),
                    evaluation_user_batch_size=int(declared.get("evaluation_user_batch_size", 256)),
                    seed=seed,
                )
                model, fit_measurement = fit_sasrec(
                    dataset,
                    fit_splits,
                    model_config,
                    processed_root=Path(dataset_config.processed_dir),
                    device=args.device,
                )
                configuration = model_config.to_dict()
                device = model.device
            else:
                from omnirank.models.baselines.bpr import BPRConfig

                model_config = BPRConfig(
                    embedding_dim=args.embedding_dim or int(declared.get("embedding_dim", 64)),
                    learning_rate=(
                        args.learning_rate
                        if args.learning_rate is not None
                        else float(declared.get("learning_rate", 0.005))
                    ),
                    regularization=(
                        args.regularization
                        if args.regularization is not None
                        else float(declared.get("regularization", 1e-4))
                    ),
                    batch_size=args.batch_size or int(declared.get("batch_size", 4096)),
                    epochs=args.epochs or int(declared.get("epochs", 20)),
                    negatives_per_positive=(
                        args.negatives_per_positive
                        or int(declared.get("negatives_per_positive", 1))
                    ),
                    evaluation_user_batch_size=int(declared.get("evaluation_user_batch_size", 512)),
                    seed=seed,
                )
                model, fit_measurement = fit_bpr(
                    dataset, fit_splits, model_config, device=args.device
                )
                configuration = model_config.to_dict()
                device = model.device
        except OmniRankError as exc:
            logger.error("train.fit_failed", run_id=run_id, reason=str(exc))
            return TRAINING_ERROR_EXIT

        result = run_experiment(
            model,
            dataset,
            config,
            model_name=args.model,
            model_version=args.version,
            fit_splits=fit_splits,
            target_split=target_split,
            fit_measurement=fit_measurement,
            configuration=configuration,
        )

        if not args.no_register:
            artifact_dir = (
                Path(config.paths.models_dir) / config.data.dataset_name / args.model / args.version
            )
            model.save(artifact_dir)
            registry = ArtifactRegistry(
                Path(config.paths.metadata_dir), artifact_root=Path(config.paths.artifact_root)
            )
            try:
                register_baseline(
                    registry,
                    model_name=args.model,
                    model_version=args.version,
                    artifact_dir=artifact_dir,
                    dataset_identity=dataset.identity.to_dict(),
                    configuration_hash=config.training_config_hash,
                    random_seed=seed,
                    device=device,
                    metrics={
                        # Prefixed by the split they were measured on, so a
                        # validation number can never be mistaken for a test one.
                        f"{target_split}_{key}": value
                        for key, value in result.strict.flat().items()
                        if key in ("recall@20", "ndcg@20", "coverage@20", "novelty@20")
                    },
                    fit_splits=fit_splits,
                    evaluation_protocol="full_catalogue",
                    mapping_checksum=dataset.mapping_metadata.get("item_mapping_checksum", ""),
                    extra_notes=f"stage={args.stage}",
                    overwrite=args.overwrite,
                )
            except OmniRankError as exc:
                logger.error("train.registration_failed", run_id=run_id, reason=str(exc))
                return TRAINING_ERROR_EXIT

        flat = result.strict.flat()
        logger.info(
            "train.completed",
            run_id=run_id,
            model=args.model,
            version=args.version,
            stage=args.stage,
            fit_splits="+".join(fit_splits),
            target_split=target_split,
            **{key: round(flat[key], 6) for key in ("recall@20", "ndcg@20") if key in flat},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
