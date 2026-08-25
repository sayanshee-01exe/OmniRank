#!/usr/bin/env python
"""Measure candidate aggregation over the fitted retrieval models.

    python scripts/compare_aggregation.py --stage validation

Fits every source **once**, then evaluates each source alone and every
configured blend against them. Fitting dominates the cost here, so refitting per
blend would multiply a one-hour job by the number of strategies for no
additional information.

Every model and every blend is scored by ``run_experiment`` -- the same driver
used for the Phase 3 baselines and the Phase 4 selection runs. A blend evaluated
through its own path would be comparing harnesses as much as strategies.

Also reports the two diagnostics accuracy cannot express: candidate recall (the
ceiling the ranker inherits) and source overlap (whether the ensemble is
actually an ensemble).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from omnirank.core.config import load_config
from omnirank.core.exceptions import ConfigurationError, OmniRankError
from omnirank.core.logging import configure_logging, get_logger, run_context
from omnirank.data.processed import load_processed_dataset
from omnirank.evaluation.reporting import REPORT_ROOT, write_csv, write_json
from omnirank.models.baselines.runner import (
    MATRIX_FACTORIZATION,
    POPULARITY,
    boundary_for_stage,
    fit_bpr,
    fit_popularity,
    run_experiment,
)
from omnirank.retrieval.aggregation import build_aggregator
from omnirank.retrieval.blended import BlendedRetriever
from omnirank.retrieval.diagnostics import candidate_recall, source_overlap
from omnirank.retrieval.runner import (
    LIGHTGCN,
    SASREC,
    TWO_TOWER,
    fit_lightgcn,
    fit_sasrec,
    fit_two_tower,
)

CONFIG_ERROR_EXIT = 2
RUN_ERROR_EXIT = 3

PHASE_ROOT = REPORT_ROOT.parent / "phase_04"
PHASE_3_SELECTION = REPORT_ROOT / "selected_configuration.json"
PHASE_4_SELECTION = PHASE_ROOT / "selected_configuration.json"
PHASE_5_SELECTION = REPORT_ROOT.parent / "phase_05" / "selected_configuration.json"
AGGREGATION_CONFIG = Path("configs/models/aggregation.yaml")

DIAGNOSTIC_DEPTHS = (20, 100, 300)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--data-config", default="configs/data/pixelrec50k.yaml")
    parser.add_argument("--stage", default="selection", choices=("selection", "final"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument(
        "--sources",
        default="popularity,matrix_factorization,lightgcn,sasrec",
        help="Comma-separated sources to fit. Add two_tower for five-source fusion.",
    )
    parser.add_argument(
        "--subset-users",
        type=int,
        default=None,
        help="Restrict two-tower training to the first N users (development).",
    )
    return parser.parse_args(argv)


def _locked(model_name: str) -> dict[str, Any]:
    """Read a model's locked configuration from its phase's selection record."""
    if model_name == TWO_TOWER:
        path = PHASE_5_SELECTION
    elif model_name in (LIGHTGCN, SASREC):
        path = PHASE_4_SELECTION
    else:
        path = PHASE_3_SELECTION
    if not path.is_file():
        raise OmniRankError(
            f"No locked configuration for {model_name}. Run the selection and "
            f"lock stages first (expected {path})."
        )
    record = json.loads(path.read_text()).get(model_name)
    if not record:
        raise OmniRankError(f"{model_name} is absent from {path}")
    return {
        key: value
        for key, value in record.items()
        if not key.startswith("validation_") and key != "inherited_from"
    }


def _fit_sources(
    names: list[str],
    dataset: Any,
    config: Any,
    args: argparse.Namespace,
    fit_splits: tuple[str, ...],
) -> dict[str, Any]:
    """Fit every requested source once, at its locked configuration."""
    processed_root = Path(config.data.dataset.processed_dir)
    fitted: dict[str, Any] = {}
    for name in names:
        locked = _locked(name)
        if name == POPULARITY:
            from omnirank.models.baselines.popularity import PopularityConfig

            model, _ = fit_popularity(
                dataset,
                fit_splits,
                PopularityConfig(
                    variant=locked.get("variant", "time_decay"),
                    half_life_days=float(locked.get("half_life_days", 365.0)),
                ),
            )
        elif name == MATRIX_FACTORIZATION:
            from omnirank.models.baselines.bpr import BPRConfig

            model, _ = fit_bpr(dataset, fit_splits, BPRConfig(**locked), device=args.device)
        elif name == LIGHTGCN:
            from omnirank.models.lightgcn import LightGCNConfig

            model, _ = fit_lightgcn(
                dataset, fit_splits, LightGCNConfig(**locked), device=args.device
            )
        elif name == SASREC:
            from omnirank.models.sasrec import SASRecConfig

            model, _ = fit_sasrec(
                dataset,
                fit_splits,
                SASRecConfig(**locked),
                processed_root=processed_root,
                device=args.device,
            )
        elif name == TWO_TOWER:
            import numpy as np

            from omnirank.models.two_tower import TwoTowerConfig, TwoTowerRetriever
            from omnirank.retrieval.runner import load_item_tags, load_sequences

            # The locked record stores the ablation label beside the
            # hyperparameters; it is provenance, not a constructor argument.
            settings = {k: v for k, v in locked.items() if k not in ("label", "seed")}
            settings.setdefault("seed", 42)
            (network, store, _), _ = fit_two_tower(
                dataset,
                fit_splits,
                TwoTowerConfig(**settings),
                processed_root=processed_root,
                device=args.device,
                subset_users=args.subset_users,
            )
            tags, _ = load_item_tags(processed_root, dataset.num_items)
            sequences = load_sequences(processed_root, fit_splits)
            if args.subset_users is not None:
                sequences = sequences[sequences["internal_user_id"] < args.subset_users]
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
            model = TwoTowerRetriever.from_trained(
                network, store, dataset, histories, warm, tags, device=args.device
            )
            model.export_item_embeddings()
        else:
            raise OmniRankError(f"Unknown source: {name}")
        fitted[name] = model
    return fitted


def _blend_definitions() -> list[dict[str, Any]]:
    """Read the blend grid from configs/models/aggregation.yaml."""
    if not AGGREGATION_CONFIG.is_file():
        raise OmniRankError(f"Missing aggregation config: {AGGREGATION_CONFIG}")
    payload = yaml.safe_load(AGGREGATION_CONFIG.read_text())
    return list(payload.get("experiments", []))


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
    logger = get_logger("omnirank.compare_aggregation")
    if config.data.dataset is None:
        print("The selected profile declares no data.dataset block.", file=sys.stderr)
        return CONFIG_ERROR_EXIT

    requested = [name.strip() for name in args.sources.split(",") if name.strip()]
    fit_splits, target = boundary_for_stage(args.stage)

    with run_context(stage="compare_aggregation") as run_id:
        try:
            dataset = load_processed_dataset(
                Path(config.data.dataset.processed_dir),
                Path(config.paths.mappings_dir) / config.data.dataset_name,
                verify_checksums=not args.skip_checksums,
            )
            logger.info("aggregation.fitting_sources", run_id=run_id, sources=requested)
            sources = _fit_sources(requested, dataset, config, args, fit_splits)
            blends = _blend_definitions()
        except OmniRankError as exc:
            logger.error("aggregation.setup_failed", run_id=run_id, reason=str(exc))
            return RUN_ERROR_EXIT

        rows: list[dict[str, Any]] = []

        # Each source alone, so a blend can be compared with its own best member
        # rather than only with the other blends.
        for name, model in sources.items():
            result = run_experiment(
                model,
                dataset,
                config,
                model_name=name,
                model_version=f"agg-solo-{args.stage}",
                fit_splits=fit_splits,
                target_split=target,
                fit_measurement=None,
                configuration={},
                bootstrap=False,
            )
            flat = result.strict.flat()
            rows.append(
                {
                    "system": name,
                    "kind": "single",
                    "strategy": "-",
                    **{
                        k: round(flat[k], 8)
                        for k in ("recall@20", "ndcg@20", "coverage@20")
                        if k in flat
                    },
                }
            )
            logger.info(
                "aggregation.solo",
                run_id=run_id,
                system=name,
                ndcg=round(flat.get("ndcg@20", 0.0), 6),
            )

        for definition in blends:
            members = [s for s in definition.get("sources", []) if s in sources]
            if len(members) < 2:
                logger.warning(
                    "aggregation.blend_skipped",
                    run_id=run_id,
                    blend=definition.get("name"),
                    detail="Fewer than two of its sources were fitted.",
                    requested=definition.get("sources"),
                    available=sorted(sources),
                )
                continue
            blend = BlendedRetriever(
                {name: sources[name] for name in members},
                build_aggregator(
                    definition["strategy"],
                    source_weights=definition.get("source_weights"),
                    normalization=definition.get("normalization", "rank_percentile"),
                ),
                name=definition["name"],
            )
            result = run_experiment(
                blend,
                dataset,
                config,
                model_name=definition["name"],
                model_version=f"agg-{args.stage}",
                fit_splits=fit_splits,
                target_split=target,
                fit_measurement=None,
                configuration={"sources": members},
                bootstrap=False,
            )
            flat = result.strict.flat()
            rows.append(
                {
                    "system": definition["name"],
                    "kind": "blend",
                    "strategy": definition["strategy"],
                    "sources": "+".join(members),
                    **{
                        k: round(flat[k], 8)
                        for k in ("recall@20", "ndcg@20", "coverage@20")
                        if k in flat
                    },
                }
            )
            logger.info(
                "aggregation.blend",
                run_id=run_id,
                blend=definition["name"],
                ndcg=round(flat.get("ndcg@20", 0.0), 6),
            )
            write_csv(rows, PHASE_ROOT / f"aggregation_comparison_{args.stage}.csv")

        write_csv(rows, PHASE_ROOT / f"aggregation_comparison_{args.stage}.csv")

        # Diagnostics over the raw source lists, at the deepest useful pool.
        depth = max(DIAGNOSTIC_DEPTHS)
        users = sorted(dataset.external_to_internal_users())
        per_source = {name: model.recommend_batch(users, depth) for name, model in sources.items()}
        # The splits carry internal ids; recommendations carry external ones.
        internal_to_external_user = {
            internal: external
            for external, internal in dataset.external_to_internal_users().items()
        }
        internal_to_external_item = dataset.internal_to_external_items()
        targets: dict[str, set[str]] = {}
        for row in dataset.split(target).itertuples():
            external_user = internal_to_external_user.get(int(row.internal_user_id))
            external_item = internal_to_external_item.get(int(row.internal_item_id))
            if external_user is not None and external_item is not None:
                targets.setdefault(external_user, set()).add(external_item)

        # Items any fitted source could return. A target outside this is a
        # coverage gap, not a retrieval failure, and is reported separately.
        reachable = {
            internal_to_external_item[internal]
            for model in sources.values()
            for internal in model.fit_item_catalogue
            if internal in internal_to_external_item
        }

        diagnostics: dict[str, Any] = {}
        if len(per_source) >= 2:
            diagnostics["source_overlap"] = source_overlap(per_source, depth=depth).to_dict()
        if targets:
            # The pool is the union of each source's top-cut, which is what the
            # aggregator had available -- not a concatenation, whose ordering
            # would make the depth cut meaningless.
            diagnostics["candidate_recall"] = [
                candidate_recall(
                    {
                        user: sorted(
                            {
                                item
                                for source in per_source.values()
                                for item in source.get(user, [])[:cut]
                            }
                        )
                        for user in users
                    },
                    targets,
                    depth=cut * len(per_source),
                    reachable_items=reachable,
                ).to_dict()
                for cut in DIAGNOSTIC_DEPTHS
            ]
        write_json(diagnostics, PHASE_ROOT / f"retrieval_diagnostics_{args.stage}.json")
        logger.info("aggregation.complete", run_id=run_id, systems=len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
