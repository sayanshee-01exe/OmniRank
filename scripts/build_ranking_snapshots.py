#!/usr/bin/env python
"""Build point-in-time candidate snapshots for ranker training.

    python scripts/build_ranking_snapshots.py --offset 3 --split train
    python scripts/build_ranking_snapshots.py --offset 2 --split validation

For one rolling-fold offset this **refits all five retrievers from scratch on
that fold's pre-cutoff interactions**, retrieves, fuses, labels and writes a
canonical snapshot.

Refitting is the expensive part and the whole point. The cheap alternative --
reusing artifacts fitted on the full training split and merely truncating each
user's history at query time -- is not point-in-time: a model that trained on an
interaction has encoded it in its weights whether or not you feed it back in.
The resulting ranker would be scored against a candidate pool better than
serving can produce, and the error flatters every metric downstream.

Offset 1 is the official test target. ``build_fold`` refuses it outright, so no
invocation of this script can reach it.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from omnirank.core.config import load_config
from omnirank.core.exceptions import ConfigurationError, OmniRankError
from omnirank.core.logging import configure_logging, get_logger, run_context
from omnirank.data.processed import load_processed_dataset
from omnirank.data.rolling import build_fold
from omnirank.ranking.candidate_snapshot import (
    SOURCES,
    RetrieverIdentity,
    SnapshotStats,
    build_manifest,
    build_snapshot_rows,
    guard_overwrite,
    snapshot_checksum,
    validate_snapshot,
    write_manifest,
    write_snapshot_atomically,
)

CONFIG_ERROR_EXIT = 2
RUN_ERROR_EXIT = 3

PROCESSED = Path("data/processed/pixelrec50k")
RANKING_ROOT = PROCESSED / "ranking"

#: Phase 5 measured candidate recall saturating at a per-source budget of 500.
DEFAULT_BUDGET = 500

#: Users scored per retrieval batch. Bounded so a 50,000-user fold does not
#: materialise fifty thousand candidate lists at once.
USER_BATCH = 2000

#: Cap on the fused pool per query. Five sources at budget 500 is up to 2,500
#: candidates before dedup; at 50,000 users that is ~100M rows, which no
#: downstream stage can hold. Serving would truncate the fused list too, so
#: this makes the offline pool the same shape as the online one rather than
#: training the ranker on a depth it will never see.
FUSED_POOL_LIMIT = 600


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--data-config", default="configs/data/pixelrec50k.yaml")
    parser.add_argument(
        "--offset",
        type=int,
        required=True,
        help="Rolling-fold target offset. Offset 1 is the reserved test target.",
    )
    parser.add_argument("--split", required=True, choices=("train", "validation"))
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "mps"))
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Cap the number of queries. For smoke runs only; never for a real snapshot.",
    )
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing snapshot for the same fold and split.",
    )
    parser.add_argument(
        "--allow-source-failure",
        action="store_true",
        help=(
            "DEVELOPMENT ONLY. Continue when a retriever fails, marking the "
            "snapshot degraded. Degraded snapshots are refused by ranker "
            "selection and by final evaluation."
        ),
    )
    return parser.parse_args(argv)


def _fit_sources(
    dataset: Any,
    history: pd.DataFrame,
    targets: pd.DataFrame,
    config: Any,
    args: argparse.Namespace,
    logger: Any,
    run_id: str,
) -> tuple[dict[str, Any], list[RetrieverIdentity]]:
    """Refit all five retrievers on ``history`` alone.

    Every model is constructed fresh. Nothing is loaded from
    ``artifacts/models`` here, deliberately: those were fitted on
    train+validation and knowing about post-cutoff interactions is exactly the
    property this function exists to avoid.
    """
    import yaml

    from omnirank.models.baselines.runner import fit_bpr, fit_popularity
    from omnirank.retrieval.runner import fit_lightgcn, fit_sasrec, fit_two_tower

    mapping = dataset.mapping_metadata.get("item_mapping_checksum", "")
    boundary = f"interactions strictly before rolling offset {args.offset}"
    # The temporal claim, checked the way the fold is actually cut. Rolling
    # folds cut at a per-user offset, not at a wall-clock instant: one user's
    # third-from-last interaction is years away from another's. Comparing a
    # global max-history against a global min-target therefore compares two
    # unrelated users and always fails, which is what an earlier version of
    # this guard did. The meaningful invariant is per user.
    per_user_history = history.groupby("internal_user_id")["timestamp"].max()
    per_user_target = targets.set_index("internal_user_id")["timestamp"]
    aligned = per_user_history.to_frame("max_history").join(
        per_user_target.rename("target"), how="inner"
    )
    violations = int((aligned["max_history"] >= aligned["target"]).sum())
    if violations:
        raise OmniRankError(
            "Some users' fit history reaches their own prediction cutoff; "
            "this is not point-in-time",
            users_violating=violations,
            users_checked=len(aligned),
        )

    max_fit_timestamp = int(history["timestamp"].max())
    # Recorded as the earliest cutoff any query in this fold predicts. It is a
    # summary for the manifest, not the boundary the check above enforces.
    boundary_timestamp = int(per_user_target.min())
    identity_common = {
        "fit_fold": f"offset_{args.offset}",
        "fit_boundary": boundary,
        "fit_boundary_timestamp": boundary_timestamp,
        "max_fit_timestamp": max_fit_timestamp,
        "fit_interactions": len(history),
        "fit_users": int(history["internal_user_id"].nunique()),
        "fit_items": int(history["internal_item_id"].nunique()),
        "dataset_version": str(dataset.identity.to_dict().get("dataset_version", "")),
        "split_version": str(dataset.identity.to_dict().get("split_version", "")),
        "mapping_checksum": mapping,
        "candidate_budget": args.budget,
        "device": args.device,
    }

    models: dict[str, Any] = {}
    identities: list[RetrieverIdentity] = []

    def record(
        source: str, model: Any, configuration_hash: str, seed: int | None, fit_seconds: float
    ) -> None:
        models[source] = model
        identities.append(
            RetrieverIdentity(
                source=source,
                model_class=type(model).__name__,
                model_version=f"pit-offset{args.offset}-{source}",
                configuration_hash=configuration_hash,
                seed=seed,
                fit_seconds=fit_seconds,
                **identity_common,
            )
        )
        logger.info(
            "ranking.source_fitted",
            run_id=run_id,
            source=source,
            offset=args.offset,
            interactions=len(history),
            seconds=round(fit_seconds, 1),
        )

    started = time.perf_counter()
    step = time.perf_counter()
    popularity_config = _locked_config("popularity")
    model, _ = fit_popularity(dataset, ("train",), popularity_config, interactions=history)
    record(
        "popularity",
        model,
        f"halflife{popularity_config.half_life_days:g}",
        getattr(popularity_config, "seed", None),
        time.perf_counter() - step,
    )

    step = time.perf_counter()
    bpr_config = _locked_config("matrix_factorization")
    model, _ = fit_bpr(dataset, ("train",), bpr_config, device=args.device, interactions=history)
    record(
        "matrix_factorization",
        model,
        f"d{bpr_config.embedding_dim}",
        getattr(bpr_config, "seed", None),
        time.perf_counter() - step,
    )

    step = time.perf_counter()
    lightgcn_config = _locked_config("lightgcn")
    model, _ = fit_lightgcn(dataset, ("train",), lightgcn_config, device=args.device, edges=history)
    record(
        "lightgcn",
        model,
        f"L{lightgcn_config.num_layers}",
        getattr(lightgcn_config, "seed", None),
        time.perf_counter() - step,
    )

    # Sequential and content models consume per-user example rows rather than
    # an edge list, built from this fold's history only.
    from omnirank.retrieval.runner import sequences_from_fold

    sequences = sequences_from_fold(_FoldView(history), maximum_history_length=50)
    step = time.perf_counter()
    sasrec_config = _locked_config("sasrec")
    model, _ = fit_sasrec(
        dataset,
        ("train",),
        sasrec_config,
        processed_root=PROCESSED,
        device=args.device,
        sequences=sequences,
    )
    record(
        "sasrec",
        model,
        f"d{sasrec_config.embedding_dim}",
        getattr(sasrec_config, "seed", None),
        time.perf_counter() - step,
    )

    step = time.perf_counter()
    two_tower_config = _two_tower_config()
    (network, store, _), _ = fit_two_tower(
        dataset,
        ("train",),
        two_tower_config,
        processed_root=PROCESSED,
        device=args.device,
        sequences=sequences,
    )
    record(
        "two_tower",
        _wrap_two_tower(network, store, dataset, sequences, args.device),
        two_tower_config.label,
        getattr(two_tower_config, "seed", None),
        time.perf_counter() - step,
    )

    logger.info(
        "ranking.all_sources_fitted",
        run_id=run_id,
        offset=args.offset,
        seconds=round(time.perf_counter() - started, 1),
    )
    del yaml
    return models, identities


class _FoldView:
    """Adapts a plain history frame to what ``sequences_from_fold`` expects."""

    def __init__(self, history: pd.DataFrame) -> None:
        self.history = history
        self.name = "point_in_time"
        # The last event per user becomes that user's training target; the rest
        # is their input history. This mirrors the leave-one-out shape the
        # retrievers were originally fitted under, applied to the fold prefix.
        ordered = history.sort_values(["internal_user_id", "interaction_order"])
        self.targets = ordered.groupby("internal_user_id", as_index=False).tail(1)
        cut = set(
            zip(self.targets["internal_user_id"], self.targets["interaction_order"], strict=True)
        )
        mask = [
            (user, order) not in cut
            for user, order in zip(
                ordered["internal_user_id"], ordered["interaction_order"], strict=True
            )
        ]
        self.history = ordered.loc[mask]


#: Where each source's locked hyperparameters live. Read from the selection
#: record rather than from the tracked YAML defaults, because the record is what
#: the phase actually chose and the YAML is development scaffolding.
LOCKED_SELECTION: Final[dict[str, str]] = {
    "popularity": "reports/metrics/phase_03/selected_configuration.json",
    "matrix_factorization": "reports/metrics/phase_03/selected_configuration.json",
    "lightgcn": "reports/metrics/phase_04/selected_configuration.json",
    "sasrec": "reports/metrics/phase_04/selected_configuration.json",
}


def _locked_config(source: str) -> Any:
    """Build one source's config from its locked selection record.

    Only fields the config class declares are carried over: the record also
    holds the validation metrics that justified the choice, and those are
    provenance rather than hyperparameters.
    """

    record = json.loads(Path(LOCKED_SELECTION[source]).read_text())[source]

    if source == "popularity":
        from omnirank.models.baselines.popularity import PopularityConfig

        cls: Any = PopularityConfig
    elif source == "matrix_factorization":
        from omnirank.models.baselines.bpr import BPRConfig

        cls = BPRConfig
    elif source == "lightgcn":
        from omnirank.models.lightgcn import LightGCNConfig

        cls = LightGCNConfig
    else:
        from omnirank.models.sasrec import SASRecConfig

        cls = SASRecConfig

    accepted = {field.name for field in dataclasses.fields(cls)}
    return cls(**{key: value for key, value in record.items() if key in accepted})


def _two_tower_config() -> Any:
    """The locked Phase 5 two-tower configuration."""
    import yaml

    from omnirank.models.two_tower import TwoTowerConfig

    raw = yaml.safe_load(Path("configs/models/two_tower.yaml").read_text())["two_tower"]
    selected = yaml.safe_load(Path("configs/models/phase5_selected.yaml").read_text())
    block = selected["models"]["candidate_generators"]["two_tower"]

    accepted = {field.name for field in dataclasses.fields(TwoTowerConfig)}
    raw.update({key: value for key, value in block.items() if key in accepted})
    raw["device"] = "cpu"
    return TwoTowerConfig(**raw)


def _wrap_two_tower(
    network: Any, store: Any, dataset: Any, sequences: pd.DataFrame, device: str
) -> Any:
    """Wrap a freshly fitted two-tower network in its retrieval surface."""
    import numpy as np

    from omnirank.models.two_tower import TwoTowerRetriever
    from omnirank.retrieval.runner import load_item_tags

    tags, _ = load_item_tags(PROCESSED, dataset.num_items)
    histories: dict[int, list[int]] = {}
    warm = np.zeros(dataset.num_items, dtype=bool)
    for user, items, target in zip(
        sequences["internal_user_id"],
        sequences["item_sequence"],
        sequences["target_item"],
        strict=True,
    ):
        histories[int(user)] = [int(item) for item in items]
        warm[[int(item) for item in items]] = True
        warm[int(target)] = True
    retriever = TwoTowerRetriever.from_trained(
        network, store, dataset, histories, warm, tags, device=device
    )
    retriever.export_item_embeddings()
    return retriever


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
    logger = get_logger("omnirank.ranking_snapshots")
    RANKING_ROOT.mkdir(parents=True, exist_ok=True)

    with run_context(stage="ranking_snapshots") as run_id:
        try:
            dataset = load_processed_dataset(
                Path(config.data.dataset.processed_dir),
                Path(config.paths.mappings_dir) / config.data.dataset_name,
                verify_checksums=not args.skip_checksums,
            )
            # `build_fold` refuses offset 1, so the reserved test target cannot
            # be reached from here even by a typo.
            fold = build_fold(dataset.fit_interactions(("train", "validation")), offset=args.offset)
        except OmniRankError as exc:
            logger.error("ranking.setup_failed", run_id=run_id, reason=str(exc))
            return RUN_ERROR_EXIT

        destination = RANKING_ROOT / f"{args.split}_candidates.parquet"
        manifest_path = RANKING_ROOT / f"{args.split}_snapshot_manifest.json"
        try:
            guard_overwrite(
                destination,
                manifest_path,
                overwrite=args.overwrite,
                fold_id=f"offset_{args.offset}",
                split=args.split,
            )
        except OmniRankError as exc:
            logger.error("ranking.overwrite_refused", run_id=run_id, reason=str(exc))
            return RUN_ERROR_EXIT

        logger.info(
            "ranking.fold_ready",
            run_id=run_id,
            offset=args.offset,
            history_rows=len(fold.history),
            targets=len(fold.targets),
        )

        started = time.perf_counter()
        tracemalloc.start()
        try:
            models, identities = _fit_sources(
                dataset,
                fold.history,
                fold.targets,
                config,
                args,
                logger,
                run_id,
            )
            frame, stats, failed = _retrieve_and_label(models, dataset, fold, args, logger, run_id)
        except OmniRankError as exc:
            tracemalloc.stop()
            logger.error("ranking.build_failed", run_id=run_id, reason=str(exc))
            return RUN_ERROR_EXIT
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        validate_snapshot(frame)
        for identity in identities:
            if identity.source in failed:
                identity = identity

        manifest = build_manifest(
            fold_id=f"offset_{args.offset}",
            split=args.split,
            stats=stats,
            retrievers=identities,
            dataset_identity=dataset.identity.to_dict(),
            mapping_checksum=dataset.mapping_metadata.get("item_mapping_checksum", ""),
            candidate_budget=args.budget,
            aggregation={"strategy": "reciprocal_rank_fusion", "sources": list(SOURCES)},
            checksum=snapshot_checksum(frame),
            degraded=bool(failed),
            degraded_sources=failed,
            wall_seconds=time.perf_counter() - started,
            peak_memory_mb=peak_bytes / 1e6,
        )

        # Manifest first would leave a record of a snapshot that does not exist
        # if the write failed; data first would leave data nothing can identify.
        # The atomic rename makes the data step all-or-nothing, so the manifest
        # follows it.
        write_snapshot_atomically(frame, destination)
        write_manifest(manifest, manifest_path)
        logger.info(
            "ranking.snapshot_written",
            run_id=run_id,
            path=str(destination),
            degraded=bool(failed),
            degraded_sources=failed,
            **stats.to_dict(),
        )
    return 0


def _retrieve_and_label(
    models: dict[str, Any],
    dataset: Any,
    fold: Any,
    args: argparse.Namespace,
    logger: Any,
    run_id: str,
) -> tuple[pd.DataFrame, SnapshotStats, list[str]]:
    """Retrieve from every source, fuse, and label against the held-out target.

    Returns the frame, its statistics, and the list of sources that failed. A
    failure is fatal unless ``--allow-source-failure`` was given: a snapshot
    missing a retriever describes a candidate pool serving would never produce,
    and a ranker trained on it is fitted to the wrong distribution.
    """
    from omnirank.models.base import Candidate
    from omnirank.retrieval.aggregation import build_aggregator

    external_user = {
        internal: name for name, internal in dataset.external_to_internal_users().items()
    }
    external_item = dataset.internal_to_external_items()
    internal_item = {name: internal for internal, name in external_item.items()}

    # The cutoff policy: a query may use only interactions strictly earlier
    # than the target it is predicting. The target's own timestamp is therefore
    # the exclusive boundary, and it is a real time rather than a fold label.
    targets = fold.targets.set_index("internal_user_id")
    target_items = targets["internal_item_id"].to_dict()
    target_times = targets["timestamp"].to_dict()

    users = [user for user in sorted(target_items) if user in external_user]
    if args.max_users is not None:
        users = users[: args.max_users]

    aggregator = build_aggregator("reciprocal_rank_fusion")
    stats = SnapshotStats()
    blocks: list[pd.DataFrame] = []
    failed_sources: list[str] = []

    for start in range(0, len(users), USER_BATCH):
        chunk = users[start : start + USER_BATCH]
        names = [external_user[user] for user in chunk]
        per_source_lists: dict[str, dict[str, list[Any]]] = {}
        for source, model in models.items():
            try:
                per_source_lists[source] = model.recommend_batch_scored(names, args.budget)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"[:200]
                logger.error(
                    "ranking.source_retrieval_failed",
                    run_id=run_id,
                    source=source,
                    reason=reason,
                )
                if not args.allow_source_failure:
                    raise OmniRankError(
                        f"Required retriever {source!r} failed during snapshot generation. "
                        "Pass --allow-source-failure only for development snapshots.",
                    ) from exc
                if source not in failed_sources:
                    failed_sources.append(source)
                per_source_lists[source] = {}

        rows: list[dict[str, Any]] = []
        for user, name in zip(chunk, names, strict=True):
            target_internal = int(target_items[user])
            target_external = external_item.get(target_internal)
            cutoff = target_times.get(user)
            if target_external is None or cutoff is None:
                continue

            per_source = {
                source: per_source_lists.get(source, {}).get(name, []) for source in SOURCES
            }
            for source, scored in per_source.items():
                stats.source_contributions[source] = stats.source_contributions.get(
                    source, 0
                ) + len(scored)

            candidate_lists = {
                source: [
                    Candidate(
                        item_id=candidate.item_id,
                        # RRF consumes ranks, so the value here only orders the
                        # list; the model's real score travels separately in
                        # `per_source` and lands in its own column.
                        score=1.0 / candidate.rank,
                        sources=(source,),
                    )
                    for candidate in scored
                ]
                for source, scored in per_source.items()
            }
            if not any(candidate_lists.values()):
                stats.queries += 1
                stats.zero_positive_queries += 1
                stats.candidates_per_query.append(0)
                stats.cutoffs.append(int(cutoff))
                continue

            fused_result = aggregator.aggregate(candidate_lists, limit=FUSED_POOL_LIMIT)
            fused = [(item.item_id, float(item.score)) for item in fused_result.candidates]

            query_rows = build_snapshot_rows(
                query_id=f"{args.split}:offset{args.offset}:{name}",
                external_user_id=name,
                internal_user_id=user,
                target_external_item=target_external,
                target_internal_item=target_internal,
                as_of_timestamp=int(cutoff),
                fold_id=f"offset_{args.offset}",
                split=args.split,
                candidate_budget=args.budget,
                per_source=per_source,
                fused=fused,
                external_to_internal_item=internal_item,
            )
            rows.extend(query_rows)
            stats.queries += 1
            stats.rows += len(query_rows)
            stats.candidates_per_query.append(len(query_rows))
            stats.cutoffs.append(int(cutoff))
            if any(row["label"] for row in query_rows):
                stats.positive_queries += 1
            else:
                stats.zero_positive_queries += 1

        if rows:
            blocks.append(pd.DataFrame(rows))
        logger.info(
            "ranking.batch_done",
            run_id=run_id,
            users_done=min(start + USER_BATCH, len(users)),
            users_total=len(users),
            candidate_recall=round(stats.candidate_recall, 6),
        )

    if not blocks:
        raise OmniRankError("No candidates were produced for any query")
    return pd.concat(blocks, ignore_index=True), stats, failed_sources


if __name__ == "__main__":
    raise SystemExit(main())
