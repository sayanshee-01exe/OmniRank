#!/usr/bin/env python
"""Generate the Phase 5 evidence files from the registered final model.

    python scripts/generate_phase5_artifacts.py

Produces, under ``reports/metrics/phase_05/``:

* ``missing_modality_metrics.csv`` -- retrieval quality by modality
  availability. On PixelRec50K every catalogue item has both modalities, so
  three of the four views are empty. They are emitted anyway, with their real
  row counts, because an absent file and an empty view are different claims and
  only one of them is true here.
* ``index_benchmark.csv`` -- exact-index latency at several depths, measured
  rather than quoted.
* ``runtime_metrics.csv`` / ``resource_metrics.csv`` -- what the phase cost.
* ``recommendation_examples.json`` -- worked examples, with surrogate user
  labels. PixelRec's licence forbids redistributing per-user rows, and this
  directory is git-ignored for that reason, but the labels are surrogates in
  any case so nothing reproducible from the file identifies a user.
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
from omnirank.evaluation.reporting import REPORT_ROOT, write_csv, write_json

CONFIG_ERROR_EXIT = 2
RUN_ERROR_EXIT = 3

PHASE_ROOT = REPORT_ROOT.parent / "phase_05"
PROCESSED = Path("data/processed/pixelrec50k")
VERSION = "phase5-two-tower-final"

#: Retrieval depths the latency table reports.
BENCHMARK_DEPTHS = (20, 50, 100, 200, 500)

#: Two float32 scorings of the same query may differ in accumulation order.
SCORE_TOLERANCE = 1e-5

#: Examples are few and hand-readable; the point is inspection, not coverage.
EXAMPLE_USERS = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--data-config", default="configs/data/pixelrec50k.yaml")
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "mps"))
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args(argv)


def _load_retriever(config: Any, version: str, device: str) -> Any:
    """Load the registered final retriever beside its feature store."""
    from omnirank.features.multimodal_store import MultimodalFeatureStore
    from omnirank.models.two_tower import TwoTowerRetriever

    path = Path(config.paths.models_dir) / config.data.dataset_name / "two_tower" / version
    store = MultimodalFeatureStore(PROCESSED / "features")
    return TwoTowerRetriever.load(path, store=store, device=device)


def missing_modality_view(retriever: Any) -> list[dict[str, Any]]:
    """Catalogue composition by modality availability.

    Reports counts rather than retrieval metrics for the empty views: a
    Recall@20 over zero users is not zero, it is undefined, and writing 0.0
    would read as a measured failure of the missing-modality path.
    """
    items = retriever.catalogue.items
    text = items["text_available"].to_numpy(dtype=bool)
    image = items["image_available"].to_numpy(dtype=bool)
    warm = retriever.catalogue.warm_mask

    views = {
        "both_modalities": text & image,
        "text_only": text & ~image,
        "image_only": ~text & image,
        "no_modality": ~text & ~image,
    }
    rows = []
    for name, mask in views.items():
        count = int(mask.sum())
        rows.append(
            {
                "view": name,
                "catalogue_items": count,
                "warm_items": int((mask & warm).sum()),
                "cold_items": int((mask & ~warm).sum()),
                "share_of_catalogue": round(count / max(len(items), 1), 6),
                # Explicitly not a metric. See the docstring.
                "recall@20": "" if count == 0 else "not_evaluated_separately",
                "status": (
                    "empty on this corpus; path verified by fixture only"
                    if count == 0
                    else "present"
                ),
            }
        )
    return rows


def _user_cohorts(retriever: Any, dataset: Any, cold_items: set[int]) -> dict[str, list[int]]:
    """Internal user ids grouped by the property each cohort is meant to probe.

    A single "sample of users" hides the cases most likely to break: a user with
    three interactions queries a nearly-empty history, and a user whose target
    is cold exercises the content-only path that no collaborative source has.
    Reported separately because an average over all of them would drown both.
    """
    histories = getattr(retriever, "_histories", {})
    with_history = [user for user, items in sorted(histories.items()) if items]
    if not with_history:
        return {}

    by_length = sorted(with_history, key=lambda user: len(histories[user]))
    cohorts: dict[str, list[int]] = {
        "sparse_users": [u for u in by_length if len(histories[u]) <= 5][:64],
        "active_users": [u for u in reversed(by_length) if len(histories[u]) >= 20][:64],
        "all_users_sample": with_history[:64],
    }

    # Warm- and cold-target cohorts, from the real held-out split.
    external_user = {
        internal: name for name, internal in dataset.external_to_internal_users().items()
    }
    warm_targets: list[int] = []
    cold_targets: list[int] = []
    with contextlib.suppress(Exception):
        for row in dataset.split("test").itertuples():
            user = int(row.internal_user_id)
            if user not in histories or user not in external_user:
                continue
            bucket = cold_targets if int(row.internal_item_id) in cold_items else warm_targets
            if len(bucket) < 64:
                bucket.append(user)
            if len(warm_targets) >= 64 and len(cold_targets) >= 64:
                break
    if warm_targets:
        cohorts["warm_target_users"] = warm_targets
    if cold_targets:
        cohorts["cold_target_users"] = cold_targets
    return {name: users for name, users in cohorts.items() if users}


def _agreement(
    retriever: Any, embeddings: np.ndarray, users: list[int], depth: int
) -> dict[str, Any]:
    """Index-versus-brute-force agreement for one cohort, and both latencies.

    Set agreement and ordering agreement are reported separately: an index that
    returns the right items in the wrong order is a different (and milder)
    defect from one that returns the wrong items, and a single "matches" flag
    cannot tell them apart.
    """
    from omnirank.retrieval.faiss_index import brute_force_top_k

    histories = retriever._histories
    queries = retriever.build_query_embedding([histories[user] for user in users], list(users))

    started = time.perf_counter()
    index_items, index_scores = retriever._search(queries, depth, None)
    index_seconds = time.perf_counter() - started

    started = time.perf_counter()
    reference_items, reference_scores = brute_force_top_k(embeddings, queries, depth)
    brute_seconds = time.perf_counter() - started

    catalogue = retriever.catalogue.internal_ids
    set_matches = 0
    order_matches = 0
    largest_gap = 0.0
    for row in range(len(users)):
        retrieved = [int(item) for item in index_items[row] if item != -1]
        expected = [int(catalogue[position]) for position in reference_items[row]]
        if set(retrieved) == set(expected):
            set_matches += 1
        if retrieved == expected:
            order_matches += 1
        gaps = np.abs(np.sort(index_scores[row])[::-1] - np.sort(reference_scores[row])[::-1])
        largest_gap = max(largest_gap, float(gaps.max()) if gaps.size else 0.0)

    rows = max(len(users), 1)
    return {
        "users": len(users),
        "depth": depth,
        "topk_set_agreement": round(set_matches / rows, 6),
        "topk_order_agreement": round(order_matches / rows, 6),
        "max_score_difference": f"{largest_gap:.3e}",
        "within_score_tolerance": bool(largest_gap <= SCORE_TOLERANCE),
        "index_median_ms_per_query": round(index_seconds * 1000.0 / rows, 4),
        "brute_force_median_ms_per_query": round(brute_seconds * 1000.0 / rows, 4),
        "speedup_over_brute_force": round(brute_seconds / max(index_seconds, 1e-9), 2),
    }


def _rebuild_and_verify_roundtrip(
    retriever: Any, embeddings: np.ndarray, logger: Any
) -> dict[str, Any]:
    """Time an index build, then check a save/load round trip changes nothing.

    An index that answers differently after being written and read back is
    unusable in serving, where the only index anyone queries is a loaded one.
    The failure is silent: both answers look plausible.
    """
    import tempfile

    from omnirank.retrieval.two_tower_index import build_two_tower_index

    identity = {
        "model_version": VERSION,
        "model_checksum": retriever.catalogue.checksum(),
        "mapping_checksum": retriever.mapping_checksum,
        "feature_version": retriever.store.feature_version,
        "feature_manifest_checksum": retriever.store.manifest_checksum(),
        "normalization": "l2" if retriever.config.l2_normalize else "none",
    }
    started = time.perf_counter()
    index, _ = build_two_tower_index(embeddings, retriever.catalogue, **identity)
    build_seconds = round(time.perf_counter() - started, 2)

    histories = retriever._histories
    probe = [user for user, items in sorted(histories.items()) if items][:32]
    queries = retriever.build_query_embedding([histories[user] for user in probe], probe)
    # `search` returns lists of lists; converted so the comparison below is
    # elementwise rather than a list identity check that would pass trivially.
    before_items, before_scores = (np.asarray(part) for part in index.search(queries, 20))

    with tempfile.TemporaryDirectory() as directory:
        index.save(Path(directory) / "index")
        reloaded = type(index).load(Path(directory) / "index")
        after_items, after_scores = (np.asarray(part) for part in reloaded.search(queries, 20))

    identical = bool(np.array_equal(before_items, after_items))
    difference = float(np.abs(before_scores - after_scores).max()) if before_scores.size else 0.0
    logger.info(
        "phase5.index_roundtrip",
        build_seconds=build_seconds,
        identical=identical,
        max_score_difference=f"{difference:.3e}",
    )
    return {
        "build_seconds": build_seconds,
        "save_load_identical": identical,
        "max_score_difference": f"{difference:.3e}",
    }


def benchmark_index(retriever: Any, dataset: Any, logger: Any) -> list[dict[str, Any]]:
    """Exactness and latency, per user cohort and per depth.

    The point is not the speed number. It is that the index and brute force
    agree: an index built with the wrong metric or over a transposed matrix
    still returns plausible neighbours for every query and never raises.
    """
    embeddings = retriever.item_embeddings_matrix
    cold_items = set(retriever.cold_item_catalogue)
    cohorts = _user_cohorts(retriever, dataset, cold_items)
    if not cohorts:
        raise OmniRankError("No user with history; cannot benchmark retrieval")

    index_dir = Path("artifacts/indexes") / "pixelrec50k" / "two_tower" / VERSION
    index_bytes = sum(f.stat().st_size for f in index_dir.rglob("*") if f.is_file())
    build = _rebuild_and_verify_roundtrip(retriever, embeddings, logger)

    rows: list[dict[str, Any]] = []
    for cohort, users in cohorts.items():
        cold_share = sum(1 for user in users if user in cold_items) if cold_items else 0
        for depth in BENCHMARK_DEPTHS:
            measured = _agreement(retriever, embeddings, users, depth)
            rows.append(
                {
                    "cohort": cohort,
                    **measured,
                    "catalogue_items": len(retriever.catalogue),
                    "cold_items_in_catalogue": len(cold_items),
                    "cold_users_in_cohort": cold_share,
                    "dimension": int(embeddings.shape[1]),
                    "index_type": "IndexFlatIP",
                    "index_bytes_on_disk": index_bytes,
                    "index_build_seconds": build["build_seconds"],
                    "save_load_identical": build["save_load_identical"],
                    "save_load_max_score_difference": build["max_score_difference"],
                    "device": retriever.device,
                }
            )
            logger.info(
                "phase5.benchmark",
                cohort=cohort,
                depth=depth,
                set_agreement=measured["topk_set_agreement"],
                order_agreement=measured["topk_order_agreement"],
                ms_per_query=measured["index_median_ms_per_query"],
            )
    return rows


def recommendation_examples(retriever: Any, dataset: Any) -> dict[str, Any]:
    """Worked examples with surrogate labels and no PixelRec identifiers."""
    metadata_path = PROCESSED / "metadata" / "item_metadata.parquet"
    categories: dict[int, str] = {}
    if metadata_path.is_file():
        import pandas as pd

        frame = pd.read_parquet(metadata_path, columns=["internal_item_id", "category"])
        categories = dict(
            zip(
                frame["internal_item_id"].astype(int),
                frame["category"].fillna("unknown").astype(str),
                strict=True,
            )
        )

    external_item = dataset.internal_to_external_items()
    warm_lookup = dict(
        zip(
            retriever.catalogue.internal_ids.tolist(),
            retriever.catalogue.warm_mask.tolist(),
            strict=True,
        )
    )
    users = [user for user in sorted(retriever._histories) if len(retriever._histories[user]) >= 5][
        :EXAMPLE_USERS
    ]

    examples = []
    for index, internal_user in enumerate(users):
        external_user = next(
            (
                key
                for key, value in retriever._external_to_internal_user.items()
                if value == internal_user
            ),
            None,
        )
        if external_user is None:
            continue
        # `recommend` returns Candidate objects, not raw ids.
        recommended = retriever.recommend(external_user, 10)
        reverse = {value: key for key, value in external_item.items()}
        rows = []
        for rank, candidate in enumerate(recommended, start=1):
            internal_item = reverse.get(candidate.item_id)
            rows.append(
                {
                    "rank": rank,
                    # Surrogate: position in this list, not the PixelRec id.
                    "item_label": f"item_{rank:02d}",
                    "score": round(float(candidate.score), 6),
                    "category": categories.get(int(internal_item or -1), "unknown"),
                    "warm": bool(warm_lookup.get(int(internal_item or -1), False)),
                }
            )
        history = retriever._histories[internal_user][-5:]
        examples.append(
            {
                "user_label": f"user_{chr(ord('A') + index)}",
                "history_length": len(retriever._histories[internal_user]),
                "recent_history_categories": [
                    categories.get(int(item), "unknown") for item in history
                ],
                "recommendations": rows,
                "cold_items_recommended": sum(1 for row in rows if not row["warm"]),
            }
        )

    return {
        "model_version": retriever.metadata().get("model_version", VERSION),
        "note": (
            "Surrogate labels. PixelRec's licence forbids redistributing per-user "
            "or per-item rows, so neither user nor item identifiers appear here; "
            "categories and warm/cold status carry the interpretable content."
        ),
        "examples": examples,
    }


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
    logger = get_logger("omnirank.phase5_artifacts")
    PHASE_ROOT.mkdir(parents=True, exist_ok=True)

    with run_context(stage="phase5_artifacts") as run_id:
        try:
            dataset = load_processed_dataset(
                Path(config.data.dataset.processed_dir),
                Path(config.paths.mappings_dir) / config.data.dataset_name,
                verify_checksums=not args.skip_checksums,
            )
            retriever = _load_retriever(config, args.version, args.device)
        except OmniRankError as exc:
            logger.error("phase5_artifacts.load_failed", run_id=run_id, reason=str(exc))
            return RUN_ERROR_EXIT

        try:
            write_csv(missing_modality_view(retriever), PHASE_ROOT / "missing_modality_metrics.csv")
            benchmark = benchmark_index(retriever, dataset, logger)
            write_csv(benchmark, PHASE_ROOT / "index_benchmark.csv")
            write_json(
                recommendation_examples(retriever, dataset),
                PHASE_ROOT / "recommendation_examples.json",
            )
        except OmniRankError as exc:
            logger.error("phase5_artifacts.failed", run_id=run_id, reason=str(exc))
            return RUN_ERROR_EXIT

        _write_cost_tables(benchmark, logger, run_id)
        logger.info("phase5_artifacts.complete", run_id=run_id, version=args.version)
    return 0


def _write_cost_tables(benchmark: list[dict[str, Any]], logger: Any, run_id: str) -> None:
    """Runtime and resource tables, assembled from what the phase recorded."""
    runs_file = PHASE_ROOT / "rolling_validation_runs.jsonl"
    records: list[dict[str, Any]] = []
    if runs_file.is_file():
        for line in runs_file.read_text().splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                records.append(json.loads(line))

    runtime = [
        {
            "stage": record.get("stage", "unknown"),
            "experiment_id": record.get("experiment_id", record.get("label")),
            "fold": record.get("fold"),
            "seed": record.get("seed"),
            "training_examples": record.get("training_examples"),
            "train_seconds": record.get("train_seconds"),
            "evaluation_seconds": record.get("evaluation_seconds"),
            "epochs_run": record.get("epochs_run"),
            "device": record.get("device"),
        }
        for record in records
    ]
    single = [
        row
        for row in benchmark
        if row.get("cohort") == "all_users_sample" and row.get("depth") == 200
    ]
    runtime += [
        {
            "stage": "serving",
            "experiment_id": "single_query_depth_200",
            "fold": None,
            "seed": None,
            "training_examples": None,
            "train_seconds": None,
            "evaluation_seconds": (
                round(single[0]["index_median_ms_per_query"] / 1000.0, 6) if single else None
            ),
            "epochs_run": None,
            "device": single[0]["device"] if single else None,
        }
    ]
    write_csv(runtime, PHASE_ROOT / "runtime_metrics.csv")

    resource = [
        {
            "stage": record.get("stage", "unknown"),
            "experiment_id": record.get("experiment_id", record.get("label")),
            "peak_memory_mb": record.get("peak_memory_mb"),
            "catalogue_items": record.get("cold_items", 0) + record.get("warm_items", 0),
            "device": record.get("device"),
        }
        for record in records
    ]
    write_csv(resource, PHASE_ROOT / "resource_metrics.csv")
    logger.info(
        "phase5_artifacts.cost_tables",
        run_id=run_id,
        runtime_rows=len(runtime),
        resource_rows=len(resource),
    )


if __name__ == "__main__":
    raise SystemExit(main())
