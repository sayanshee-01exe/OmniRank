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

#: Query batch sizes, so the per-query cost of batching is visible.
BENCHMARK_BATCHES = (1, 32, 256)

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


def benchmark_index(retriever: Any, logger: Any) -> list[dict[str, Any]]:
    """Measure retrieval latency at several depths and batch sizes."""
    users = sorted(retriever._external_to_internal_user)
    internal = [
        retriever._external_to_internal_user[user]
        for user in users
        if retriever._histories.get(retriever._external_to_internal_user[user])
    ][:512]
    if not internal:
        raise OmniRankError("No user with history; cannot benchmark retrieval")

    rows = []
    for batch in BENCHMARK_BATCHES:
        chunk = internal[:batch]
        queries = retriever.build_query_embedding(
            [retriever._histories[user] for user in chunk], chunk
        )
        for depth in BENCHMARK_DEPTHS:
            # One warm-up, then three timed passes: the first call pays for
            # lazy allocation that no later query repeats.
            retriever._search(queries, depth, None)
            samples = []
            for _ in range(3):
                started = time.perf_counter()
                retriever._search(queries, depth, None)
                samples.append((time.perf_counter() - started) * 1000.0)
            total = float(np.median(samples))
            rows.append(
                {
                    "batch_size": batch,
                    "depth": depth,
                    "median_batch_ms": round(total, 3),
                    "median_per_query_ms": round(total / batch, 4),
                    "catalogue_items": len(retriever.catalogue),
                    "dimension": int(retriever.item_embeddings_matrix.shape[1]),
                    "index_type": "exact_brute_force_torch",
                    "device": retriever.device,
                }
            )
            logger.info(
                "phase5.benchmark",
                batch=batch,
                depth=depth,
                per_query_ms=rows[-1]["median_per_query_ms"],
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
            benchmark = benchmark_index(retriever, logger)
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
    single = [row for row in benchmark if row["batch_size"] == 1 and row["depth"] == 200]
    runtime += [
        {
            "stage": "serving",
            "experiment_id": "single_query_depth_200",
            "fold": None,
            "seed": None,
            "training_examples": None,
            "train_seconds": None,
            "evaluation_seconds": round(single[0]["median_batch_ms"] / 1000.0, 6)
            if single
            else None,
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
