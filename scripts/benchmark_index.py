#!/usr/bin/env python
"""Benchmark FAISS index types against exact search.

    python scripts/benchmark_index.py --model lightgcn --version phase4-lightgcn-final

Every approximate index is measured against **`flat_ip`**, never against another
approximation. Recall relative to another approximation says nothing about
whether either is right, and two indexes can agree closely while both being
wrong in the same way.

Reports, per index type and per k: build time, query latency, and recall against
exact. An approximate index that is fast and 0.6-recall is not a faster index,
it is a different and worse retriever, and the table has to make that visible.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import yaml

from omnirank.core.config import load_config
from omnirank.core.exceptions import ConfigurationError, OmniRankError
from omnirank.core.logging import configure_logging, get_logger, run_context
from omnirank.evaluation.reporting import REPORT_ROOT, write_csv

CONFIG_ERROR_EXIT = 2
RUN_ERROR_EXIT = 3

PHASE_ROOT = REPORT_ROOT.parent / "phase_04"
BENCHMARK_CONFIG = Path("configs/models/faiss.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--data-config", default="configs/data/pixelrec50k.yaml")
    parser.add_argument("--model", required=True, choices=("lightgcn", "sasrec"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--queries", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _recall_against_exact(found: list[list[int]], exact: np.ndarray) -> float:
    """Mean fraction of the exact top-k that the index also returned."""
    return float(
        np.mean(
            [
                len(set(row) & set(reference)) / max(len(reference), 1)
                for row, reference in zip(found, exact.tolist(), strict=True)
            ]
        )
    )


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
    logger = get_logger("omnirank.benchmark_index")

    model_path = (
        Path(config.paths.models_dir) / config.data.dataset_name / args.model / args.version
    )
    if not model_path.is_dir():
        logger.error("benchmark.model_not_found", expected=str(model_path))
        return CONFIG_ERROR_EXIT

    settings = yaml.safe_load(BENCHMARK_CONFIG.read_text()).get("benchmark", {})
    candidates = settings.get("candidates", [{"type": "flat_ip"}])
    k_values = settings.get("k_values", [10, 20, 100])

    with run_context(stage="benchmark_index", model=args.model) as run_id:
        from omnirank.retrieval.faiss_index import FaissVectorIndex, brute_force_top_k

        if args.model == "lightgcn":
            from omnirank.models.lightgcn import LightGCN

            model: Any = LightGCN.load(model_path, device="cpu")
            queries_all = model.user_embeddings()
        else:
            from omnirank.models.sasrec import SASRec

            model = SASRec.load(model_path, device="cpu")
            queries_all = model.item_embeddings()

        embeddings = model.item_embeddings()
        rng = np.random.default_rng(args.seed)
        sample = rng.choice(
            queries_all.shape[0], size=min(args.queries, queries_all.shape[0]), replace=False
        )
        queries = np.ascontiguousarray(queries_all[sample])

        exact_by_k = {k: brute_force_top_k(embeddings, queries, k)[0] for k in k_values}

        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            index = FaissVectorIndex(
                index_type=candidate["type"],
                build_parameters=candidate.get("build_parameters"),
            )
            started = time.perf_counter()
            try:
                index.build(embeddings)
            except OmniRankError as exc:
                logger.error(
                    "benchmark.build_failed",
                    run_id=run_id,
                    index_type=candidate["type"],
                    reason=str(exc),
                )
                continue
            build_seconds = time.perf_counter() - started

            for k in k_values:
                started = time.perf_counter()
                found, _ = index.search(queries, k)
                query_seconds = time.perf_counter() - started
                recall = _recall_against_exact(found, exact_by_k[k])
                rows.append(
                    {
                        "index_type": candidate["type"],
                        "k": k,
                        "vectors": index.num_vectors,
                        "dimension": index.dimension,
                        "build_seconds": round(build_seconds, 4),
                        "queries": len(found),
                        "total_query_seconds": round(query_seconds, 5),
                        "microseconds_per_query": round(query_seconds / len(found) * 1e6, 2),
                        "recall_vs_exact": round(recall, 6),
                        "is_exact": candidate["type"] in ("flat_ip", "flat_l2"),
                    }
                )
                logger.info(
                    "benchmark.measured",
                    run_id=run_id,
                    index_type=candidate["type"],
                    k=k,
                    recall_vs_exact=round(recall, 6),
                    microseconds_per_query=round(query_seconds / len(found) * 1e6, 2),
                )
            write_csv(rows, PHASE_ROOT / f"index_benchmark_{args.model}.csv")

        write_csv(rows, PHASE_ROOT / f"index_benchmark_{args.model}.csv")
        logger.info("benchmark.complete", run_id=run_id, measurements=len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
