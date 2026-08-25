#!/usr/bin/env python
"""Build and register a FAISS index over a trained model's item embeddings.

    python scripts/build_index.py --model lightgcn \
        --version phase4-lightgcn-selection --index-type flat_ip --verify-exact

The index is stamped with the model name, model version, embedding checksum and
item mapping checksum. Loading it against anything else is refused (ADR-006): a
mismatched index does not fail, it returns confident nonsense, and every id it
returns resolves to the wrong item.

``--verify-exact`` checks the built index against exact brute force before it is
written. For a flat index the agreement must be total. This is the only check
that separates "fast" from "fast and wrong" -- an index built with the wrong
metric, or over a transposed matrix, still answers every query plausibly.

Building an index does not extend a model's catalogue. Both LightGCN and SASRec
are collaborative, so an item absent from fitting has no embedding and is absent
from the index. No cold-start capability is implied.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from omnirank.core.config import load_config
from omnirank.core.exceptions import ConfigurationError, OmniRankError
from omnirank.core.logging import configure_logging, get_logger, run_context

CONFIG_ERROR_EXIT = 2
RUN_ERROR_EXIT = 3
VERIFICATION_FAILED_EXIT = 4

SUPPORTED_MODELS = ("lightgcn", "sasrec")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--data-config", default="configs/data/pixelrec50k.yaml")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--version", required=True, help="Registered model version.")
    parser.add_argument(
        "--index-type", default="flat_ip", choices=("flat_ip", "flat_l2", "hnsw", "ivf_flat")
    )
    parser.add_argument("--output", default=None, help="Defaults to the configured index dir.")
    parser.add_argument(
        "--verify-exact",
        action="store_true",
        help="Check the index against brute force before writing it.",
    )
    parser.add_argument(
        "--verify-queries",
        type=int,
        default=256,
        help="Query vectors sampled for verification.",
    )
    parser.add_argument("--verify-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _load_model(model_name: str, path: Path) -> Any:
    """Load a trained retrieval model from its artifact directory."""
    if model_name == "lightgcn":
        from omnirank.models.lightgcn import LightGCN

        return LightGCN.load(path, device="cpu")
    from omnirank.models.sasrec import SASRec

    return SASRec.load(path, device="cpu")


def _verify(index: Any, embeddings: np.ndarray, args: argparse.Namespace, logger: Any) -> bool:
    """Check the index against exact brute force. Returns True if it agrees."""
    from omnirank.retrieval.faiss_index import brute_force_top_k

    rng = np.random.default_rng(args.seed)
    sample = rng.choice(
        embeddings.shape[0], size=min(args.verify_queries, embeddings.shape[0]), replace=False
    )
    queries = embeddings[sample]

    found, scores = index.search(queries, args.verify_k)
    expected, expected_scores = brute_force_top_k(embeddings, queries, args.verify_k)

    exact = args.index_type in ("flat_ip", "flat_l2")
    matches = sum(
        1 for row, reference in zip(found, expected.tolist(), strict=True) if row == reference
    )
    agreement = matches / len(found)
    overlap = float(
        np.mean(
            [
                len(set(row) & set(reference)) / max(len(reference), 1)
                for row, reference in zip(found, expected.tolist(), strict=True)
            ]
        )
    )
    max_difference = float(np.abs(np.array(scores) - expected_scores).max())

    logger.info(
        "index.verified",
        index_type=args.index_type,
        queries=len(found),
        exact_order_agreement=round(agreement, 6),
        set_overlap=round(overlap, 6),
        max_score_difference=max_difference,
        exact_expected=exact,
    )
    if exact and agreement < 1.0:
        logger.error(
            "index.verification_failed",
            detail=(
                "A flat index must reproduce brute force exactly. A mismatch "
                "means the index does not represent these embeddings."
            ),
            exact_order_agreement=round(agreement, 6),
        )
        return False
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
    logger = get_logger("omnirank.build_index")

    # Must match the layout scripts/train.py registers under.
    model_path = (
        Path(config.paths.models_dir) / config.data.dataset_name / args.model / args.version
    )
    if not model_path.is_dir():
        logger.error(
            "index.model_not_found",
            detail="Train and register the model first.",
            expected=str(model_path),
        )
        return CONFIG_ERROR_EXIT

    with run_context(stage="build_index", model=args.model, version=args.version) as run_id:
        from omnirank.retrieval.faiss_index import FaissVectorIndex, embedding_checksum

        try:
            model = _load_model(args.model, model_path)
            embeddings = model.item_embeddings()
        except OmniRankError as exc:
            logger.error("index.model_unreadable", run_id=run_id, reason=str(exc))
            return RUN_ERROR_EXIT

        index = FaissVectorIndex(
            index_type=args.index_type,
            metric=config.models.index.metric,
            index_version=config.models.index.index_version,
        )
        try:
            index.build(embeddings, metric=config.models.index.metric)
        except OmniRankError as exc:
            logger.error("index.build_failed", run_id=run_id, reason=str(exc))
            return RUN_ERROR_EXIT

        if args.verify_exact and not _verify(index, embeddings, args, logger):
            return VERIFICATION_FAILED_EXIT

        metadata = model.metadata()
        index.attach_metadata(
            model_name=args.model,
            model_version=args.version,
            item_mapping_checksum=metadata.get("mapping_checksum", ""),
            build_timestamp=datetime.now(UTC).isoformat(),
        )
        destination = Path(
            args.output
            or Path(config.paths.indexes_dir) / config.data.dataset_name / args.model / args.version
        )
        index.save(destination)
        logger.info(
            "index.registered",
            run_id=run_id,
            path=str(destination),
            vectors=index.num_vectors,
            dimension=index.dimension,
            embedding_checksum=embedding_checksum(embeddings)[:16],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
