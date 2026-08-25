#!/usr/bin/env python
"""Align PixelRec's published text and image vectors to the item mapping.

    python scripts/prepare_multimodal_features.py \
        --data-config configs/data/pixelrec50k.yaml

PixelRec publishes two ~8.6 GiB JSON objects covering all 408,374 full-PixelRec
items. PixelRec50K needs 69,347 of them, so roughly 17 GB is read to keep about
570 MB. Nothing is ever fully parsed: the source is walked incrementally and
only wanted ids are materialised, so peak memory is the output, not the input.

Writes memory-mappable ``.npy`` matrices plus a manifest recording checksums,
dimensions, coverage and identity. A model that loads these must be able to
prove they belong to the mapping it was fitted against, which is what the
manifest is for.

**Encoder identity is not guessed.** PixelRec does not document which encoder
produced these vectors, so they are recorded as `pixelrec_published_text_1024d`
and `pixelrec_published_image_1024d` with `encoder_identity: unknown`. Calling
them CLIP or BERT embeddings would be a claim the source does not support.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from omnirank.artifacts.metadata import detect_git_commit
from omnirank.core.config import load_config
from omnirank.core.exceptions import ConfigurationError, OmniRankError
from omnirank.core.logging import configure_logging, get_logger, run_context
from omnirank.data.pixelrec.features import align_features, write_feature_matrix
from omnirank.evaluation.experiment import measure

CONFIG_ERROR_EXIT = 2
RUN_ERROR_EXIT = 3

#: Feature schema version. Bump when the stored layout or meaning changes in a
#: way that makes an existing store unreadable or incomparable.
FEATURE_VERSION = "1"

MODALITIES = ("text", "image")
#: Recorded instead of an encoder name the source never documents.
FEATURE_NAMES = {
    "text": "pixelrec_published_text_1024d",
    "image": "pixelrec_published_image_1024d",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--data-config", default="configs/data/pixelrec50k.yaml")
    parser.add_argument(
        "--modalities", default="text,image", help="Comma-separated subset to align."
    )
    parser.add_argument(
        "--expected-dimension",
        type=int,
        default=None,
        help="Assert every vector has this width. Omit to take it from the file.",
    )
    parser.add_argument(
        "--compare-float16",
        action="store_true",
        help=(
            "Also measure float16 storage against float32 and report the error, "
            "without changing the stored dtype."
        ),
    )
    return parser.parse_args(argv)


def file_checksum(path: Path, *, block_bytes: int = 8 << 20) -> str:
    """SHA-256 of a file, read in blocks so an 8.6 GiB source fits in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def float16_error(matrix: np.ndarray) -> dict[str, float]:
    """Measure what float16 storage would cost, rather than assuming it is free.

    Halving the store is attractive, but these are raw encoder outputs whose
    scale the source does not document -- values near float16's subnormal range
    would lose precision silently, and a retrieval score is a dot product of
    1024 of them, so per-element error accumulates.
    """
    downcast = matrix.astype("float16").astype("float32")
    absolute = np.abs(matrix - downcast)
    denominator = np.abs(matrix)
    relative = np.divide(absolute, denominator, out=np.zeros_like(absolute), where=denominator > 0)
    sample = matrix[: min(2048, matrix.shape[0])]
    sample_down = downcast[: min(2048, matrix.shape[0])]
    exact = sample @ sample[:1].T
    approximate = sample_down @ sample_down[:1].T
    return {
        "max_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "max_relative_error": float(relative.max()),
        "max_dot_product_error": float(np.abs(exact - approximate).max()),
        "float32_bytes": int(matrix.nbytes),
        "float16_bytes": int(matrix.nbytes // 2),
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
    logger = get_logger("omnirank.prepare_features")
    dataset_config = config.data.dataset
    if dataset_config is None:
        print("The selected profile declares no data.dataset block.", file=sys.stderr)
        return CONFIG_ERROR_EXIT

    processed = Path(dataset_config.processed_dir)
    mappings = Path(config.paths.mappings_dir) / config.data.dataset_name
    features_dir = processed / "features"
    raw_dir = Path(dataset_config.raw_dir)

    mapping_path = mappings / "item_id_mapping.parquet"
    if not mapping_path.is_file():
        logger.error("features.mapping_missing", expected=str(mapping_path))
        return CONFIG_ERROR_EXIT
    item_mapping = pd.read_parquet(mapping_path)
    mapping_metadata = json.loads((mappings / "mapping_metadata.json").read_text())
    mapping_checksum = mapping_metadata.get("item_mapping_checksum", "")

    requested = [name.strip() for name in args.modalities.split(",") if name.strip()]
    manifest: dict[str, Any] = {
        "feature_version": FEATURE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": detect_git_commit() or "unknown",
        "dataset_name": config.data.dataset_name,
        "dataset_version": config.data.dataset_version,
        "item_mapping_checksum": mapping_checksum,
        "mapping_version": mapping_metadata.get("mapping_version"),
        "catalogue_items": len(item_mapping),
        "dtype": "float32",
        "storage": "memory_map",
        "normalization_applied": False,
        "normalization_note": (
            "Raw published vectors. The source documents no normalisation, so none "
            "is claimed and none is applied here; the model normalises its own "
            "outputs instead."
        ),
        "licence": (
            "PixelRec is Westlake Representation Learning Lab, non-commercial "
            "research/education only. Aligned matrices are derived data and are "
            "git-ignored; do not redistribute."
        ),
        "modalities": {},
    }

    with run_context(stage="prepare_multimodal_features") as run_id:
        for modality in requested:
            source = raw_dir / f"{modality}_feature.json"
            logger.info(
                "features.alignment_started",
                run_id=run_id,
                modality=modality,
                source=str(source),
                present=source.is_file(),
            )
            try:
                with measure(f"align_{modality}", track_memory=True) as timer:
                    index, matrix, validation = align_features(
                        modality,
                        source if source.is_file() else None,
                        item_mapping,
                        expected_dimension=args.expected_dimension,
                        # Not guessed. PixelRec does not document the encoder.
                        encoder=None,
                    )
            except OmniRankError as exc:
                logger.error(
                    "features.alignment_failed", run_id=run_id, modality=modality, reason=str(exc)
                )
                return RUN_ERROR_EXIT

            features_dir.mkdir(parents=True, exist_ok=True)
            index.to_parquet(features_dir / f"{modality}_feature_index.parquet", index=False)
            matrix_path = features_dir / f"{modality}_features.npy"
            written = write_feature_matrix(matrix, matrix_path)

            block: dict[str, Any] = {
                "name": FEATURE_NAMES.get(modality, f"pixelrec_published_{modality}"),
                "encoder_identity": "unknown",
                "encoder_note": (
                    "PixelRec publishes these vectors without documenting the encoder. "
                    "Recorded as unknown rather than guessed as CLIP/BERT/SBERT."
                ),
                **validation.to_dict(),
                "index_file": f"{modality}_feature_index.parquet",
                "runtime_seconds": round(timer.result.seconds, 2),
                "peak_memory_mb": round(timer.result.peak_memory_mb, 1),
            }
            if source.is_file():
                block["source_sha256"] = file_checksum(source)
            if written is not None:
                block.update(written)
                block["matrix_file"] = matrix_path.name
                block["matrix_sha256"] = file_checksum(matrix_path)
            if args.compare_float16 and matrix is not None:
                block["float16_comparison"] = float16_error(matrix)

            manifest["modalities"][modality] = block
            logger.info(
                "features.alignment_finished",
                run_id=run_id,
                modality=modality,
                coverage=round(validation.coverage, 6),
                seconds=round(timer.result.seconds, 1),
                peak_memory_mb=round(timer.result.peak_memory_mb, 1),
            )
            del matrix

        mask = _modality_mask(features_dir, item_mapping, requested)
        mask.to_parquet(features_dir / "modality_mask.parquet", index=False)
        manifest["modality_counts"] = {
            "both": int((mask["has_text_feature"] & mask["has_image_feature"]).sum()),
            "text_only": int((mask["has_text_feature"] & ~mask["has_image_feature"]).sum()),
            "image_only": int((~mask["has_text_feature"] & mask["has_image_feature"]).sum()),
            "neither": int((~mask["has_text_feature"] & ~mask["has_image_feature"]).sum()),
        }
        manifest_path = features_dir / "multimodal_feature_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        logger.info(
            "features.manifest_written",
            run_id=run_id,
            path=str(manifest_path),
            **manifest["modality_counts"],
        )
    return 0


def _modality_mask(
    features_dir: Path, item_mapping: pd.DataFrame, modalities: list[str]
) -> pd.DataFrame:
    """One row per catalogue item with a per-modality availability flag."""
    mask = item_mapping.loc[:, ["internal_item_id", "external_item_id"]].copy()
    for modality in MODALITIES:
        column = f"has_{modality}_feature"
        path = features_dir / f"{modality}_feature_index.parquet"
        if modality in modalities and path.is_file():
            index = pd.read_parquet(path)
            mask = mask.merge(
                index.loc[:, ["internal_item_id", column]], on="internal_item_id", how="left"
            )
            mask[column] = mask[column].fillna(value=False).astype(bool)
        else:
            mask[column] = False
    return mask.sort_values("internal_item_id").reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
