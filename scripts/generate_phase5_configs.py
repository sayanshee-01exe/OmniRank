#!/usr/bin/env python
"""Generate the Phase 5 configuration files from the records that own them.

    python scripts/generate_phase5_configs.py
    python scripts/generate_phase5_configs.py --check

Two files, both derived and neither hand-written:

* ``configs/features/pixelrec_published.yaml`` from the real feature manifest,
  so its dimensions, checksums and coverage are what alignment actually
  produced rather than what someone believed at the time.
* ``configs/models/phase5_selected.yaml`` from the locked selection record, so
  the tracked configuration and the experiment that chose it cannot drift.

``--check`` fails when either file disagrees with its source. Hand-editing them
is how a config and the run that justified it quietly stop describing the same
model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnirank.artifacts.metadata import detect_git_commit

MISSING_SOURCE_EXIT = 2
DRIFT_EXIT = 3

PROCESSED = Path("data/processed/pixelrec50k")
FEATURE_MANIFEST = PROCESSED / "features" / "multimodal_feature_manifest.json"
DATASET_MANIFEST = PROCESSED / "dataset_manifest.json"
SELECTION = Path("reports/metrics/phase_05/selected_configuration.json")

FEATURE_CONFIG = Path("configs/features/pixelrec_published.yaml")
SELECTED_CONFIG = Path("configs/models/phase5_selected.yaml")

#: Emitted in this order so regeneration is byte-stable.
MODEL_KEYS = (
    "embedding_dim",
    "text_projection_dim",
    "image_projection_dim",
    "user_id_embedding_dim",
    "item_id_embedding_dim",
    "tag_embedding_dim",
    "hidden_dims",
    "activation",
    "history_pooling",
    "recency_decay",
    "maximum_history_length",
    "modality_fusion",
    "use_text",
    "use_image",
    "use_tag",
    "use_user_id_embedding",
    "use_item_id_residual",
    "l2_normalize",
    "temperature",
    "mask_false_negatives",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "max_epochs",
    "early_stopping_patience",
    "gradient_clip_norm",
    "evaluation_user_batch_size",
    "seed",
)


#: Characters that change a YAML scalar's meaning if left unquoted. A colon is
#: the one that actually bit: a `selected_by` value describing a two-stage
#: process contained ": ", which turns the rest of the line into a nested
#: mapping and makes the whole file unparseable.
_YAML_UNSAFE = (": ", " #", "\n")
_YAML_UNSAFE_PREFIX = (
    "#",
    "- ",
    "? ",
    "! ",
    "& ",
    "* ",
    "@",
    "`",
    "|",
    ">",
    "%",
    "[",
    "{",
    ",",
    '"',
    "'",
)


def _scalar(value: Any) -> str:
    """Render a YAML scalar without pulling in a serialiser.

    Quotes anything whose unquoted form would parse as something other than a
    plain string. Emitting YAML by hand is fine; emitting *invalid* YAML by
    hand is what this guards against.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_scalar(item) for item in value) + "]"
    if isinstance(value, float):
        return f"{value:g}"
    if value is None:
        return "null"
    if isinstance(value, int):
        return str(value)

    text = str(value)
    needs_quoting = (
        not text
        or text.strip() != text
        or any(marker in text for marker in _YAML_UNSAFE)
        or text.startswith(_YAML_UNSAFE_PREFIX)
        or text.endswith(":")
        # A bare word that YAML would read as a bool, null or number is a
        # string here and must stay one.
        or text.lower() in {"true", "false", "yes", "no", "on", "off", "null", "~"}
    )
    if not needs_quoting:
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return f'"{escaped}"'


def render_feature_config(manifest: dict[str, Any], dataset: dict[str, Any]) -> str:
    """Render the feature-source config from the real alignment manifest."""
    text = manifest["modalities"].get("text", {})
    image = manifest["modalities"].get("image", {})
    counts = manifest.get("modality_counts", {})
    return "\n".join(
        [
            "# ---------------------------------------------------------------------------",
            "# PixelRec published multimodal vectors.",
            "#",
            "# GENERATED from data/processed/pixelrec50k/features/"
            "multimodal_feature_manifest.json.",
            "# DO NOT HAND-EDIT. Re-align features, then run:",
            "#     python scripts/generate_phase5_configs.py",
            "#",
            "# The encoder is recorded as `unknown` because PixelRec publishes these",
            "# vectors without documenting how they were produced. Calling them CLIP or",
            "# BERT embeddings would be a claim the source does not support, and every",
            "# downstream comparison would inherit it.",
            "# ---------------------------------------------------------------------------",
            "features:",
            "  name: pixelrec_published_multimodal",
            f"  version: {_scalar(manifest.get('feature_version'))}",
            "",
            "  dataset:",
            f"    name: {_scalar(manifest.get('dataset_name'))}",
            f"    version: {_scalar(manifest.get('dataset_version'))}",
            f"    catalogue_items: {_scalar(manifest.get('catalogue_items'))}",
            "",
            "  text:",
            f"    enabled: {_scalar(bool(text.get('available')))}",
            "    path: data/processed/pixelrec50k/features/text_features.npy",
            f"    dimension: {_scalar(text.get('dimension'))}",
            "    dtype: float32",
            "    encoder_identity: unknown",
            f"    coverage: {_scalar(round(float(text.get('coverage', 0.0)), 6))}",
            f"    items_matched: {_scalar(text.get('rows_matched'))}",
            "",
            "  image:",
            f"    enabled: {_scalar(bool(image.get('available')))}",
            "    path: data/processed/pixelrec50k/features/image_features.npy",
            f"    dimension: {_scalar(image.get('dimension'))}",
            "    dtype: float32",
            "    encoder_identity: unknown",
            f"    coverage: {_scalar(round(float(image.get('coverage', 0.0)), 6))}",
            f"    items_matched: {_scalar(image.get('rows_matched'))}",
            "",
            "  storage:",
            "    type: memory_map",
            "    text_index_path: data/processed/pixelrec50k/features/text_feature_index.parquet",
            "    image_index_path: data/processed/pixelrec50k/features/image_feature_index.parquet",
            "    modality_mask_path: data/processed/pixelrec50k/features/modality_mask.parquet",
            "",
            "  modality_counts:",
            f"    both: {_scalar(counts.get('both'))}",
            f"    text_only: {_scalar(counts.get('text_only'))}",
            f"    image_only: {_scalar(counts.get('image_only'))}",
            f"    neither: {_scalar(counts.get('neither'))}",
            "",
            "  normalization:",
            "    # Raw published vectors. The source documents no normalisation, so",
            "    # none is claimed and none is applied; the model normalises its own",
            "    # outputs instead.",
            "    input_vectors_normalized: false",
            "",
            "  compatibility:",
            f"    feature_manifest_checksum: {_scalar(_manifest_checksum(manifest))}",
            f"    item_mapping_checksum: {_scalar(manifest.get('item_mapping_checksum'))}",
            f"    dataset_manifest_checksum: {_scalar(_dataset_checksum())}",
            f"    dataset_configuration_hash: {_scalar(dataset.get('configuration_hash'))}",
            "",
            "  licence: >-",
            "    PixelRec is Westlake Representation Learning Lab, non-commercial",
            "    research and education only. The aligned matrices are derived data,",
            "    are git-ignored, and must not be redistributed.",
            "",
        ]
    )


def _manifest_checksum(manifest: dict[str, Any]) -> str:
    """Identity hash over the manifest's stable fields.

    Matches ``MultimodalFeatureStore.manifest_checksum`` so the config and the
    store agree on what "the same features" means.
    """
    payload = {
        "feature_version": str(manifest.get("feature_version", "")),
        "item_mapping_checksum": str(manifest.get("item_mapping_checksum", "")),
        "catalogue_items": manifest.get("catalogue_items"),
        "modalities": {
            name: {
                "dimension": block.get("dimension"),
                "source_sha256": block.get("source_sha256"),
                "rows_matched": block.get("rows_matched"),
            }
            for name, block in sorted(manifest.get("modalities", {}).items())
            if block.get("available")
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _dataset_checksum() -> str:
    """SHA-256 of the dataset manifest itself."""
    if not DATASET_MANIFEST.is_file():
        return ""
    return hashlib.sha256(DATASET_MANIFEST.read_bytes()).hexdigest()


def render_selected_config(selection: dict[str, Any]) -> str:
    """Render the locked Phase 5 configuration from the selection record."""
    block = selection["two_tower"]
    identity = selection.get("dataset_identity", {})
    lines = [
        "# ---------------------------------------------------------------------------",
        "# Phase 5 SELECTED configuration.",
        "#",
        "# GENERATED from reports/metrics/phase_05/selected_configuration.json, which",
        "# was locked before any test metric was read. This file records what was",
        "# actually chosen; configs/models/two_tower.yaml holds development defaults",
        "# and is a different thing entirely.",
        "#",
        "# DO NOT HAND-EDIT. Re-select, then run:",
        "#     python scripts/generate_phase5_configs.py",
        "# ---------------------------------------------------------------------------",
        "models:",
        "  candidate_generators:",
        "    two_tower:",
        "      enabled: false",
        "      phase: 5",
        "      top_k: 300",
    ]
    for key in MODEL_KEYS:
        if key in block:
            lines.append(f"      {key}: {_scalar(block[key])}")

    lines += [
        "",
        "selection:",
        f"  ablation: {_scalar(block.get('label'))}",
        f"  selected_by: {_scalar(selection.get('selected_by'))}",
        f"  fit_splits: {_scalar(selection.get('fit_splits'))}",
        f"  target_split: {_scalar(selection.get('target_split'))}",
        f"  configuration_hash: {_scalar(configuration_hash(block))}",
        "",
        "dataset_identity:",
    ]
    for key in sorted(identity):
        lines.append(f"  {key}: {_scalar(identity[key])}")

    lines += [
        "",
        "# Validation metrics recorded at selection time, for provenance only.",
        "# These are NOT test metrics.",
        "#",
        f"#   strict ndcg@20   = {block.get('ndcg@20')}",
        f"#   strict recall@20 = {block.get('recall@20')}",
        f"#   cold ndcg@20     = {block.get('cold_ndcg@20')}",
        "#",
        "# The identity residual was measured to zero cold retrieval AND lower warm",
        "# NDCG, so the selected configuration does not use it. See",
        "# docs/phase_reports/phase_05_report.md.",
        "",
    ]
    return "\n".join(lines)


def configuration_hash(block: dict[str, Any]) -> str:
    """Stable hash over the hyperparameters, excluding recorded metrics."""
    payload = {key: value for key, value in sorted(block.items()) if key in set(MODEL_KEYS)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if a tracked file differs from the record it is derived from.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    outputs: list[tuple[Path, str]] = []

    if not FEATURE_MANIFEST.is_file():
        print(f"Feature manifest not found: {FEATURE_MANIFEST}", file=sys.stderr)
        return MISSING_SOURCE_EXIT
    manifest = json.loads(FEATURE_MANIFEST.read_text())
    dataset = json.loads(DATASET_MANIFEST.read_text()) if DATASET_MANIFEST.is_file() else {}
    outputs.append((FEATURE_CONFIG, render_feature_config(manifest, dataset)))

    if SELECTION.is_file():
        outputs.append((SELECTED_CONFIG, render_selected_config(json.loads(SELECTION.read_text()))))
    elif not args.check:
        print(
            f"No locked selection at {SELECTION}; skipping {SELECTED_CONFIG}. "
            "Run the lock stage first.",
            file=sys.stderr,
        )

    drifted: list[str] = []
    for path, rendered in outputs:
        if args.check:
            current = path.read_text() if path.is_file() else ""
            if current != rendered:
                drifted.append(str(path))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
        print(f"Wrote {path}")

    if args.check:
        if drifted:
            print(
                "Generated configuration is out of date with its source record:\n  "
                + "\n  ".join(drifted)
                + "\nRegenerate with: python scripts/generate_phase5_configs.py",
                file=sys.stderr,
            )
            return DRIFT_EXIT
        print(f"{len(outputs)} generated configuration file(s) match their sources.")
    else:
        print(f"git commit: {detect_git_commit() or 'unknown'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
