"""Dataset preparation pipeline - orchestration only.

Every stage lives in its own module and is independently testable. This file
does nothing but call them in order, wire their outputs together, and write the
results. If a stage's logic starts appearing here, it belongs in that stage's
module instead.

Stage order, and why it is this order:

1. **inspect / validate** - fail on a missing or wrong-shaped file before doing
   any expensive work.
2. **profile raw** - describe the source *before* cleaning, so every later
   "we removed N rows" claim has a baseline.
3. **canonicalize** - source vocabulary to OmniRank vocabulary.
4. **clean** - reject unusable rows, preserving every rejection.
5. **filter** - iterative k-core, once, on the whole log.
6. **map ids** - after filtering, so no dense index is spent on a removed entity.
7. **split** - per-user leave-last-N, after mapping so splits share one id space.
8. **derive** - graph, sequences, statistics, slices; all training-only where
   the statistic could otherwise encode the future.
9. **check leakage** - and abort the build on any critical failure.
10. **write** - outputs, reports, manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from omnirank.core.config import AppConfig
from omnirank.core.exceptions import ConfigurationError, DataError
from omnirank.core.logging import get_logger, run_context
from omnirank.data import leakage as leakage_checks
from omnirank.data.cleaning import RejectedRecords, clean_interactions, clean_items
from omnirank.data.filtering import apply_iterative_filtering
from omnirank.data.io import write_json, write_parquet, write_text
from omnirank.data.manifest import MANIFEST_FILENAME, PIPELINE_VERSION, build_manifest
from omnirank.data.mapping import (
    MAPPING_VERSION,
    DatasetMappings,
    build_dataset_mappings,
    write_mappings,
)
from omnirank.data.pixelrec.canonical import (
    canonicalize_interactions,
    canonicalize_items,
    derive_users,
)
from omnirank.data.pixelrec.features import align_features, write_feature_matrix
from omnirank.data.pixelrec.loaders import PixelRec50KLoader
from omnirank.data.profiling import (
    profile_processed,
    profile_raw,
    render_processed_profile_markdown,
    render_raw_profile_markdown,
    write_distribution_csv,
)
from omnirank.data.sequences import build_all_sequences
from omnirank.data.slices import build_all_slices
from omnirank.data.splitters import SPLIT_NAMES, TRAIN, split_leave_last_n
from omnirank.data.statistics import build_item_popularity, build_user_statistics

logger = get_logger(__name__)

#: Bumped when the split protocol changes in a way that makes two splits
#: incomparable. Recorded in split metadata and in the manifest.
SPLIT_VERSION = "1"

COLLABORATIVE_COLUMNS = (
    "internal_user_id",
    "internal_item_id",
    "interaction_order",
    "event_type",
    "interaction_weight",
    "split",
)

GRAPH_COLUMNS = ("internal_user_id", "internal_item_id", "edge_weight", "interaction_order")

ITEM_METADATA_COLUMNS = (
    "internal_item_id",
    "external_item_id",
    "title",
    "description",
    "category",
    "image_reference",
    "text_feature_reference",
    "image_feature_reference",
    "source_metadata",
)


@dataclass(slots=True)
class PipelineOptions:
    """Run-time switches that do not belong in the dataset profile."""

    overwrite: bool = False
    validate_only: bool = False
    profile_only: bool = False
    subset_users: int | None = None


@dataclass(slots=True)
class PipelineResult:
    """Everything the run produced, for the report and for tests."""

    outputs: dict[str, Any] = field(default_factory=dict)
    reports: dict[str, Any] = field(default_factory=dict)
    manifest_path: Path | None = None
    leakage_passed: bool = True
    counts: dict[str, int] = field(default_factory=dict)


def _require_dataset_config(config: AppConfig) -> Any:
    """Return the dataset path block, or fail with an actionable message."""
    if config.data.dataset is None:
        raise ConfigurationError(
            "The selected domain profile declares no `data.dataset` block, so the "
            "pipeline does not know where its files live. Run with "
            "--config configs/data/pixelrec50k.yaml, or add a dataset block to the profile.",
            domain=config.data.domain,
        )
    return config.data.dataset


def run_pipeline(
    config: AppConfig,
    options: PipelineOptions | None = None,
    *,
    project_root: Path | str = ".",
) -> PipelineResult:
    """Execute the full dataset preparation pipeline.

    Args:
        config: Validated configuration carrying the selected domain profile.
        options: Run-time switches (overwrite, validate-only, subset).
        project_root: Root that relative configured paths resolve against.

    Returns:
        A :class:`PipelineResult` describing every output and report.

    Raises:
        ConfigurationError: The profile has no dataset block.
        DataError: A stage failed, or a critical leakage check did not pass.
    """
    options = options or PipelineOptions()
    root = Path(project_root)
    dataset = _require_dataset_config(config)
    data_config = config.data
    overwrite = options.overwrite or data_config.processing.overwrite
    subset_users = options.subset_users or data_config.processing.subset_users

    raw_dir = root / dataset.raw_dir
    interim_dir = root / dataset.interim_dir
    processed_dir = root / dataset.processed_dir
    mappings_dir = root / config.paths.mappings_dir / data_config.dataset_name
    reports_dir = root / "reports" / "data_quality" / data_config.dataset_name

    result = PipelineResult()

    with run_context(stage="prepare_data", dataset=data_config.dataset_name) as run_id:
        logger.info(
            "pipeline.started",
            run_id=run_id,
            dataset=f"{data_config.dataset_name}@{data_config.dataset_version}",
            pipeline_version=PIPELINE_VERSION,
            subset_users=subset_users,
            validate_only=options.validate_only,
            profile_only=options.profile_only,
        )

        # -- 1. inspect and validate source files --------------------------- #
        loader = PixelRec50KLoader(
            raw_dir,
            chunk_size=data_config.processing.chunk_size,
            compute_checksums=data_config.processing.validate_checksums,
            subset_users=subset_users,
        )
        loader.check_files_present()
        if options.validate_only:
            logger.info("pipeline.validate_only_complete", run_id=run_id)
            result.reports["validate_only"] = True
            return result

        raw = loader.load()

        # -- 2/3. canonicalize, then profile the source as delivered -------- #
        canonical_interactions = canonicalize_interactions(raw.interactions)
        canonical_items = canonicalize_items(raw.items)

        raw_profile = profile_raw(
            canonical_interactions,
            canonical_items,
            dataset_name=data_config.dataset_name,
            source_files=[file.to_dict() for file in raw.source_files],
        )
        raw_reports_dir = reports_dir / "raw"
        result.reports["raw_profile.json"] = write_json(
            raw_profile.to_dict(), raw_reports_dir / "raw_profile.json", overwrite=True
        )
        result.reports["raw_profile.md"] = write_text(
            render_raw_profile_markdown(raw_profile),
            raw_reports_dir / "raw_profile.md",
            overwrite=True,
        )
        for name, frame in (
            ("missingness.csv", raw_profile.missingness),
            ("user_activity.csv", raw_profile.user_activity),
            ("item_popularity.csv", raw_profile.item_popularity),
            ("feature_coverage.csv", raw_profile.feature_coverage),
        ):
            write_distribution_csv(frame, raw_reports_dir / name)

        if options.profile_only:
            logger.info("pipeline.profile_only_complete", run_id=run_id)
            result.counts = {
                "raw_interactions": len(canonical_interactions),
                "raw_items": len(canonical_items),
            }
            return result

        # -- 4. clean -------------------------------------------------------- #
        sink = RejectedRecords()
        cleaned_items, item_step = clean_items(canonical_items, sink)
        cleaned_interactions, interaction_step = clean_interactions(
            canonical_interactions,
            set(cleaned_items["external_item_id"].astype(str)),
            sink,
            min_timestamp=pd.Timestamp(data_config.validation.min_timestamp),
            max_timestamp=pd.Timestamp.now(tz="UTC"),
            allowed_event_types=set(data_config.event_types),
            drop_duplicates=data_config.validation.drop_duplicate_events,
        )
        rejected = sink.to_frame()
        write_distribution_csv(rejected, raw_reports_dir / "validation_failures.csv")

        cleaning_report = {
            "steps": [item_step.to_dict(), interaction_step.to_dict()],
            "total_rejected": len(rejected),
            "rejected_by_reason": (
                rejected["rejection_reason"].value_counts().to_dict() if len(rejected) else {}
            ),
        }

        # -- 5. filter ------------------------------------------------------- #
        filtering = apply_iterative_filtering(
            cleaned_interactions,
            enabled=data_config.filtering.enabled,
            min_user_interactions=data_config.filtering.min_interactions_per_user,
            min_item_interactions=data_config.filtering.min_interactions_per_item,
        )
        filtered = filtering.interactions
        filtering_report = filtering.report()

        filter_reports_dir = reports_dir / "filtering"
        result.reports["filtering_report.json"] = write_json(
            filtering_report, filter_reports_dir / "filtering_report.json", overwrite=True
        )
        result.reports["filtering_report.md"] = write_text(
            _render_filtering_markdown(filtering_report),
            filter_reports_dir / "filtering_report.md",
            overwrite=True,
        )

        if filtered.empty:
            raise DataError("Filtering produced an empty interaction log")

        # -- 6. id mappings (post-filtering population) ---------------------- #
        mappings = build_dataset_mappings(filtered, dataset_version=data_config.dataset_version)
        mapped = mappings.attach_internal_ids(filtered)
        mapping_outputs = write_mappings(mappings, mappings_dir, overwrite=True)
        result.outputs.update({f"mappings/{k}": v for k, v in mapping_outputs.items()})

        # -- 7. split -------------------------------------------------------- #
        split_result = split_leave_last_n(mapped, data_config.splitting)
        frame = split_result.interactions

        # -- 8. derived datasets --------------------------------------------- #
        item_popularity = build_item_popularity(
            frame, long_tail_quantile=data_config.slices.long_tail_quantile
        )
        user_statistics = build_user_statistics(frame, item_popularity)
        sequence_frames, sequence_stats = build_all_sequences(
            frame,
            maximum_length=data_config.sequences.max_length,
            minimum_length=data_config.sequences.min_length,
        )
        graph_edges = _build_graph_edges(frame)

        item_metadata = _build_item_metadata(cleaned_items, mappings)
        text_index, text_matrix, text_validation = align_features(
            "text",
            (raw_dir / data_config.features.text_feature_file)
            if data_config.features.validate_text_features
            else None,
            mappings.items.frame,
            expected_dimension=data_config.features.expected_dimension,
            encoder=data_config.features.text_encoder,
        )
        image_index, image_matrix, image_validation = align_features(
            "image",
            (raw_dir / data_config.features.image_feature_file)
            if data_config.features.validate_image_features
            else None,
            mappings.items.frame,
            expected_dimension=data_config.features.expected_dimension,
            encoder=data_config.features.image_encoder,
        )

        slice_frames, slice_definitions = build_all_slices(
            frame,
            user_statistics=user_statistics,
            item_popularity=item_popularity,
            item_metadata=item_metadata,
            text_index=text_index,
            image_index=image_index,
            long_tail_quantile=data_config.slices.long_tail_quantile,
        )

        # -- 9. leakage ------------------------------------------------------ #
        leakage_report = leakage_checks.run_all_checks(
            frame,
            sequences=sequence_frames,
            graph_edges=graph_edges,
            item_popularity=item_popularity,
            user_statistics=user_statistics,
            feature_frames={
                "item_training_popularity": item_popularity,
                "user_training_statistics": user_statistics,
                "text_feature_index": text_index,
                "image_feature_index": image_index,
            },
        )
        leakage_dir = reports_dir / "leakage"
        result.reports["leakage_report.json"] = write_json(
            leakage_report.to_dict(), leakage_dir / "leakage_report.json", overwrite=True
        )
        result.reports["leakage_report.md"] = write_text(
            _render_leakage_markdown(leakage_report.to_dict()),
            leakage_dir / "leakage_report.md",
            overwrite=True,
        )
        result.leakage_passed = leakage_report.passed
        # Fails the build. A dataset with critical leakage produces optimistic
        # metrics that nobody can reproduce, and shipping it is worse than
        # shipping nothing.
        leakage_report.raise_if_failed()

        # -- 10. write outputs ----------------------------------------------- #
        result.outputs.update(
            _write_outputs(
                frame=frame,
                interim_dir=interim_dir,
                processed_dir=processed_dir,
                users=derive_users(cleaned_interactions),
                items=cleaned_items,
                rejected=rejected,
                graph_edges=graph_edges,
                sequence_frames=sequence_frames,
                item_metadata=item_metadata,
                user_statistics=user_statistics,
                item_popularity=item_popularity,
                text_index=text_index,
                image_index=image_index,
                text_matrix=text_matrix,
                image_matrix=image_matrix,
                slice_frames=slice_frames,
                overwrite=overwrite,
            )
        )

        split_metadata = {
            **split_result.statistics(),
            "split_version": SPLIT_VERSION,
            "dataset_version": data_config.dataset_version,
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "configuration_hash": config.training_config_hash,
            "random_seed": config.seed,
        }
        result.outputs["split_metadata.json"] = write_json(
            split_metadata, processed_dir / "split_metadata.json", overwrite=True
        )

        # -- processed profile ------------------------------------------------ #
        processed_profile = profile_processed(
            frame,
            raw_profile=raw_profile,
            cleaning_report=cleaning_report,
            filtering_report=filtering_report,
            split_statistics=split_metadata,
            sequence_statistics=[stats.to_dict() for stats in sequence_stats],
            feature_validations=[text_validation.to_dict(), image_validation.to_dict()],
            slice_definitions=[definition.to_dict() for definition in slice_definitions],
            leakage_report=leakage_report.to_dict(),
        )
        processed_reports_dir = reports_dir / "processed"
        result.reports["processed_profile.json"] = write_json(
            processed_profile.to_dict(),
            processed_reports_dir / "processed_profile.json",
            overwrite=True,
        )
        result.reports["processed_profile.md"] = write_text(
            render_processed_profile_markdown(processed_profile),
            processed_reports_dir / "processed_profile.md",
            overwrite=True,
        )

        # -- manifest --------------------------------------------------------- #
        manifest = build_manifest(
            dataset_name=data_config.dataset_name,
            dataset_version=data_config.dataset_version,
            source_repository=dataset.source_repository,
            licence=dataset.licence,
            source_files=raw.provenance,
            configuration_hash=config.training_config_hash,
            random_seed=config.seed,
            mapping_version=MAPPING_VERSION,
            split_version=SPLIT_VERSION,
            split_strategy=split_result.strategy,
            ordering_field=split_result.ordering_field,
            filtering_configuration=filtering_report["configuration"],
            raw_row_counts={
                "interactions": len(canonical_interactions),
                "items": len(canonical_items),
            },
            processed_row_counts={
                "interactions": len(frame),
                "items": len(item_metadata),
                "users": int(frame["internal_user_id"].nunique()),
            },
            user_counts={
                "raw": raw_profile.interactions.get("unique_users", 0),
                "processed": int(frame["internal_user_id"].nunique()),
            },
            item_counts={
                "raw": raw_profile.interactions.get("unique_items", 0),
                "processed": int(frame["internal_item_id"].nunique()),
            },
            interaction_counts=split_result.sizes,
            feature_dimensions={
                "text": text_validation.dimension,
                "image": image_validation.dimension,
            },
            feature_coverage={
                "text": round(text_validation.coverage, 6),
                "image": round(image_validation.coverage, 6),
            },
            output_files=result.outputs,
            known_limitations=_known_limitations(text_validation, image_validation, subset_users),
            repo_root=root,
            subset_users=subset_users,
        )
        manifest_descriptor = write_json(
            manifest.to_dict(), processed_dir / MANIFEST_FILENAME, overwrite=True
        )
        result.manifest_path = Path(manifest_descriptor["path"])
        result.counts = {
            "users": int(frame["internal_user_id"].nunique()),
            "items": int(frame["internal_item_id"].nunique()),
            "interactions": len(frame),
            **split_result.sizes,
        }
        logger.info("pipeline.completed", run_id=run_id, **result.counts)

    return result


def _build_graph_edges(frame: pd.DataFrame) -> pd.DataFrame:
    """Build user-item graph edges from training interactions only.

    Edge weights are **binary** (1.0). PixelRec records a single implicit event
    type with no intensity, so a count-based or confidence weight would express
    a signal the source does not carry. Repeated (user, item) pairs are
    aggregated by keeping the earliest interaction order and the raw repeat
    count, so a later weighting scheme has the data it needs.
    """
    train = frame[frame["split"] == TRAIN]
    if train.empty:
        return pd.DataFrame(columns=[*GRAPH_COLUMNS, "interaction_count"])
    grouped = train.groupby(["internal_user_id", "internal_item_id"], observed=True)
    edges = grouped.agg(
        interaction_order=("interaction_order", "min"),
        interaction_count=("interaction_order", "size"),
    ).reset_index()
    edges["edge_weight"] = 1.0
    return edges.loc[:, [*GRAPH_COLUMNS, "interaction_count"]]


def _build_item_metadata(items: pd.DataFrame, mappings: DatasetMappings) -> pd.DataFrame:
    """Join canonical item metadata onto internal ids, catalogue order."""
    merged = mappings.items.frame.merge(items, on="external_item_id", how="left")
    for column in ITEM_METADATA_COLUMNS:
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged.loc[:, list(ITEM_METADATA_COLUMNS)]


def _write_outputs(
    *,
    frame: pd.DataFrame,
    interim_dir: Path,
    processed_dir: Path,
    users: pd.DataFrame,
    items: pd.DataFrame,
    rejected: pd.DataFrame,
    graph_edges: pd.DataFrame,
    sequence_frames: dict[str, pd.DataFrame],
    item_metadata: pd.DataFrame,
    user_statistics: pd.DataFrame,
    item_popularity: pd.DataFrame,
    text_index: pd.DataFrame,
    image_index: pd.DataFrame,
    text_matrix: Any,
    image_matrix: Any,
    slice_frames: dict[str, pd.DataFrame],
    overwrite: bool,
) -> dict[str, Any]:
    """Write every processed artifact and return their descriptors."""
    outputs: dict[str, Any] = {}

    # interim: the canonical entities, before filtering and splitting
    outputs["interim/canonical_users.parquet"] = write_parquet(
        users,
        interim_dir / "canonical_users.parquet",
        sort_by=["external_user_id"],
        overwrite=overwrite,
    )
    outputs["interim/canonical_items.parquet"] = write_parquet(
        items,
        interim_dir / "canonical_items.parquet",
        sort_by=["external_item_id"],
        overwrite=overwrite,
    )
    outputs["interim/canonical_interactions.parquet"] = write_parquet(
        frame,
        interim_dir / "canonical_interactions.parquet",
        sort_by=["external_user_id", "interaction_order"],
        overwrite=overwrite,
    )
    outputs["interim/rejected_records.parquet"] = write_parquet(
        rejected,
        interim_dir / "rejected_records.parquet",
        overwrite=overwrite,
    )

    # processed: split interaction tables
    for split in SPLIT_NAMES:
        subset = frame[frame["split"] == split]
        outputs[f"{split}_interactions.parquet"] = write_parquet(
            subset,
            processed_dir / f"{split}_interactions.parquet",
            columns=COLLABORATIVE_COLUMNS,
            sort_by=["internal_user_id", "interaction_order"],
            overwrite=overwrite,
        )
    outputs["collaborative/interactions.parquet"] = write_parquet(
        frame,
        processed_dir / "collaborative" / "interactions.parquet",
        columns=COLLABORATIVE_COLUMNS,
        sort_by=["internal_user_id", "interaction_order"],
        overwrite=overwrite,
    )

    outputs["graph/train_graph_edges.parquet"] = write_parquet(
        graph_edges,
        processed_dir / "graph" / "train_graph_edges.parquet",
        sort_by=["internal_user_id", "internal_item_id"],
        overwrite=overwrite,
    )

    for split, sequences in sequence_frames.items():
        outputs[f"sequential/{split}_sequences.parquet"] = write_parquet(
            sequences,
            processed_dir / "sequential" / f"{split}_sequences.parquet",
            sort_by=["internal_user_id", "target_order"],
            overwrite=overwrite,
        )

    outputs["metadata/item_metadata.parquet"] = write_parquet(
        item_metadata,
        processed_dir / "metadata" / "item_metadata.parquet",
        columns=ITEM_METADATA_COLUMNS,
        sort_by=["internal_item_id"],
        overwrite=overwrite,
    )

    features_dir = processed_dir / "features"
    outputs["features/user_training_statistics.parquet"] = write_parquet(
        user_statistics,
        features_dir / "user_training_statistics.parquet",
        sort_by=["internal_user_id"],
        overwrite=overwrite,
    )
    outputs["features/item_training_popularity.parquet"] = write_parquet(
        item_popularity,
        features_dir / "item_training_popularity.parquet",
        sort_by=["internal_item_id"],
        overwrite=overwrite,
    )
    outputs["features/text_feature_index.parquet"] = write_parquet(
        text_index,
        features_dir / "text_feature_index.parquet",
        sort_by=["internal_item_id"],
        overwrite=overwrite,
    )
    outputs["features/image_feature_index.parquet"] = write_parquet(
        image_index,
        features_dir / "image_feature_index.parquet",
        sort_by=["internal_item_id"],
        overwrite=overwrite,
    )
    for modality, matrix in (("text", text_matrix), ("image", image_matrix)):
        descriptor = write_feature_matrix(matrix, features_dir / f"{modality}_features.npy")
        if descriptor is not None:
            outputs[f"features/{modality}_features.npy"] = descriptor

    slices_dir = processed_dir / "evaluation_slices"
    slice_manifest: list[dict[str, Any]] = []
    for name, slice_frame in slice_frames.items():
        descriptor = write_parquet(
            slice_frame,
            slices_dir / f"{name}.parquet",
            sort_by=["entity_id"],
            overwrite=overwrite,
        )
        outputs[f"evaluation_slices/{name}.parquet"] = descriptor
        slice_manifest.append({"slice_name": name, **descriptor})
    outputs["evaluation_slices/slice_manifest.json"] = write_json(
        slice_manifest, slices_dir / "slice_manifest.json", overwrite=True
    )
    return outputs


def _known_limitations(text: Any, image: Any, subset_users: int | None) -> list[str]:
    """Assemble the manifest's honest-caveats list from what actually happened."""
    limitations = [
        "PixelRec50K records one undifferentiated implicit event type. It is mapped to "
        "`interaction`; no click/view/purchase distinction exists in the source.",
        "No user metadata exists: users are derived from the interaction log and have no "
        "attributes, not even a creation date.",
        "Items have no publication date, price, brand, or rating. Those fields are absent, "
        "not null-filled.",
        "Item engagement counters (view_number, thumbup_number, ...) are platform-wide "
        "lifetime totals with no timestamp. They are preserved in source_metadata and "
        "deliberately excluded from the feature path: they cannot be point-in-time bounded "
        "and would leak future popularity.",
    ]
    if not text.available:
        limitations.append(
            "Text feature vectors were not present at build time; text coverage is 0.0. "
            "The published file is ~8.65 GiB of JSON covering all 408,374 full-PixelRec items."
        )
    if not image.available:
        limitations.append(
            "Image feature vectors were not present at build time; image coverage is 0.0. "
            "The published file is ~8.60 GiB of JSON covering all 408,374 full-PixelRec items."
        )
    if subset_users is not None:
        limitations.append(
            f"DEVELOPMENT SUBSET: only {subset_users} users were processed. Counts and any "
            "statistic derived from them are not comparable to a full run."
        )
    return limitations


def _render_filtering_markdown(report: dict[str, Any]) -> str:
    """Render the filtering report."""
    lines = [
        "# Filtering report - pixelrec50k",
        "",
        f"Enabled: **{report['enabled']}** · converged: **{report['converged']}** · "
        f"iterations: **{report['iteration_count']}**",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "|---|---:|",
    ]
    for key, value in report["configuration"].items():
        lines.append(f"| `{key}` | {value} |")

    before = report.get("before", {})
    if before:
        lines += [
            "",
            "## Population before filtering",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Users | {before.get('total_users', 0):,} |",
            f"| Items | {before.get('total_items', 0):,} |",
            f"| Interactions | {before.get('total_interactions', 0):,} |",
            f"| Singleton items | {before.get('singleton_items', 0):,} |",
            f"| Items below item threshold | {before.get('items_below_item_threshold', 0):,} |",
            f"| Users below user threshold | {before.get('users_below_user_threshold', 0):,} |",
        ]

    lines += [
        "",
        "## Iterations",
        "",
        "| # | Users removed | Items removed | Interactions removed "
        "| Users left | Items left | Interactions left |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for iteration in report["iterations"]:
        lines.append(
            f"| {iteration['iteration']} | {iteration['users_removed']:,} | "
            f"{iteration['items_removed']:,} | {iteration['interactions_removed']:,} | "
            f"{iteration['users_remaining']:,} | {iteration['items_remaining']:,} | "
            f"{iteration['interactions_remaining']:,} |"
        )
    after = report["after"]
    lines += [
        "",
        "## After filtering",
        "",
        f"Users **{after['users']:,}** · items **{after['items']:,}** · "
        f"interactions **{after['interactions']:,}**",
    ]
    return "\n".join(lines) + "\n"


def _render_leakage_markdown(report: dict[str, Any]) -> str:
    """Render the leakage report."""
    lines = [
        "# Leakage report - pixelrec50k",
        "",
        f"**{'PASSED' if report['passed'] else 'FAILED'}** — "
        f"{report['passed_checks']}/{report['total_checks']} checks passed · "
        f"{report['critical_failures']} critical failures · {report['warnings']} warnings",
        "",
        "A critical failure aborts the pipeline: leakage makes offline metrics *better*, "
        "so it cannot be caught by looking at results.",
        "",
        "| Check | Severity | Result | Detail |",
        "|---|---|---|---|",
    ]
    for check in report["checks"]:
        verdict = (
            "pass"
            if check["passed"]
            else ("**FAIL**" if check["severity"] == "critical" else "warn")
        )
        lines.append(
            f"| `{check['check_id']}` | {check['severity']} | {verdict} | {check['detail']} |"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "COLLABORATIVE_COLUMNS",
    "GRAPH_COLUMNS",
    "ITEM_METADATA_COLUMNS",
    "SPLIT_VERSION",
    "PipelineOptions",
    "PipelineResult",
    "run_pipeline",
]
