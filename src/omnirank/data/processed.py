"""Read back a Phase 2 processed dataset, with its identity verified.

Phase 3 consumes Phase 2's outputs and must not trust them blindly. Before any
model sees a row, :func:`load_processed_dataset` checks that the manifest exists,
that its schema/split/mapping versions are ones this code understands, that every
required file is present, and — optionally — that the files still hash to what
the manifest recorded.

Why the checks are worth their cost: a model silently trained against a
regenerated split, or against mappings from a different dataset version, produces
metrics that look fine and mean nothing. The failure is invisible downstream, so
it has to be caught here.

**Timestamps.** The processed split tables deliberately carry only
``interaction_order`` (see ``docs/data/processed_schemas.md``). Time-decayed
popularity needs real event ages, so timestamps are read from
``interim/canonical_interactions.parquet`` — itself a manifest-listed,
checksummed Phase 2 output — and joined on. Nothing is regenerated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pandas as pd

from omnirank.core.exceptions import DataError, DataSourceError
from omnirank.core.logging import get_logger
from omnirank.data.io import sha256_file

logger = get_logger(__name__)

#: Phase 2 schema/split/mapping versions this code was written against. A
#: mismatch is a hard failure rather than a warning: the column meanings, the
#: split protocol, and the id space are all encoded in these numbers.
SUPPORTED_SCHEMA_VERSIONS: Final = frozenset({"2"})
SUPPORTED_SPLIT_VERSIONS: Final = frozenset({"1"})
SUPPORTED_MAPPING_VERSIONS: Final = frozenset({"1"})

MANIFEST_FILENAME: Final = "dataset_manifest.json"

TRAIN: Final = "train"
VALIDATION: Final = "validation"
TEST: Final = "test"

#: Files a Phase 3 run cannot proceed without, relative to the processed root.
REQUIRED_PROCESSED_FILES: Final = (
    "train_interactions.parquet",
    "validation_interactions.parquet",
    "test_interactions.parquet",
    "split_metadata.json",
    "features/item_training_popularity.parquet",
    "metadata/item_metadata.parquet",
)

REQUIRED_MAPPING_FILES: Final = (
    "user_id_mapping.parquet",
    "item_id_mapping.parquet",
    "mapping_metadata.json",
)


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    """The manifest fields that identify one processed dataset build.

    Recorded verbatim in every model artifact and every evaluation report, so a
    metric can always be traced back to the exact data that produced it.
    """

    dataset_name: str
    dataset_version: str
    schema_version: str
    split_version: str
    mapping_version: str
    split_strategy: str
    ordering_field: str
    pipeline_version: str
    configuration_hash: str
    manifest_sha256: str
    data_git_commit: str | None = None
    subset_users: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Artifact- and report-ready description."""
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "schema_version": self.schema_version,
            "split_version": self.split_version,
            "mapping_version": self.mapping_version,
            "split_strategy": self.split_strategy,
            "ordering_field": self.ordering_field,
            "pipeline_version": self.pipeline_version,
            "dataset_configuration_hash": self.configuration_hash,
            "dataset_manifest_sha256": self.manifest_sha256,
            "data_git_commit": self.data_git_commit,
            "subset_users": self.subset_users,
        }

    @property
    def label(self) -> str:
        """Short human-readable identity, e.g. ``pixelrec50k@v1/split1``."""
        return f"{self.dataset_name}@{self.dataset_version}/split{self.split_version}"


@dataclass(slots=True)
class ProcessedDataset:
    """A verified Phase 2 dataset, ready for fitting and evaluation."""

    identity: DatasetIdentity
    root: Path
    #: Split interaction tables, keyed by split name. Internal ids plus
    #: ``interaction_order``, ``timestamp``, ``event_type``, ``interaction_weight``.
    splits: dict[str, pd.DataFrame]
    #: external_user_id <-> internal_user_id
    user_mapping: pd.DataFrame
    #: external_item_id <-> internal_item_id
    item_mapping: pd.DataFrame
    #: Training-only popularity, as written by Phase 2.
    item_popularity: pd.DataFrame
    #: internal_item_id -> category, for category-diversity reporting.
    item_categories: pd.DataFrame
    mapping_metadata: dict[str, Any] = field(default_factory=dict)
    checksums_verified: bool = False

    # -- convenience ------------------------------------------------------- #
    @property
    def num_users(self) -> int:
        """Users in the mapping."""
        return len(self.user_mapping)

    @property
    def num_items(self) -> int:
        """Items in the mapping."""
        return len(self.item_mapping)

    def split(self, name: str) -> pd.DataFrame:
        """One split's interactions.

        Raises:
            DataError: Unknown split name.
        """
        if name not in self.splits:
            raise DataError("Unknown split", requested=name, available=sorted(self.splits))
        return self.splits[name]

    def fit_interactions(self, splits: tuple[str, ...]) -> pd.DataFrame:
        """Concatenate the named splits into one fit table.

        The fit boundary is the single most important thing to get right in this
        phase: it defines both what the model learns from and which items count
        as "already seen" at evaluation time. Passing it explicitly - rather than
        defaulting it - is why it is a required argument everywhere.
        """
        frames = [self.split(name) for name in splits]
        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            "dataset.fit_interactions",
            splits=list(splits),
            rows=len(combined),
            users=int(combined["internal_user_id"].nunique()),
            items=int(combined["internal_item_id"].nunique()),
        )
        return combined

    def internal_to_external_items(self) -> dict[int, str]:
        """Reverse item lookup, for converting model output to public ids."""
        return dict(
            zip(
                self.item_mapping["internal_item_id"].astype(int),
                self.item_mapping["external_item_id"].astype(str),
                strict=True,
            )
        )

    def external_to_internal_users(self) -> dict[str, int]:
        """Forward user lookup, for accepting public ids at the interface."""
        return dict(
            zip(
                self.user_mapping["external_user_id"].astype(str),
                self.user_mapping["internal_user_id"].astype(int),
                strict=True,
            )
        )


def _read_manifest(processed_root: Path) -> tuple[dict[str, Any], str]:
    """Read the dataset manifest and its own checksum."""
    path = processed_root / MANIFEST_FILENAME
    if not path.is_file():
        raise DataSourceError(
            "Processed dataset manifest not found. Run the Phase 2 pipeline first: "
            "`python scripts/prepare_data.py --config configs/data/pixelrec50k.yaml`.",
            expected=str(path),
        )
    try:
        payload: dict[str, Any] = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise DataSourceError(
            "Dataset manifest is not valid JSON", path=str(path), reason=str(exc)[:200]
        ) from exc
    return payload, sha256_file(path)


def _check_versions(manifest: dict[str, Any]) -> None:
    """Reject a dataset built by a Phase 2 version this code does not understand.

    Raises:
        DataError: Any of the three versions is unsupported, naming both the
            found and the supported values.
    """
    checks = (
        ("schema_version", SUPPORTED_SCHEMA_VERSIONS),
        ("split_version", SUPPORTED_SPLIT_VERSIONS),
        ("mapping_version", SUPPORTED_MAPPING_VERSIONS),
    )
    problems = [
        f"{field_name}={manifest.get(field_name)!r} (supported: {sorted(supported)})"
        for field_name, supported in checks
        if str(manifest.get(field_name)) not in supported
    ]
    if problems:
        raise DataError(
            "Processed dataset was built by an incompatible Phase 2 version. "
            "Rebuild it, or update the supported versions in "
            "omnirank.data.processed after reviewing what changed.",
            problems=problems,
        )


def _check_required_files(processed_root: Path, mappings_root: Path) -> None:
    """Verify every file Phase 3 needs is actually on disk."""
    missing = [
        str(processed_root / name)
        for name in REQUIRED_PROCESSED_FILES
        if not (processed_root / name).is_file()
    ]
    missing += [
        str(mappings_root / name)
        for name in REQUIRED_MAPPING_FILES
        if not (mappings_root / name).is_file()
    ]
    if missing:
        raise DataSourceError(
            "Processed dataset is incomplete. Re-run the Phase 2 pipeline.",
            missing=missing,
        )


def _verify_checksums(
    manifest: dict[str, Any], project_root: Path, *, limit_to: tuple[str, ...] | None = None
) -> list[str]:
    """Re-hash manifest outputs and return the names that no longer match.

    Only files still present are checked; a manifest entry for a file that has
    been deleted is reported by the required-file check instead, with a clearer
    message than "checksum missing".
    """
    mismatched: list[str] = []
    for name, descriptor in manifest.get("output_files", {}).items():
        if limit_to is not None and not name.endswith(limit_to):
            continue
        if not isinstance(descriptor, dict):
            continue
        recorded = descriptor.get("sha256")
        path_value = descriptor.get("path")
        if not recorded or not path_value:
            continue
        path = Path(path_value)
        if not path.is_absolute():
            path = project_root / path
        if not path.is_file():
            continue
        if sha256_file(path) != recorded:
            mismatched.append(name)
    return mismatched


def _attach_timestamps(splits: dict[str, pd.DataFrame], interim_path: Path) -> None:
    """Join real event timestamps onto the split tables, in place.

    The processed split schema carries ``interaction_order`` but not
    ``timestamp``. Time-decayed popularity needs genuine ages, so they come from
    the interim canonical table - a checksummed Phase 2 output - rather than
    being approximated from the order, which would be a fabricated timestamp.

    Raises:
        DataSourceError: The interim table is missing.
        DataError: A split row has no matching timestamp, which would mean the
            interim table and the split tables came from different builds.
    """
    if not interim_path.is_file():
        raise DataSourceError(
            "Interim canonical interactions not found. Time-decayed popularity "
            "needs real event timestamps, which the processed split tables do "
            "not carry. Re-run the Phase 2 pipeline to regenerate it.",
            expected=str(interim_path),
        )
    canonical = pd.read_parquet(
        interim_path,
        columns=["internal_user_id", "internal_item_id", "interaction_order", "timestamp"],
    )
    keys = ["internal_user_id", "internal_item_id", "interaction_order"]
    for name, frame in splits.items():
        merged = frame.merge(canonical, on=keys, how="left", validate="one_to_one")
        missing = int(merged["timestamp"].isna().sum())
        if missing:
            raise DataError(
                "Split rows have no matching timestamp in the interim canonical "
                "table. The processed splits and the interim table are from "
                "different Phase 2 builds.",
                split=name,
                missing_rows=missing,
            )
        merged["timestamp"] = merged["timestamp"].astype("int64")
        splits[name] = merged


def load_processed_dataset(
    processed_root: Path | str,
    mappings_root: Path | str,
    *,
    project_root: Path | str = ".",
    verify_checksums: bool = True,
    with_timestamps: bool = True,
) -> ProcessedDataset:
    """Load and verify a Phase 2 processed dataset.

    Args:
        processed_root: ``data/processed/<dataset>``.
        mappings_root: ``artifacts/mappings/<dataset>``.
        project_root: Root that manifest-relative paths resolve against.
        verify_checksums: Re-hash the manifest's outputs. Costs one full read of
            the processed tables (~40 MB here); worth it before a training run,
            skippable in tight loops.
        with_timestamps: Join real timestamps from the interim canonical table.

    Returns:
        A verified :class:`ProcessedDataset`.

    Raises:
        DataSourceError: The manifest or a required file is missing.
        DataError: Versions are unsupported, or checksums do not match.
    """
    processed = Path(processed_root)
    mappings = Path(mappings_root)
    root = Path(project_root)

    manifest, manifest_sha = _read_manifest(processed)
    _check_versions(manifest)
    _check_required_files(processed, mappings)

    verified = False
    if verify_checksums:
        mismatched = _verify_checksums(manifest, root)
        if mismatched:
            raise DataError(
                "Processed files no longer match the checksums recorded in the "
                "dataset manifest. The data was modified after it was built; "
                "re-run the Phase 2 pipeline rather than training on it.",
                mismatched=mismatched[:10],
                mismatched_count=len(mismatched),
            )
        verified = True

    identity = DatasetIdentity(
        dataset_name=manifest["dataset_name"],
        dataset_version=manifest["dataset_version"],
        schema_version=str(manifest["schema_version"]),
        split_version=str(manifest["split_version"]),
        mapping_version=str(manifest["mapping_version"]),
        split_strategy=manifest["split_strategy"],
        ordering_field=manifest["ordering_field"],
        pipeline_version=manifest["pipeline_version"],
        configuration_hash=manifest["configuration_hash"],
        manifest_sha256=manifest_sha,
        data_git_commit=manifest.get("git_commit"),
        subset_users=manifest.get("subset_users"),
    )

    splits = {
        name: pd.read_parquet(processed / f"{name}_interactions.parquet")
        for name in (TRAIN, VALIDATION, TEST)
    }
    if with_timestamps:
        _attach_timestamps(
            splits,
            root / "data" / "interim" / identity.dataset_name / "canonical_interactions.parquet",
        )

    item_metadata = pd.read_parquet(
        processed / "metadata" / "item_metadata.parquet",
        columns=["internal_item_id", "category"],
    )

    dataset = ProcessedDataset(
        identity=identity,
        root=processed,
        splits=splits,
        user_mapping=pd.read_parquet(mappings / "user_id_mapping.parquet"),
        item_mapping=pd.read_parquet(mappings / "item_id_mapping.parquet"),
        item_popularity=pd.read_parquet(
            processed / "features" / "item_training_popularity.parquet"
        ),
        item_categories=item_metadata,
        mapping_metadata=json.loads((mappings / "mapping_metadata.json").read_text()),
        checksums_verified=verified,
    )
    logger.info(
        "dataset.loaded",
        identity=identity.label,
        users=dataset.num_users,
        items=dataset.num_items,
        rows={name: len(frame) for name, frame in splits.items()},
        checksums_verified=verified,
    )
    return dataset


def load_evaluation_slice(processed_root: Path | str, slice_name: str) -> set[int]:
    """Read one Phase 2 evaluation slice as a set of internal ids.

    Returns an empty set for a slice file that exists but is empty - which is a
    real state (``users_cold_start`` is empty by construction under leave-last-N)
    and must not be confused with a missing slice.

    Raises:
        DataSourceError: The slice file does not exist.
    """
    path = Path(processed_root) / "evaluation_slices" / f"{slice_name}.parquet"
    if not path.is_file():
        raise DataSourceError("Evaluation slice not found", slice_name=slice_name, path=str(path))
    frame = pd.read_parquet(path)
    if frame.empty:
        return set()
    return set(frame["entity_id"].astype(int))


__all__ = [
    "MANIFEST_FILENAME",
    "REQUIRED_MAPPING_FILES",
    "REQUIRED_PROCESSED_FILES",
    "SUPPORTED_MAPPING_VERSIONS",
    "SUPPORTED_SCHEMA_VERSIONS",
    "SUPPORTED_SPLIT_VERSIONS",
    "TEST",
    "TRAIN",
    "VALIDATION",
    "DatasetIdentity",
    "ProcessedDataset",
    "load_evaluation_slice",
    "load_processed_dataset",
]
