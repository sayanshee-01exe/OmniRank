"""Dataset manifest and versioning.

The manifest is the contract between a processed dataset and everyone who later
uses it. Its purpose is narrow and testable: **another developer, given this
file and the repository, can determine whether their rebuild is the same
dataset.**

That requires three things, all recorded here: what went in (source files with
checksums), how it was processed (config hash, seed, pipeline version, git
commit, split and mapping versions), and what came out (output files with
checksums and row counts).

``known_limitations`` is a required field, not an optional courtesy. A dataset
whose gaps are undocumented gets used as though it had none.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from omnirank.core.logging import get_logger

logger = get_logger(__name__)

#: Bumped when the pipeline changes in a way that alters its outputs for
#: identical inputs. Recorded in every manifest so two datasets built by
#: different pipeline versions are never mistaken for each other.
PIPELINE_VERSION: Final = "2.0.0"

#: Bumped when the processed table schemas change.
SCHEMA_VERSION: Final = "2"

MANIFEST_FILENAME: Final = "dataset_manifest.json"


def detect_git_commit(repo_root: Path | str | None = None) -> str | None:
    """Return the current commit SHA, or ``None`` outside a git checkout.

    Never raises. A dataset built from a source tree with no git history is
    legitimate; it simply has no commit to record, and recording ``null`` is
    more honest than recording a placeholder.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


@dataclass(slots=True)
class DatasetManifest:
    """Everything needed to identify and reproduce a processed dataset."""

    dataset_name: str
    dataset_version: str
    source_repository: str
    source_files: dict[str, Any]
    processing_timestamp: str
    configuration_hash: str
    random_seed: int
    mapping_version: str
    split_version: str
    split_strategy: str
    ordering_field: str
    filtering_configuration: dict[str, Any]
    raw_row_counts: dict[str, int]
    processed_row_counts: dict[str, int]
    user_counts: dict[str, int]
    item_counts: dict[str, int]
    interaction_counts: dict[str, int]
    feature_dimensions: dict[str, int | None]
    feature_coverage: dict[str, float]
    output_files: dict[str, Any]
    known_limitations: list[str]
    licence: str
    pipeline_version: str = PIPELINE_VERSION
    schema_version: str = SCHEMA_VERSION
    git_commit: str | None = None
    python_version: str = field(default_factory=platform.python_version)
    subset_users: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready manifest payload."""
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "source_repository": self.source_repository,
            "licence": self.licence,
            "source_files": self.source_files,
            "source_checksums": {
                name: descriptor.get("sha256") for name, descriptor in self.source_files.items()
            },
            "source_file_sizes": {
                name: descriptor.get("bytes") for name, descriptor in self.source_files.items()
            },
            "processing_timestamp": self.processing_timestamp,
            "pipeline_version": self.pipeline_version,
            "schema_version": self.schema_version,
            "git_commit": self.git_commit,
            "python_version": self.python_version,
            "configuration_hash": self.configuration_hash,
            "random_seed": self.random_seed,
            "mapping_version": self.mapping_version,
            "split_version": self.split_version,
            "split_strategy": self.split_strategy,
            "ordering_field": self.ordering_field,
            "filtering_configuration": self.filtering_configuration,
            "subset_users": self.subset_users,
            "raw_row_counts": self.raw_row_counts,
            "processed_row_counts": self.processed_row_counts,
            "user_counts": self.user_counts,
            "item_counts": self.item_counts,
            "interaction_counts": self.interaction_counts,
            "feature_dimensions": self.feature_dimensions,
            "feature_coverage": self.feature_coverage,
            "output_files": self.output_files,
            "output_checksums": {
                name: descriptor.get("sha256")
                for name, descriptor in self.output_files.items()
                if isinstance(descriptor, dict)
            },
            "known_limitations": self.known_limitations,
        }


def build_manifest(
    *,
    dataset_name: str,
    dataset_version: str,
    source_repository: str,
    licence: str,
    source_files: dict[str, Any],
    configuration_hash: str,
    random_seed: int,
    mapping_version: str,
    split_version: str,
    split_strategy: str,
    ordering_field: str,
    filtering_configuration: dict[str, Any],
    raw_row_counts: dict[str, int],
    processed_row_counts: dict[str, int],
    user_counts: dict[str, int],
    item_counts: dict[str, int],
    interaction_counts: dict[str, int],
    feature_dimensions: dict[str, int | None],
    feature_coverage: dict[str, float],
    output_files: dict[str, Any],
    known_limitations: list[str],
    repo_root: Path | str | None = None,
    subset_users: int | None = None,
) -> DatasetManifest:
    """Assemble a manifest, detecting environment fields automatically."""
    manifest = DatasetManifest(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        source_repository=source_repository,
        licence=licence,
        source_files=source_files,
        processing_timestamp=datetime.now(tz=UTC).isoformat(),
        configuration_hash=configuration_hash,
        random_seed=random_seed,
        mapping_version=mapping_version,
        split_version=split_version,
        split_strategy=split_strategy,
        ordering_field=ordering_field,
        filtering_configuration=filtering_configuration,
        raw_row_counts=raw_row_counts,
        processed_row_counts=processed_row_counts,
        user_counts=user_counts,
        item_counts=item_counts,
        interaction_counts=interaction_counts,
        feature_dimensions=feature_dimensions,
        feature_coverage=feature_coverage,
        output_files=output_files,
        known_limitations=known_limitations,
        git_commit=detect_git_commit(repo_root),
        subset_users=subset_users,
    )
    logger.info(
        "manifest.built",
        dataset=f"{dataset_name}@{dataset_version}",
        outputs=len(output_files),
        git_commit=manifest.git_commit,
    )
    return manifest


__all__ = [
    "MANIFEST_FILENAME",
    "PIPELINE_VERSION",
    "SCHEMA_VERSION",
    "DatasetManifest",
    "build_manifest",
    "detect_git_commit",
]
