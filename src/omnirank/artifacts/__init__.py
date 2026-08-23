"""Artifact management: metadata contract and the filesystem registry."""

from __future__ import annotations

from omnirank.artifacts.metadata import (
    ArtifactMetadata,
    ArtifactType,
    SupportedDevice,
    build_metadata,
)
from omnirank.artifacts.registry import ArtifactRegistry

__all__ = [
    "ArtifactMetadata",
    "ArtifactRegistry",
    "ArtifactType",
    "SupportedDevice",
    "build_metadata",
]
