"""Data layer: contracts, validation, identifier mapping, splitting.

Import order within this package is strictly ``schemas -> {validation,
id_mapping, loaders} -> {preprocessing, splitting}``. Nothing here imports from
``models``, ``retrieval``, ``ranking``, or ``api``.
"""

from __future__ import annotations

from omnirank.data.id_mapping import UNKNOWN_INDEX, IdMapping
from omnirank.data.loaders import DatasetBundle, DatasetLoader
from omnirank.data.preprocessing import PreprocessedDataset, Preprocessor
from omnirank.data.schemas import EventType, Interaction, Item, User
from omnirank.data.splitting import DataSplit, SplitBoundaries, Splitter, check_split_integrity
from omnirank.data.validation import (
    ValidatedBatch,
    ValidationReport,
    ValidationRule,
    validate_batch,
)

__all__ = [
    "UNKNOWN_INDEX",
    "DataSplit",
    "DatasetBundle",
    "DatasetLoader",
    "EventType",
    "IdMapping",
    "Interaction",
    "Item",
    "PreprocessedDataset",
    "Preprocessor",
    "SplitBoundaries",
    "Splitter",
    "User",
    "ValidatedBatch",
    "ValidationReport",
    "ValidationRule",
    "check_split_integrity",
    "validate_batch",
]
