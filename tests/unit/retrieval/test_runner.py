"""The Phase 4 fit drivers.

Thin by design -- the point of this module is that LightGCN and SASRec are
*fitted* differently and *evaluated* identically, through the Phase 3
``run_experiment``. What is worth testing here is the loading it does on the
way: sequential examples live outside ``ProcessedDataset`` because only SASRec
reads them, so a missing or renamed file has to fail with an actionable message
rather than a bare ``FileNotFoundError`` from inside pandas.
"""

from __future__ import annotations

import pandas as pd
import pytest

from omnirank.core.exceptions import DataSourceError
from omnirank.retrieval.runner import (
    HYBRID,
    LIGHTGCN,
    SASREC,
    SEQUENCE_SUBDIR,
    load_sequences,
)


@pytest.fixture
def sequence_root(tmp_path):
    """A processed root holding two split sequence files."""
    directory = tmp_path / SEQUENCE_SUBDIR
    directory.mkdir()
    for split, rows in (("train", 3), ("validation", 2)):
        pd.DataFrame(
            {
                "internal_user_id": range(rows),
                "item_sequence": [[1, 2] for _ in range(rows)],
                "target_item": [3] * rows,
                "split": [split] * rows,
            }
        ).to_parquet(directory / f"{split}_sequences.parquet")
    return tmp_path


class TestModelNames:
    def test_names_match_the_registered_identifiers(self) -> None:
        """These strings key config blocks, artifact paths and selection records."""
        assert (LIGHTGCN, SASREC, HYBRID) == (
            "lightgcn",
            "sasrec",
            "popularity_bpr_hybrid",
        )


class TestLoadSequences:
    def test_loads_one_split(self, sequence_root) -> None:
        assert len(load_sequences(sequence_root, ("train",))) == 3

    def test_concatenates_several_splits(self, sequence_root) -> None:
        """The final stage fits on train+validation, so both files are read."""
        combined = load_sequences(sequence_root, ("train", "validation"))
        assert len(combined) == 5
        assert set(combined["split"]) == {"train", "validation"}

    def test_index_is_reset_after_concatenation(self, sequence_root) -> None:
        """Duplicate index labels would misalign every positional lookup later."""
        combined = load_sequences(sequence_root, ("train", "validation"))
        assert combined.index.tolist() == list(range(5))

    def test_missing_file_names_the_path_it_wanted(self, tmp_path) -> None:
        with pytest.raises(DataSourceError) as caught:
            load_sequences(tmp_path, ("train",))
        assert "train_sequences.parquet" in str(caught.value)

    def test_a_present_split_does_not_excuse_a_missing_one(self, sequence_root) -> None:
        """Silently returning the splits that happened to exist would train on less
        data than asked for, and nothing downstream would notice."""
        with pytest.raises(DataSourceError):
            load_sequences(sequence_root, ("train", "test"))
