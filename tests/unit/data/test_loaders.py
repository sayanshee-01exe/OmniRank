"""PixelRec50K raw loaders: file checks, schema assertions, chunking, subsetting."""

from __future__ import annotations

import pytest

from omnirank.core.exceptions import DataSourceError
from omnirank.data.pixelrec.loaders import (
    INTERACTION_COLUMNS,
    ITEM_INFO_COLUMNS,
    PixelRec50KLoader,
)
from tests.fixtures.pixelrec import FixtureSpec, write_fixture


@pytest.fixture
def loader(pixelrec_fixture_dir):
    return PixelRec50KLoader(pixelrec_fixture_dir, chunk_size=7, compute_checksums=False)


class TestFilePresence:
    def test_accepts_a_complete_directory(self, loader):
        loader.check_files_present()

    def test_missing_directory_names_the_download_script(self, tmp_path):
        with pytest.raises(DataSourceError) as exc:
            PixelRec50KLoader(tmp_path / "absent").check_files_present()
        assert "download_pixelrec50k.py" in str(exc.value)

    def test_missing_file_is_named(self, pixelrec_fixture_dir):
        (pixelrec_fixture_dir / "item_info.csv").unlink()
        with pytest.raises(DataSourceError) as exc:
            PixelRec50KLoader(pixelrec_fixture_dir).check_files_present()
        assert "item_info.csv" in str(exc.value)


class TestSchemaValidation:
    def test_wrong_header_is_rejected(self, pixelrec_fixture_dir):
        """A silently renamed upstream column must fail at load, not at training."""
        path = pixelrec_fixture_dir / "interaction.csv"
        lines = path.read_text().splitlines()
        lines[0] = "item,user,ts"
        path.write_text("\n".join(lines))
        with pytest.raises(DataSourceError) as exc:
            PixelRec50KLoader(pixelrec_fixture_dir).load_interactions()
        assert "header does not match" in str(exc.value)

    def test_error_names_expected_and_found_columns(self, pixelrec_fixture_dir):
        path = pixelrec_fixture_dir / "interaction.csv"
        lines = path.read_text().splitlines()
        lines[0] = "item,user,ts"
        path.write_text("\n".join(lines))
        with pytest.raises(DataSourceError) as exc:
            PixelRec50KLoader(pixelrec_fixture_dir).load_interactions()
        assert "item_id" in str(exc.value) and "ts" in str(exc.value)

    def test_a_non_csv_file_is_rejected_clearly(self, pixelrec_fixture_dir):
        (pixelrec_fixture_dir / "interaction.csv").write_bytes(b"\x00\x01\x02binary")
        with pytest.raises(DataSourceError):
            PixelRec50KLoader(pixelrec_fixture_dir).load_interactions()

    def test_expected_headers_are_the_official_ones(self):
        assert INTERACTION_COLUMNS == ("item_id", "user_id", "timestamp")
        assert ITEM_INFO_COLUMNS[0] == "item_id"
        assert ITEM_INFO_COLUMNS[-3:] == ("title", "tag", "description")


class TestLoading:
    def test_loads_interactions(self, loader):
        frame = loader.load_interactions()
        assert set(frame.columns) == {*INTERACTION_COLUMNS, "source_row_id"}
        assert len(frame) > 0

    def test_identifiers_stay_strings(self, loader):
        """Numeric coercion of an id column would silently merge distinct items."""
        frame = loader.load_interactions()
        assert frame["item_id"].dtype == "string"
        assert frame["user_id"].dtype == "string"

    def test_source_row_ids_are_unique_and_contiguous(self, loader):
        frame = loader.load_interactions()
        assert frame["source_row_id"].tolist() == list(range(len(frame)))

    def test_chunk_size_does_not_change_the_result(self, pixelrec_fixture_dir):
        small = PixelRec50KLoader(pixelrec_fixture_dir, chunk_size=3, compute_checksums=False)
        large = PixelRec50KLoader(pixelrec_fixture_dir, chunk_size=100_000, compute_checksums=False)
        assert small.load_interactions().equals(large.load_interactions())

    def test_loads_items(self, loader):
        frame = loader.load_items()
        assert list(frame.columns) == list(ITEM_INFO_COLUMNS)

    def test_load_bundles_both_tables_with_provenance(self, pixelrec_fixture_dir):
        raw = PixelRec50KLoader(pixelrec_fixture_dir, compute_checksums=True).load()
        assert len(raw.interactions) > 0
        assert len(raw.items) > 0
        assert set(raw.provenance) == {"interaction.csv", "item_info.csv"}
        assert all(entry["sha256"] for entry in raw.provenance.values())

    def test_checksums_can_be_skipped(self, pixelrec_fixture_dir):
        raw = PixelRec50KLoader(pixelrec_fixture_dir, compute_checksums=False).load()
        assert raw.provenance["interaction.csv"]["sha256"] == ""


class TestSubset:
    def test_keeps_whole_user_histories(self, pixelrec_fixture_dir):
        """Truncating a user mid-history would corrupt splitting and sequences."""
        full = PixelRec50KLoader(pixelrec_fixture_dir, compute_checksums=False).load_interactions()
        subset = PixelRec50KLoader(
            pixelrec_fixture_dir, compute_checksums=False, subset_users=3
        ).load_interactions()
        kept = set(subset["user_id"])
        assert len(kept) == 3
        for user in kept:
            assert (subset["user_id"] == user).sum() == (full["user_id"] == user).sum()

    def test_selection_is_deterministic(self, pixelrec_fixture_dir):
        def load():
            return set(
                PixelRec50KLoader(
                    pixelrec_fixture_dir, compute_checksums=False, subset_users=4
                ).load_interactions()["user_id"]
            )

        assert load() == load()

    def test_subset_larger_than_the_dataset_keeps_everything(self, pixelrec_fixture_dir):
        full = PixelRec50KLoader(pixelrec_fixture_dir, compute_checksums=False).load_interactions()
        subset = PixelRec50KLoader(
            pixelrec_fixture_dir, compute_checksums=False, subset_users=10_000
        ).load_interactions()
        assert len(subset) == len(full)


class TestFixtureShape:
    """The fixture must mimic the real file, or the tests above prove nothing."""

    def test_fixture_contains_the_edge_cases_it_claims(self, tmp_path):
        spec = write_fixture(tmp_path / "raw", FixtureSpec())
        loader = PixelRec50KLoader(tmp_path / "raw", compute_checksums=False)
        interactions = loader.load_interactions()
        items = loader.load_items()

        assert interactions.duplicated(subset=["item_id", "user_id", "timestamp"]).sum() == (
            spec.duplicate_rows
        )
        dangling = ~interactions["item_id"].isin(set(items["item_id"]))
        assert dangling.sum() == spec.dangling_interactions
        assert items["title"].isna().sum() == spec.items_without_title
