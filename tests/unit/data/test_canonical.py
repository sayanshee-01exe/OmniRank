"""Source-to-canonical mapping, including the fields that must never be invented."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from omnirank.data.pixelrec.canonical import (
    DEFAULT_EVENT_TYPE,
    canonicalize_interactions,
    canonicalize_items,
    derive_users,
)
from omnirank.data.pixelrec.loaders import PixelRec50KLoader
from omnirank.data.schemas import EventType, Interaction, Item, User


@pytest.fixture
def raw(pixelrec_fixture_dir):
    loader = PixelRec50KLoader(pixelrec_fixture_dir, compute_checksums=False)
    return loader.load_interactions(), loader.load_items()


class TestInteractions:
    def test_event_type_is_the_generic_one(self, raw):
        """PixelRec measured engagement, not clicks or purchases."""
        frame = canonicalize_interactions(raw[0])
        assert set(frame["event_type"]) == {DEFAULT_EVENT_TYPE}
        assert DEFAULT_EVENT_TYPE == "interaction"

    def test_generic_event_type_exists_in_the_canonical_vocabulary(self):
        assert EventType(DEFAULT_EVENT_TYPE) is EventType.INTERACTION

    def test_weight_is_uniform(self, raw):
        frame = canonicalize_interactions(raw[0])
        assert set(frame["interaction_weight"]) == {1.0}

    def test_interaction_ids_are_derived_and_unique(self, raw):
        frame = canonicalize_interactions(raw[0])
        assert frame["interaction_id"].is_unique
        assert frame["interaction_id"].str.startswith("pr50k-").all()

    def test_interaction_ids_are_reproducible(self, raw):
        assert canonicalize_interactions(raw[0])["interaction_id"].tolist() == (
            canonicalize_interactions(raw[0])["interaction_id"].tolist()
        )

    def test_epoch_seconds_become_utc_datetimes(self, raw):
        frame = canonicalize_interactions(raw[0])
        assert str(frame["event_timestamp_utc"].dt.tz) == "UTC"

    def test_integer_timestamp_is_preserved_as_the_ordering_key(self, raw):
        """Ordering must not depend on datetime parsing being correct."""
        frame = canonicalize_interactions(raw[0])
        assert frame["timestamp"].dtype == "Int64"
        # The datetime column must be exactly the integer column reinterpreted,
        # so the two can never drift apart.
        expected = pd.to_datetime(frame["timestamp"].astype("int64"), unit="s", utc=True)
        assert (frame["event_timestamp_utc"] == expected).all()

    def test_identifier_whitespace_is_stripped(self):
        """A padded id would otherwise become a second, distinct user."""
        raw_frame = pd.DataFrame(
            {
                "item_id": [" i1 "],
                "user_id": ["  u1"],
                "timestamp": [1_640_995_200],
                "source_row_id": [0],
            }
        )
        frame = canonicalize_interactions(raw_frame)
        assert frame.loc[0, "external_user_id"] == "u1"
        assert frame.loc[0, "external_item_id"] == "i1"

    def test_empty_input_yields_a_typed_empty_frame(self):
        frame = canonicalize_interactions(pd.DataFrame())
        assert frame.empty
        assert "external_user_id" in frame.columns

    def test_output_satisfies_the_pydantic_contract(self, raw):
        """The frames are the fast path; the records remain the schema authority."""
        frame = canonicalize_interactions(raw[0])
        for row in frame.head(20).to_dict(orient="records"):
            Interaction(
                interaction_id=row["interaction_id"],
                user_id=row["external_user_id"],
                item_id=row["external_item_id"],
                event_type=row["event_type"],
                timestamp=row["event_timestamp_utc"],
                weight=row["interaction_weight"],
            )


class TestItems:
    def test_tag_becomes_category(self, raw):
        frame = canonicalize_items(raw[1])
        assert frame["category"].notna().any()

    def test_absent_ecommerce_fields_are_not_invented(self, raw):
        """price, brand, rating and inventory do not exist in PixelRec."""
        frame = canonicalize_items(raw[1])
        for forbidden in ("price", "brand", "rating", "inventory", "available_quantity"):
            assert forbidden not in frame.columns

    def test_no_creation_date_is_invented(self, raw):
        frame = canonicalize_items(raw[1])
        assert "created_at" not in frame.columns

    def test_missing_title_stays_missing(self, raw):
        frame = canonicalize_items(raw[1])
        assert frame["title"].isna().sum() > 0

    def test_empty_strings_normalise_to_missing(self):
        raw_frame = pd.DataFrame(
            {"item_id": ["i1"], "title": ["   "], "tag": [""], "description": ["ok"]}
        )
        frame = canonicalize_items(raw_frame)
        assert pd.isna(frame.loc[0, "title"])
        assert pd.isna(frame.loc[0, "category"])
        assert frame.loc[0, "description"] == "ok"

    def test_engagement_counters_go_to_source_metadata(self, raw):
        frame = canonicalize_items(raw[1])
        payload = json.loads(frame.loc[0, "source_metadata"])
        assert "view_number" in payload
        assert "thumbup_number" in payload

    def test_engagement_counters_are_not_top_level_columns(self, raw):
        """They are platform-wide lifetime totals with no timestamp: not features."""
        frame = canonicalize_items(raw[1])
        assert "view_number" not in frame.columns
        assert "thumbup_number" not in frame.columns

    def test_image_reference_is_an_identifier_not_a_path(self, raw):
        frame = canonicalize_items(raw[1])
        assert frame.loc[0, "image_reference"] == frame.loc[0, "external_item_id"] + ".jpg"
        assert "/" not in frame.loc[0, "image_reference"]

    def test_feature_references_are_the_item_id(self, raw):
        frame = canonicalize_items(raw[1])
        assert (frame["text_feature_reference"] == frame["external_item_id"]).all()
        assert (frame["image_feature_reference"] == frame["external_item_id"]).all()

    def test_output_satisfies_the_pydantic_contract(self, raw):
        frame = canonicalize_items(raw[1])
        for row in frame.head(20).to_dict(orient="records"):
            Item(
                item_id=row["external_item_id"],
                title=None if pd.isna(row["title"]) else row["title"],
                description=None if pd.isna(row["description"]) else row["description"],
                category=None if pd.isna(row["category"]) else row["category"],
                image_id=row["image_reference"],
            )


class TestUsers:
    def test_users_are_derived_from_interactions(self, raw):
        interactions = canonicalize_interactions(raw[0])
        users = derive_users(interactions)
        assert set(users["external_user_id"]) == set(interactions["external_user_id"])

    def test_users_carry_no_invented_attributes(self, raw):
        users = derive_users(canonicalize_interactions(raw[0]))
        assert list(users.columns) == ["external_user_id"]

    def test_users_are_sorted_and_unique(self, raw):
        users = derive_users(canonicalize_interactions(raw[0]))
        assert users["external_user_id"].is_unique
        assert users["external_user_id"].tolist() == sorted(users["external_user_id"])

    def test_output_satisfies_the_pydantic_contract(self, raw):
        users = derive_users(canonicalize_interactions(raw[0]))
        for user_id in users["external_user_id"].head(10):
            assert User(user_id=user_id).created_at is None
