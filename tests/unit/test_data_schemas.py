"""Data contract validation: User, Item, Interaction."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from omnirank.data.schemas import EventType, Interaction, Item, User


class TestUser:
    def test_valid(self, user_row):
        user = User.model_validate(user_row)
        assert user.user_id == "u1"
        assert user.created_at.tzinfo is UTC

    def test_missing_id_is_rejected(self):
        with pytest.raises(ValidationError):
            User.model_validate({"created_at": "2026-01-01T00:00:00Z"})

    def test_empty_id_is_rejected(self):
        with pytest.raises(ValidationError):
            User.model_validate({"user_id": "   ", "created_at": "2026-01-01T00:00:00Z"})

    def test_naive_timestamp_is_rejected(self):
        with pytest.raises(ValidationError) as exc:
            User.model_validate({"user_id": "u1", "created_at": datetime(2026, 1, 1)})
        assert "timezone-aware" in str(exc.value)

    def test_non_utc_timestamp_is_normalised_not_rejected(self):
        offset = timezone(timedelta(hours=5, minutes=30))
        user = User.model_validate(
            {"user_id": "u1", "created_at": datetime(2026, 1, 1, 5, 30, tzinfo=offset)}
        )
        assert user.created_at == datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

    def test_unknown_field_is_rejected(self, user_row):
        with pytest.raises(ValidationError):
            User.model_validate({**user_row, "email": "a@b.com"})

    def test_domain_specific_data_goes_in_attributes(self, user_row):
        user = User.model_validate({**user_row, "attributes": {"segment": "new"}})
        assert user.attributes["segment"] == "new"

    def test_is_immutable(self, user_row):
        user = User.model_validate(user_row)
        with pytest.raises(ValidationError):
            setattr(user, "user_id", "u2")  # noqa: B010


class TestItem:
    def test_valid(self, item_row):
        item = Item.model_validate(item_row)
        assert item.price == 99.5
        assert item.available is True

    def test_negative_price_is_rejected(self, item_row):
        with pytest.raises(ValidationError):
            Item.model_validate({**item_row, "price": -1.0})

    def test_missing_price_is_allowed(self, item_row):
        assert Item.model_validate({**item_row, "price": None}).price is None

    def test_infinite_price_is_rejected(self, item_row):
        with pytest.raises(ValidationError):
            Item.model_validate({**item_row, "price": float("inf")})

    def test_empty_title_is_rejected(self, item_row):
        with pytest.raises(ValidationError):
            Item.model_validate({**item_row, "title": ""})

    def test_non_boolean_availability_is_rejected(self, item_row):
        with pytest.raises(ValidationError):
            Item.model_validate({**item_row, "available": "maybe"})

    @pytest.mark.parametrize("value", [True, False, "true", "false", 1, 0])
    def test_recognised_availability_values(self, item_row, value):
        Item.model_validate({**item_row, "available": value})

    def test_missing_modalities_are_allowed(self, item_row):
        item = Item.model_validate(
            {**item_row, "description": None, "image_id": None, "brand": None}
        )
        assert item.description is None
        assert item.image_id is None


class TestInteraction:
    def test_valid(self, interaction_row):
        event = Interaction.model_validate(interaction_row)
        assert event.event_type is EventType.CLICK

    def test_unknown_event_type_is_rejected(self, interaction_row):
        with pytest.raises(ValidationError):
            Interaction.model_validate({**interaction_row, "event_type": "teleport"})

    @pytest.mark.parametrize("event", ["view", "click", "wishlist", "cart", "purchase"])
    def test_every_declared_event_type_without_a_value(self, interaction_row, event):
        Interaction.model_validate({**interaction_row, "event_type": event})

    def test_rating_without_a_value_is_rejected(self, interaction_row):
        with pytest.raises(ValidationError) as exc:
            Interaction.model_validate({**interaction_row, "event_type": "rating"})
        assert "event_value" in str(exc.value)

    def test_rating_with_a_value_is_accepted(self, interaction_row):
        event = Interaction.model_validate(
            {**interaction_row, "event_type": "rating", "event_value": 4.0}
        )
        assert event.event_value == 4.0

    def test_nan_event_value_is_rejected(self, interaction_row):
        with pytest.raises(ValidationError):
            Interaction.model_validate({**interaction_row, "event_value": float("nan")})

    def test_negative_weight_is_rejected(self, interaction_row):
        with pytest.raises(ValidationError):
            Interaction.model_validate({**interaction_row, "weight": -0.5})

    def test_missing_user_id_is_rejected(self, interaction_row):
        payload = dict(interaction_row)
        del payload["user_id"]
        with pytest.raises(ValidationError):
            Interaction.model_validate(payload)

    def test_naive_timestamp_is_rejected(self, interaction_row):
        with pytest.raises(ValidationError):
            Interaction.model_validate({**interaction_row, "timestamp": datetime(2026, 2, 1)})

    def test_dedup_key_ignores_interaction_id(self, interaction_row):
        first = Interaction.model_validate(interaction_row)
        resent = Interaction.model_validate({**interaction_row, "interaction_id": "e2"})
        assert first.dedup_key == resent.dedup_key

    def test_dedup_key_distinguishes_different_events(self, interaction_row):
        click = Interaction.model_validate(interaction_row)
        purchase = Interaction.model_validate({**interaction_row, "event_type": "purchase"})
        assert click.dedup_key != purchase.dedup_key
