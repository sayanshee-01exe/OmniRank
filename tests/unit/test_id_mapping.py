"""ID mapping: append-only guarantees, fingerprinting, persistence."""

from __future__ import annotations

import json

import pytest

from omnirank.core.exceptions import IdMappingError
from omnirank.data.id_mapping import UNKNOWN_INDEX, IdMapping


class TestConstruction:
    def test_indices_are_assigned_in_order(self):
        mapping = IdMapping("item", ["c", "a", "b"])
        assert mapping.to_index("c") == 0
        assert mapping.to_index("b") == 2

    def test_from_ids_sorts_by_default_for_reproducibility(self):
        first = IdMapping.from_ids("item", ["c", "a", "b"])
        second = IdMapping.from_ids("item", ["b", "c", "a"])
        assert first.ids == second.ids == ("a", "b", "c")
        assert first.fingerprint == second.fingerprint

    def test_from_ids_can_preserve_input_order(self):
        mapping = IdMapping.from_ids("item", ["c", "a"], sort=False)
        assert mapping.ids == ("c", "a")

    def test_duplicates_collapse(self):
        assert len(IdMapping("user", ["a", "a", "b"])) == 2

    def test_empty_identifier_is_rejected(self):
        with pytest.raises(IdMappingError):
            IdMapping("user", [""])


class TestAppendOnly:
    def test_existing_indices_never_move(self):
        mapping = IdMapping("item", ["a", "b"])
        before = {identifier: mapping.to_index(identifier) for identifier in ("a", "b")}
        mapping.extend(["c", "d", "a"])
        after = {identifier: mapping.to_index(identifier) for identifier in ("a", "b")}
        assert before == after
        assert mapping.to_index("c") == 2

    def test_add_is_idempotent(self):
        mapping = IdMapping("item")
        assert mapping.add("x") == mapping.add("x") == 0
        assert len(mapping) == 1


class TestLookup:
    def test_round_trip(self):
        mapping = IdMapping("item", ["a", "b", "c"])
        assert mapping.to_id(mapping.to_index("b")) == "b"

    def test_unknown_identifier_raises_by_default(self):
        with pytest.raises(IdMappingError) as exc:
            IdMapping("item", ["a"]).to_index("zzz")
        assert "zzz" in str(exc.value)

    def test_unknown_identifier_can_return_a_default(self):
        mapping = IdMapping("item", ["a"])
        assert mapping.to_index("zzz", default=UNKNOWN_INDEX) == UNKNOWN_INDEX

    def test_out_of_range_index_raises(self):
        with pytest.raises(IdMappingError):
            IdMapping("item", ["a"]).to_id(99)

    def test_negative_index_raises(self):
        """UNKNOWN_INDEX must never resolve to the last row via Python's -1."""
        with pytest.raises(IdMappingError):
            IdMapping("item", ["a", "b"]).to_id(UNKNOWN_INDEX)

    def test_vectorised_helpers(self):
        mapping = IdMapping("item", ["a", "b", "c"])
        assert mapping.to_indices(["c", "a"]) == [2, 0]
        assert mapping.to_ids([2, 0]) == ["c", "a"]

    def test_membership_and_iteration(self):
        mapping = IdMapping("item", ["a", "b"])
        assert "a" in mapping
        assert "z" not in mapping
        assert list(mapping) == ["a", "b"]


class TestFingerprint:
    def test_same_ids_same_order_match(self):
        assert (
            IdMapping("item", ["a", "b"]).fingerprint == IdMapping("item", ["a", "b"]).fingerprint
        )

    def test_different_order_differs(self):
        assert (
            IdMapping("item", ["a", "b"]).fingerprint != IdMapping("item", ["b", "a"]).fingerprint
        )

    def test_different_entity_differs(self):
        assert IdMapping("item", ["a"]).fingerprint != IdMapping("user", ["a"]).fingerprint

    def test_separator_prevents_concatenation_collisions(self):
        """['ab','c'] and ['a','bc'] must not hash alike."""
        assert IdMapping("i", ["ab", "c"]).fingerprint != IdMapping("i", ["a", "bc"]).fingerprint


class TestPersistence:
    def test_round_trip(self, tmp_path):
        mapping = IdMapping("item", ["a", "b", "c"])
        path = mapping.save(tmp_path / "nested" / "items.json")
        loaded = IdMapping.load(path)
        assert loaded.ids == mapping.ids
        assert loaded.entity == "item"
        assert loaded.fingerprint == mapping.fingerprint

    def test_saved_file_is_readable_json(self, tmp_path):
        path = IdMapping("item", ["a"]).save(tmp_path / "m.json")
        payload = json.loads(path.read_text())
        assert payload["entity"] == "item"
        assert payload["ids"] == ["a"]
        assert payload["size"] == 1

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(IdMappingError):
            IdMapping.load(tmp_path / "absent.json")

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(IdMappingError):
            IdMapping.load(path)

    def test_unsupported_format_version_raises(self, tmp_path):
        path = tmp_path / "old.json"
        path.write_text(json.dumps({"format_version": 999, "entity": "item", "ids": []}))
        with pytest.raises(IdMappingError) as exc:
            IdMapping.load(path)
        assert "format version" in str(exc.value).lower()

    def test_tampered_file_is_detected(self, tmp_path):
        """The whole point of the fingerprint: a silently edited mapping fails loudly."""
        path = IdMapping("item", ["a", "b"]).save(tmp_path / "m.json")
        payload = json.loads(path.read_text())
        payload["ids"] = ["b", "a"]  # reorder without updating the fingerprint
        path.write_text(json.dumps(payload))
        with pytest.raises(IdMappingError) as exc:
            IdMapping.load(path)
        assert "fingerprint" in str(exc.value).lower()
