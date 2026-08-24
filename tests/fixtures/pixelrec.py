"""A synthetic dataset with PixelRec50K's exact file shape.

Deliberately generated rather than sampled from the real download: the licence
forbids redistributing the dataset, tests must run without it present, and a
fixture with hand-chosen properties can exercise edge cases the real file does
not contain (missing titles, duplicate rows, dangling item references).

The generator is deterministic - the same arguments always produce byte-identical
files - so a test that passes today passes tomorrow.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

#: Matches the official interaction.csv header exactly.
INTERACTION_HEADER = ["item_id", "user_id", "timestamp"]

#: Matches the official item_info.csv header exactly.
ITEM_INFO_HEADER = [
    "item_id",
    "view_number",
    "comment_number",
    "thumbup_number",
    "share_number",
    "coin_number",
    "favorite_number",
    "barrage_number",
    "title",
    "tag",
    "description",
]

#: 2022-01-01T00:00:00Z. Well inside the configured validation window.
BASE_TIMESTAMP = 1_640_995_200

#: One hour between a user's consecutive events, so ordering is unambiguous.
TIMESTAMP_STEP = 3600


@dataclass(frozen=True)
class FixtureSpec:
    """What the generated dataset should contain."""

    users: int = 12
    items: int = 20
    interactions_per_user: int = 8
    #: Items appearing in exactly one interaction, to exercise item filtering.
    singleton_items: int = 3
    #: Duplicate (user, item, timestamp) rows, to exercise deduplication.
    duplicate_rows: int = 2
    #: Interactions pointing at an item absent from item_info.csv.
    dangling_interactions: int = 1
    #: Items whose title cell is empty, mirroring the real file's 192.
    items_without_title: int = 2
    #: Items whose description cell is empty.
    items_without_description: int = 4


def write_fixture(target: Path, spec: FixtureSpec | None = None) -> FixtureSpec:
    """Write ``interaction.csv`` and ``item_info.csv`` into ``target``.

    Returns the spec used, so a test can assert against the intended shape.
    """
    spec = spec or FixtureSpec()
    target.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, int]] = []
    for user_index in range(spec.users):
        user_id = f"u{1000 + user_index}"
        for position in range(spec.interactions_per_user):
            # Cycle through the non-singleton items so every user shares history
            # with others - otherwise there is no collaborative signal at all.
            dense_items = spec.items - spec.singleton_items
            item_index = (user_index + position) % dense_items
            rows.append(
                (
                    f"i{item_index}",
                    user_id,
                    BASE_TIMESTAMP + user_index * 100_000 + position * TIMESTAMP_STEP,
                )
            )

    # Singleton items: one interaction each, to be removed by item filtering.
    for offset in range(spec.singleton_items):
        item_index = spec.items - spec.singleton_items + offset
        rows.append((f"i{item_index}", "u1000", BASE_TIMESTAMP + 900_000 + offset))

    # Exact duplicates of existing rows.
    rows.extend(rows[: spec.duplicate_rows])

    # Interactions referencing items that item_info.csv does not contain.
    for offset in range(spec.dangling_interactions):
        rows.append((f"i{9000 + offset}", "u1001", BASE_TIMESTAMP + 950_000 + offset))

    interaction_path = target / "interaction.csv"
    with interaction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(INTERACTION_HEADER)
        writer.writerows(rows)

    item_path = target / "item_info.csv"
    with item_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(ITEM_INFO_HEADER)
        for item_index in range(spec.items):
            counters = [float(100 + item_index * 7 + offset) for offset in range(7)]
            title = "" if item_index < spec.items_without_title else f"Video number {item_index}"
            description = (
                "" if item_index < spec.items_without_description else f"Description {item_index}"
            )
            writer.writerow(
                [f"i{item_index}", *counters, title, f"Category {item_index % 3}", description]
            )
    return spec


def write_feature_file(target: Path, *, item_ids: list[str], dimension: int = 4) -> Path:
    """Write a feature JSON file with PixelRec's ``{id: [floats]}`` shape."""
    target.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        f'"{item_id}": ['
        + ", ".join(str(float(index + position)) for position in range(dimension))
        + "]"
        for index, item_id in enumerate(item_ids)
    ]
    target.write_text("{" + ", ".join(parts) + "}")
    return target


__all__ = [
    "BASE_TIMESTAMP",
    "INTERACTION_HEADER",
    "ITEM_INFO_HEADER",
    "TIMESTAMP_STEP",
    "FixtureSpec",
    "write_feature_file",
    "write_fixture",
]
