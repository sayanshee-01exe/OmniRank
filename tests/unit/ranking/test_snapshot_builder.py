"""The snapshot builder's own wiring, exercised without fitting anything.

This file exists because of a specific, expensive failure. ``_fit_sources``
fits five models in sequence; SASRec is fourth. A stale call site with the
wrong arity sat there through popularity, BPR and LightGCN -- roughly one
hour fifty minutes of real training -- and only then raised ``TypeError``.
Twice, across two folds, that was nearly three hours to learn that a function
call had the wrong number of arguments.

Patching the five fit functions turns that same check into milliseconds. The
point is not to test the models; it is to prove every code path *between* them
is reachable and correctly wired before any of them are asked to train.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from omnirank.core.exceptions import OmniRankError
from omnirank.core.logging import get_logger
from omnirank.models.base import ScoredCandidate

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_build_ranking_snapshots", PROJECT_ROOT / "scripts" / "build_ranking_snapshots.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = _load_builder()


class _Model:
    """A retriever stand-in that returns a fixed scored list."""

    def __init__(self, name: str) -> None:
        self.name = name

    def recommend_batch_scored(
        self, user_ids: list[str], k: int, *, filter_seen: bool = True
    ) -> dict[str, list[ScoredCandidate]]:
        return {
            user: [
                ScoredCandidate(
                    item_id=f"i{index}", rank=index + 1, score=10.0 - index, source=self.name
                )
                for index in range(min(k, 3))
            ]
            for user in user_ids
        }


class _Dataset:
    """Just enough ProcessedDataset for the builder's fit path."""

    num_users = 3
    num_items = 5

    def __init__(self) -> None:
        self.mapping_metadata = {"item_mapping_checksum": "mapping-abc"}

    class _Identity:
        @staticmethod
        def to_dict() -> dict[str, str]:
            return {"dataset_version": "v1", "split_version": "1"}

    identity = _Identity()

    @staticmethod
    def external_to_internal_users() -> dict[str, int]:
        return {"u0": 0, "u1": 1, "u2": 2}

    @staticmethod
    def internal_to_external_items() -> dict[int, str]:
        return {index: f"i{index}" for index in range(5)}


def history_frame() -> pd.DataFrame:
    """Three users, three pre-cutoff interactions each."""
    rows = []
    for user in range(3):
        for order in range(3):
            rows.append(
                {
                    "internal_user_id": user,
                    "internal_item_id": order,
                    "interaction_order": order,
                    "timestamp": 1_000 + order,
                }
            )
    return pd.DataFrame(rows)


def targets_frame(*, timestamp: int = 2_000) -> pd.DataFrame:
    """Each user's held-out target, strictly after their history."""
    return pd.DataFrame(
        [
            {
                "internal_user_id": user,
                "internal_item_id": 4,
                "interaction_order": 3,
                "timestamp": timestamp,
            }
            for user in range(3)
        ]
    )


class _Args:
    offset = 3
    split = "train"
    budget = 10
    device = "cpu"
    max_users = None
    allow_source_failure = False
    overwrite = True
    skip_checksums = True


@pytest.fixture
def patched_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every fit function with an instant stand-in.

    The models are covered by their own suites. What is under test here is that
    ``_fit_sources`` calls each one and records its provenance correctly.
    """
    monkeypatch.setattr(BUILDER, "_locked_config", lambda source: _StubConfig(source), raising=True)
    monkeypatch.setattr(BUILDER, "_two_tower_config", lambda: _StubConfig("two_tower"))
    monkeypatch.setattr(BUILDER, "_wrap_two_tower", lambda *a, **k: _Model("two_tower"))
    monkeypatch.setattr(
        "omnirank.models.baselines.runner.fit_popularity",
        lambda *a, **k: (_Model("popularity"), None),
    )
    monkeypatch.setattr(
        "omnirank.models.baselines.runner.fit_bpr",
        lambda *a, **k: (_Model("matrix_factorization"), None),
    )
    monkeypatch.setattr(
        "omnirank.retrieval.runner.fit_lightgcn", lambda *a, **k: (_Model("lightgcn"), None)
    )
    monkeypatch.setattr(
        "omnirank.retrieval.runner.fit_sasrec", lambda *a, **k: (_Model("sasrec"), None)
    )
    monkeypatch.setattr(
        "omnirank.retrieval.runner.fit_two_tower",
        lambda *a, **k: ((None, None, None), None),
    )
    monkeypatch.setattr(
        "omnirank.retrieval.runner.sequences_from_fold",
        lambda *a, **k: pd.DataFrame(
            {"internal_user_id": [0], "item_sequence": [[0, 1]], "target_item": [2]}
        ),
    )


class _StubConfig:
    """Carries the attributes each record() call reads."""

    def __init__(self, source: str) -> None:
        self.half_life_days = 365.0
        self.embedding_dim = 64
        self.num_layers = 3
        self.seed = 42
        self.label = f"{source}-label"


class TestFitSourcesWiring:
    """Every source is reached, called and recorded. No model is trained."""

    def test_all_five_sources_are_fitted_and_recorded(
        self, patched_fits: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The regression this file exists for: a stale call site at source 4.

        Without this, an arity error in the SASRec branch is only discovered
        after popularity, BPR and LightGCN have finished training for real.
        """
        models, identities = BUILDER._fit_sources(
            _Dataset(), history_frame(), targets_frame(), None, _Args(), get_logger("t"), "run"
        )
        assert set(models) == set(BUILDER.SOURCES)
        assert [identity.source for identity in identities] == list(BUILDER.SOURCES)

    def test_every_identity_carries_the_required_provenance(self, patched_fits: None) -> None:
        _, identities = BUILDER._fit_sources(
            _Dataset(), history_frame(), targets_frame(), None, _Args(), get_logger("t"), "run"
        )
        for identity in identities:
            payload = identity.to_dict()
            for field in (
                "model_class",
                "model_version",
                "configuration_hash",
                "seed",
                "fit_boundary_timestamp",
                "max_fit_timestamp",
                "fit_interactions",
                "fit_users",
                "fit_items",
                "mapping_checksum",
                "candidate_budget",
                "device",
                "fit_seconds",
            ):
                assert payload[field] is not None, f"{identity.source} missing {field}"

    def test_recorded_boundaries_are_respected(self, patched_fits: None) -> None:
        _, identities = BUILDER._fit_sources(
            _Dataset(), history_frame(), targets_frame(), None, _Args(), get_logger("t"), "run"
        )
        assert all(identity.boundary_respected is True for identity in identities)


class TestPerUserTemporalGuard:
    """The cutoff is per user, because the fold is cut per user.

    An earlier version compared a global max-history against a global
    min-target. Those belong to different users, so the check failed instantly
    on correct data and aborted the build.
    """

    def test_correct_per_user_data_passes(self, patched_fits: None) -> None:
        BUILDER._fit_sources(
            _Dataset(), history_frame(), targets_frame(), None, _Args(), get_logger("t"), "run"
        )

    def test_a_user_whose_history_reaches_their_target_is_refused(self, patched_fits: None) -> None:
        leaky = history_frame()
        leaky.loc[0, "timestamp"] = 2_000  # equals that user's own target
        with pytest.raises(OmniRankError, match="reaches their own prediction cutoff"):
            BUILDER._fit_sources(
                _Dataset(), leaky, targets_frame(), None, _Args(), get_logger("t"), "run"
            )

    def test_globally_late_history_for_a_different_user_is_not_a_violation(
        self, patched_fits: None
    ) -> None:
        """The false positive that stopped the first real run.

        User 2's history runs far later than user 0's target. Per user, both are
        fine; compared globally, they look like a leak.
        """
        history = history_frame()
        history.loc[history.internal_user_id == 2, "timestamp"] = 50_000
        targets = targets_frame()
        targets.loc[targets.internal_user_id == 2, "timestamp"] = 60_000
        BUILDER._fit_sources(_Dataset(), history, targets, None, _Args(), get_logger("t"), "run")


class _Fold:
    def __init__(self, history: pd.DataFrame, targets: pd.DataFrame) -> None:
        self.history = history
        self.targets = targets


class TestRetrieveAndLabel:
    def test_a_failing_required_source_aborts(self) -> None:
        class Broken:
            name = "sasrec"

            def recommend_batch_scored(self, *a: object, **k: object) -> dict[str, list[Any]]:
                raise RuntimeError("source is down")

        models = {source: _Model(source) for source in BUILDER.SOURCES}
        models["sasrec"] = Broken()
        with pytest.raises(OmniRankError, match="Required retriever"):
            BUILDER._retrieve_and_label(
                models,
                _Dataset(),
                _Fold(history_frame(), targets_frame()),
                _Args(),
                get_logger("t"),
                "run",
            )

    def test_degraded_mode_records_the_failure_instead(self) -> None:
        class Broken:
            name = "sasrec"

            def recommend_batch_scored(self, *a: object, **k: object) -> dict[str, list[Any]]:
                raise RuntimeError("source is down")

        args = _Args()
        args.allow_source_failure = True
        models = {source: _Model(source) for source in BUILDER.SOURCES}
        models["sasrec"] = Broken()
        _, _, failed = BUILDER._retrieve_and_label(
            models,
            _Dataset(),
            _Fold(history_frame(), targets_frame()),
            args,
            get_logger("t"),
            "run",
        )
        assert failed == ["sasrec"]

    def test_rows_carry_real_cutoffs_and_genuine_scores(self) -> None:
        models = {source: _Model(source) for source in BUILDER.SOURCES}
        frame, stats, failed = BUILDER._retrieve_and_label(
            models,
            _Dataset(),
            _Fold(history_frame(), targets_frame()),
            _Args(),
            get_logger("t"),
            "run",
        )
        assert not failed
        assert (frame["as_of_timestamp"] == 2_000).all()
        assert (frame["fold_id"] == "offset_3").all()
        # The stub scores 10, 9, 8 -- a reciprocal-rank stand-in would be 1, .5, .33.
        assert set(frame["popularity_score"].dropna()) <= {10.0, 9.0, 8.0}
        assert stats.queries == 3

    def test_a_missed_target_is_counted_not_inserted(self) -> None:
        """The stubs never return item i4, which is every user's target."""
        models = {source: _Model(source) for source in BUILDER.SOURCES}
        frame, stats, _ = BUILDER._retrieve_and_label(
            models,
            _Dataset(),
            _Fold(history_frame(), targets_frame()),
            _Args(),
            get_logger("t"),
            "run",
        )
        assert stats.positive_queries == 0
        assert stats.zero_positive_queries == 3
        assert stats.candidate_recall == 0.0
        assert "i4" not in set(frame["external_item_id"])

    def test_source_contributions_are_counted(self) -> None:
        models = {source: _Model(source) for source in BUILDER.SOURCES}
        _, stats, _ = BUILDER._retrieve_and_label(
            models,
            _Dataset(),
            _Fold(history_frame(), targets_frame()),
            _Args(),
            get_logger("t"),
            "run",
        )
        assert set(stats.source_contributions) == set(BUILDER.SOURCES)
        assert all(count > 0 for count in stats.source_contributions.values())

    def test_the_fused_pool_is_capped(self) -> None:
        """Five sources at budget 500 would otherwise be ~100M rows overall."""
        assert BUILDER.FUSED_POOL_LIMIT > 0
        assert BUILDER.FUSED_POOL_LIMIT < 5 * BUILDER.DEFAULT_BUDGET


class TestCallSiteArity:
    """A static guard against the exact defect that cost three hours.

    Signature drift inside a long sequential function is invisible until the
    interpreter reaches that branch. Checking arity by inspection costs
    nothing and does not wait for four models to train.
    """

    def test_every_record_call_matches_the_helper_signature(self) -> None:
        import ast

        source = (PROJECT_ROOT / "scripts" / "build_ranking_snapshots.py").read_text()
        tree = ast.parse(source)
        definition: ast.FunctionDef | None = None
        calls: list[ast.Call] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "record":
                definition = node
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "record"
            ):
                calls.append(node)

        assert definition is not None, "record() helper not found"
        required = len(definition.args.args) - len(definition.args.defaults)
        assert calls, "no record() call sites found"
        for call in calls:
            supplied = len(call.args) + len(call.keywords)
            assert supplied >= required, (
                f"record() call at line {call.lineno} passes {supplied} arguments; "
                f"{required} are required"
            )
