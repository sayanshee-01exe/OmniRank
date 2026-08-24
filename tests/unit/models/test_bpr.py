"""BPR matrix factorization.

Skipped wholesale when torch is absent, so the core test suite still runs
without the ``baseline`` extra installed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="BPR requires the 'baseline' extra")

from omnirank.core.exceptions import (  # noqa: E402
    ArtifactValidationError,
    DataError,
    ModelNotFittedError,
)
from omnirank.models.baselines.bpr import (  # noqa: E402
    UNKNOWN_ITEM_SCORE,
    BPRConfig,
    BPRFitData,
    BPRMatrixFactorization,
    resolve_torch_device,
)

CLUSTERS = 3
ITEMS_PER_CLUSTER = 10
USERS = 60


@pytest.fixture
def clustered_data() -> BPRFitData:
    """Users in three clusters, each preferring a distinct block of items.

    Learnable by construction, so "the loss went down" can be checked against
    "the model ranks the right block first".
    """
    rng = np.random.default_rng(0)
    rows = [
        (user, (user % CLUSTERS) * ITEMS_PER_CLUSTER + int(rng.integers(0, ITEMS_PER_CLUSTER)))
        for user in range(USERS)
        for _ in range(12)
    ]
    frame = pd.DataFrame(rows, columns=["internal_user_id", "internal_item_id"]).drop_duplicates()
    items = CLUSTERS * ITEMS_PER_CLUSTER
    return BPRFitData(
        interactions=frame,
        num_users=USERS,
        num_items=items,
        internal_to_external_item={index: f"i{index}" for index in range(items)},
        external_to_internal_user={f"u{user}": user for user in range(USERS)},
        mapping_checksum="checksum-xyz",
    )


@pytest.fixture
def trained(clustered_data) -> BPRMatrixFactorization:
    model = BPRMatrixFactorization(
        BPRConfig(embedding_dim=16, epochs=12, batch_size=64, learning_rate=0.05, seed=7),
        device="cpu",
    )
    model.fit(clustered_data)
    return model


class TestConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"embedding_dim": 0},
            {"embedding_dim": -1},
            {"learning_rate": 0.0},
            {"learning_rate": -0.1},
            {"regularization": -1e-5},
            {"batch_size": 0},
            {"epochs": 0},
            {"negatives_per_positive": 0},
            {"evaluation_user_batch_size": 0},
        ],
    )
    def test_invalid_values_rejected(self, kwargs):
        with pytest.raises(DataError):
            BPRConfig(**kwargs)

    def test_valid_config_accepted(self):
        assert BPRConfig(embedding_dim=8, epochs=1).embedding_dim == 8

    def test_label_is_descriptive(self):
        assert "d32" in BPRConfig(embedding_dim=32).label


class TestDeviceResolution:
    def test_cpu_is_always_available(self):
        assert resolve_torch_device("cpu").type == "cpu"

    def test_auto_never_selects_cuda(self):
        assert resolve_torch_device("auto").type in {"cpu", "mps"}

    def test_cuda_requires_explicit_permission(self):
        """Falls back rather than failing: a slower run beats no run."""
        assert resolve_torch_device("cuda", allow_cuda=False).type == "cpu"

    def test_mps_falls_back_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert resolve_torch_device("mps").type == "cpu"

    def test_mps_selected_when_available(self, monkeypatch):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert resolve_torch_device("auto").type == "mps"


class TestTraining:
    def test_loss_decreases(self, trained):
        history = trained.loss_history
        assert history[-1] < history[0]

    def test_loss_is_finite_throughout(self, trained):
        assert all(np.isfinite(value) for value in trained.loss_history)

    def test_records_one_loss_per_epoch(self, clustered_data):
        model = BPRMatrixFactorization(
            BPRConfig(embedding_dim=8, epochs=4, batch_size=64), device="cpu"
        )
        model.fit(clustered_data)
        assert len(model.loss_history) == 4

    def test_learns_the_cluster_structure(self, trained):
        """Positive scores must exceed negative ones for a learnable fixture."""
        in_block = [f"i{index}" for index in range(ITEMS_PER_CLUSTER)]
        out_block = [
            f"i{index}" for index in range(ITEMS_PER_CLUSTER, CLUSTERS * ITEMS_PER_CLUSTER)
        ]
        inside = np.mean(trained.score("u0", in_block))
        outside = np.mean(trained.score("u0", out_block))
        assert inside > outside

    def test_embedding_shapes(self, trained, clustered_data):
        assert trained._user_factors.shape == (clustered_data.num_users, 16)
        assert trained._item_factors.shape == (clustered_data.num_items, 16)

    def test_empty_fit_data_rejected(self):
        empty = pd.DataFrame(columns=["internal_user_id", "internal_item_id"])
        data = BPRFitData(empty, 1, 2, {}, {})
        with pytest.raises(DataError):
            BPRMatrixFactorization(BPRConfig(epochs=1)).fit(data)

    def test_wrong_bundle_type_rejected(self):
        with pytest.raises(DataError):
            BPRMatrixFactorization().fit({"not": "a bundle"})

    def test_diverging_learning_rate_is_caught(self, clustered_data):
        """A non-finite loss must fail loudly, not produce a nan-filled model."""
        model = BPRMatrixFactorization(
            BPRConfig(embedding_dim=8, epochs=2, batch_size=32, learning_rate=1e12), device="cpu"
        )
        try:
            model.fit(clustered_data)
        except DataError as exc:
            assert "non-finite" in str(exc).lower()
        else:
            assert all(np.isfinite(value) for value in model.loss_history)


class TestDeterminism:
    def test_same_seed_same_recommendations(self, clustered_data):
        def build():
            model = BPRMatrixFactorization(
                BPRConfig(embedding_dim=8, epochs=3, batch_size=64, seed=11), device="cpu"
            )
            model.fit(clustered_data)
            return model.recommend_batch([f"u{u}" for u in range(10)], 5)

        assert build() == build()

    def test_different_seeds_differ(self, clustered_data):
        def build(seed):
            model = BPRMatrixFactorization(
                BPRConfig(embedding_dim=8, epochs=3, batch_size=64, seed=seed), device="cpu"
            )
            model.fit(clustered_data)
            return model.recommend_batch([f"u{u}" for u in range(10)], 5)

        assert build(1) != build(2)


class TestRetrieval:
    def test_batched_matches_naive(self, trained):
        """The memory-bounded path must return exactly the naive result."""
        users = [f"u{u}" for u in range(USERS)]
        batched = trained.recommend_batch(users, 10)
        naive = {user: [c.item_id for c in trained.recommend(user, 10)] for user in users}
        assert batched == naive

    @pytest.mark.parametrize("batch_size", [1, 7, 1000])
    def test_user_batch_size_does_not_change_results(self, clustered_data, batch_size):
        reference = None
        for size in (batch_size, 3):
            model = BPRMatrixFactorization(
                BPRConfig(
                    embedding_dim=8,
                    epochs=3,
                    batch_size=64,
                    seed=5,
                    evaluation_user_batch_size=size,
                ),
                device="cpu",
            )
            model.fit(clustered_data)
            result = model.recommend_batch([f"u{u}" for u in range(20)], 5)
            if reference is None:
                reference = result
            else:
                assert result == reference

    def test_seen_items_are_masked(self, trained, clustered_data):
        seen = clustered_data.interactions
        for user in [f"u{u}" for u in range(10)]:
            internal = int(user[1:])
            observed = set(seen.loc[seen.internal_user_id == internal, "internal_item_id"].tolist())
            recommended = {int(item[1:]) for item in trained.recommend_batch([user], 10)[user]}
            assert not (recommended & observed)

    def test_never_recommends_outside_the_fit_catalogue(self, clustered_data):
        subset = clustered_data.interactions[clustered_data.interactions.internal_item_id < 20]
        data = BPRFitData(
            subset,
            clustered_data.num_users,
            clustered_data.num_items,
            clustered_data.internal_to_external_item,
            clustered_data.external_to_internal_user,
        )
        model = BPRMatrixFactorization(
            BPRConfig(embedding_dim=8, epochs=2, batch_size=64), device="cpu"
        )
        model.fit(data)
        for items in model.recommend_batch([f"u{u}" for u in range(10)], 10).values():
            assert all(int(item[1:]) < 20 for item in items)

    def test_no_padding_ids_leak_into_output(self, clustered_data):
        """A -1 sentinel must never be converted into an item id."""
        model = BPRMatrixFactorization(
            BPRConfig(embedding_dim=8, epochs=1, batch_size=64), device="cpu"
        )
        model.fit(clustered_data)
        items = model.recommend_batch([f"u{u}" for u in range(5)], 1000)
        for recommended in items.values():
            assert all(item.startswith("i") for item in recommended)

    def test_invalid_k_rejected(self, trained):
        with pytest.raises(DataError):
            trained.recommend("u0", 0)


class TestUnknownEntities:
    def test_unknown_user_returns_empty_not_random(self, trained):
        """Inventing an embedding would produce confident-looking nonsense."""
        assert trained.recommend("nobody", 10) == []

    def test_unknown_user_batch_returns_empty_list(self, trained):
        assert trained.recommend_batch(["nobody"], 10) == {"nobody": []}

    def test_unknown_user_scores_zero(self, trained):
        assert trained.score("nobody", ["i0", "i1"]) == [UNKNOWN_ITEM_SCORE] * 2

    def test_unknown_item_scores_zero_without_raising(self, trained):
        assert trained.score("u0", ["not-an-item"]) == [UNKNOWN_ITEM_SCORE]

    def test_score_preserves_input_order_and_length(self, trained):
        scores = trained.score("u0", ["i5", "not-an-item", "i1"])
        assert len(scores) == 3
        assert scores[1] == UNKNOWN_ITEM_SCORE


class TestFittedState:
    def test_recommend_before_fit_raises(self):
        with pytest.raises(ModelNotFittedError):
            BPRMatrixFactorization().recommend("u0", 5)

    def test_score_before_fit_raises(self):
        with pytest.raises(ModelNotFittedError):
            BPRMatrixFactorization().score("u0", ["i0"])


class TestPersistence:
    def test_recommendations_identical_after_load(self, trained, tmp_path):
        users = [f"u{u}" for u in range(USERS)]
        before = trained.recommend_batch(users, 10)
        trained.save(tmp_path / "model")
        loaded = BPRMatrixFactorization.load(tmp_path / "model", device="cpu")
        assert loaded.recommend_batch(users, 10) == before

    def test_scores_identical_after_load(self, trained, tmp_path):
        items = [f"i{index}" for index in range(30)]
        before = trained.score("u0", items)
        trained.save(tmp_path / "model")
        after = BPRMatrixFactorization.load(tmp_path / "model", device="cpu").score("u0", items)
        assert after == pytest.approx(before, abs=1e-9)

    def test_fitted_state_restored(self, trained, tmp_path):
        trained.save(tmp_path / "model")
        loaded = BPRMatrixFactorization.load(tmp_path / "model", device="cpu")
        assert loaded.is_fitted
        assert loaded.config.embedding_dim == trained.config.embedding_dim
        assert loaded.loss_history == trained.loss_history

    def test_saved_tensors_are_device_neutral(self, trained, tmp_path):
        """An artifact trained on MPS must load on a CPU-only host."""
        trained.save(tmp_path / "model")
        state = torch.load(tmp_path / "model" / "state.pt", map_location="cpu", weights_only=True)
        assert state["user_factors"].device.type == "cpu"

    def test_missing_files_fail_clearly(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ArtifactValidationError):
            BPRMatrixFactorization.load(tmp_path / "empty")

    def test_corrupted_state_fails_clearly(self, trained, tmp_path):
        trained.save(tmp_path / "model")
        (tmp_path / "model" / "state.pt").write_bytes(b"not a tensor file")
        with pytest.raises(ArtifactValidationError):
            BPRMatrixFactorization.load(tmp_path / "model")

    def test_wrong_model_type_fails_clearly(self, trained, tmp_path):
        trained.save(tmp_path / "model")
        path = tmp_path / "model" / "config.json"
        payload = json.loads(path.read_text())
        payload["model"] = "popularity"
        path.write_text(json.dumps(payload))
        with pytest.raises(ArtifactValidationError):
            BPRMatrixFactorization.load(tmp_path / "model")

    def test_unsupported_format_version_fails_clearly(self, trained, tmp_path):
        trained.save(tmp_path / "model")
        path = tmp_path / "model" / "config.json"
        payload = json.loads(path.read_text())
        payload["format_version"] = 999
        path.write_text(json.dumps(payload))
        with pytest.raises(ArtifactValidationError):
            BPRMatrixFactorization.load(tmp_path / "model")

    def test_mapping_mismatch_fails_clearly(self, trained):
        trained.require_mapping("checksum-xyz")
        with pytest.raises(ArtifactValidationError):
            trained.require_mapping("wrong-checksum")


class TestMetadata:
    def test_records_sampler_and_duplicate_policy(self, trained):
        metadata = trained.metadata()
        assert metadata["negative_sampler"]["strategy"] == "uniform"
        assert metadata["duplicate_positive_policy"] == "unique_binary"

    def test_records_device_and_config(self, trained):
        metadata = trained.metadata()
        assert metadata["device"] == "cpu"
        assert metadata["config"]["embedding_dim"] == 16
