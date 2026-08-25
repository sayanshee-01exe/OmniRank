"""LightGCN graph propagation and collaborative retrieval.

The propagation tests carry most of the weight here. LightGCN's only moving
part is the normalised adjacency and the layer averaging on top of it; if those
are wrong the model still trains, still reports a falling loss, and still
returns plausible recommendations -- it is simply a slower matrix
factorization. So the normalisation constants are checked against values
computed by hand rather than against the implementation's own output.

Skipped wholesale when torch is absent, so the core suite still runs without
the ``retrieval`` extra installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="LightGCN requires the 'retrieval' extra")

from omnirank.core.exceptions import (  # noqa: E402
    ArtifactValidationError,
    DataError,
    ModelNotFittedError,
)
from omnirank.models.lightgcn import (  # noqa: E402
    LightGCN,
    LightGCNConfig,
    LightGCNFitData,
    build_normalized_adjacency,
    propagate,
)

CLUSTERS = 3
ITEMS_PER_CLUSTER = 10
USERS = 60


@pytest.fixture
def clustered_data() -> LightGCNFitData:
    """Users in three clusters, each preferring a distinct block of items."""
    rng = np.random.default_rng(0)
    rows = [
        (user, (user % CLUSTERS) * ITEMS_PER_CLUSTER + int(rng.integers(0, ITEMS_PER_CLUSTER)))
        for user in range(USERS)
        for _ in range(12)
    ]
    frame = pd.DataFrame(rows, columns=["internal_user_id", "internal_item_id"]).drop_duplicates()
    items = CLUSTERS * ITEMS_PER_CLUSTER
    return LightGCNFitData(
        edges=frame,
        num_users=USERS,
        num_items=items,
        internal_to_external_item={index: f"i{index}" for index in range(items)},
        external_to_internal_user={f"u{user}": user for user in range(USERS)},
        mapping_checksum="checksum-xyz",
    )


@pytest.fixture
def fitted(clustered_data: LightGCNFitData) -> LightGCN:
    """A small, quick, converged model."""
    model = LightGCN(
        LightGCNConfig(
            embedding_dim=16,
            num_layers=2,
            max_epochs=25,
            batch_size=64,
            learning_rate=0.05,
            early_stopping_patience=30,
            seed=7,
        ),
        device="cpu",
    )
    model.fit(clustered_data)
    return model


class TestNormalizedAdjacency:
    """The symmetric normalisation, checked against hand arithmetic."""

    @staticmethod
    def _tiny() -> torch.Tensor:
        # u0 -> i0, i1;  u1 -> i1.  Degrees: u0=2, u1=1, i0=1, i1=2.
        adjacency, _ = build_normalized_adjacency(
            np.array([0, 0, 1]), np.array([0, 1, 1]), num_users=2, num_items=2
        )
        return adjacency.to_dense()

    def test_entries_are_one_over_sqrt_degree_product(self) -> None:
        dense = self._tiny()
        # Nodes are laid out [users | items], so item j sits at column 2 + j.
        assert dense[0, 2] == pytest.approx(1 / np.sqrt(2 * 1))  # u0-i0
        assert dense[0, 3] == pytest.approx(1 / np.sqrt(2 * 2))  # u0-i1
        assert dense[1, 3] == pytest.approx(1 / np.sqrt(1 * 2))  # u1-i1

    def test_adjacency_is_symmetric(self) -> None:
        dense = self._tiny()
        assert torch.equal(dense, dense.T)

    def test_no_user_user_or_item_item_edges(self) -> None:
        """The graph is bipartite: the diagonal blocks must be empty."""
        dense = self._tiny()
        assert dense[:2, :2].abs().sum() == 0
        assert dense[2:, 2:].abs().sum() == 0

    def test_isolated_node_yields_a_zero_row_not_a_nan(self) -> None:
        """An item nobody touched has degree zero; 1/sqrt(0) must not leak in."""
        adjacency, _ = build_normalized_adjacency(
            np.array([0]), np.array([0]), num_users=1, num_items=3
        )
        dense = adjacency.to_dense()
        assert not torch.isnan(dense).any()
        assert dense[3].abs().sum() == 0  # item 2, never interacted with

    def test_checksum_is_stable_and_edge_sensitive(self) -> None:
        first = build_normalized_adjacency(
            np.array([0, 0, 1]), np.array([0, 1, 1]), num_users=2, num_items=2
        )[1]
        same = build_normalized_adjacency(
            np.array([0, 0, 1]), np.array([0, 1, 1]), num_users=2, num_items=2
        )[1]
        different = build_normalized_adjacency(
            np.array([0, 0, 1]), np.array([0, 1, 0]), num_users=2, num_items=2
        )[1]
        assert first == same
        assert first != different

    def test_rejects_edges_outside_the_declared_node_counts(self) -> None:
        with pytest.raises(DataError):
            build_normalized_adjacency(np.array([0, 5]), np.array([0, 0]), num_users=2, num_items=2)

    def test_rejects_mismatched_edge_arrays(self) -> None:
        with pytest.raises(DataError):
            build_normalized_adjacency(np.array([0, 1]), np.array([0]), num_users=2, num_items=2)


class TestPropagation:
    """Layer averaging, checked against the closed form."""

    @staticmethod
    def _setup() -> tuple[torch.Tensor, torch.Tensor]:
        adjacency, _ = build_normalized_adjacency(
            np.array([0, 0, 1]), np.array([0, 1, 1]), num_users=2, num_items=2
        )
        torch.manual_seed(0)
        return adjacency, torch.randn(4, 8)

    def test_zero_layers_is_the_identity(self) -> None:
        """With no propagation LightGCN degenerates to plain matrix factorization."""
        adjacency, embeddings = self._setup()
        assert torch.allclose(propagate(adjacency, embeddings, 0), embeddings)

    def test_one_layer_is_the_mean_of_layers_zero_and_one(self) -> None:
        adjacency, embeddings = self._setup()
        expected = (embeddings + torch.sparse.mm(adjacency, embeddings)) / 2
        assert torch.allclose(propagate(adjacency, embeddings, 1), expected, atol=1e-6)

    def test_two_layers_averages_three_terms_equally(self) -> None:
        """Equal alpha_k = 1/(K+1); no learned layer weights, as the paper specifies."""
        adjacency, embeddings = self._setup()
        first = torch.sparse.mm(adjacency, embeddings)
        second = torch.sparse.mm(adjacency, first)
        expected = (embeddings + first + second) / 3
        assert torch.allclose(propagate(adjacency, embeddings, 2), expected, atol=1e-6)

    def test_propagation_never_produces_nan(self) -> None:
        adjacency, _ = build_normalized_adjacency(
            np.array([0]), np.array([0]), num_users=2, num_items=3
        )
        result = propagate(adjacency, torch.randn(5, 4), 3)
        assert not torch.isnan(result).any()


class TestConfig:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("embedding_dim", 0),
            ("num_layers", -1),
            ("learning_rate", 0.0),
            ("batch_size", 0),
            ("max_epochs", 0),
            ("negatives_per_positive", 0),
        ],
    )
    def test_rejects_invalid_values(self, field: str, value: float) -> None:
        with pytest.raises(DataError):
            LightGCNConfig(**{field: value})

    def test_zero_layers_is_allowed_because_it_is_the_mf_ablation(self) -> None:
        assert LightGCNConfig(num_layers=0).num_layers == 0


class TestFitting:
    def test_loss_decreases(self, fitted: LightGCN) -> None:
        history = fitted.loss_history
        assert len(history) > 1
        assert history[-1] < history[0]

    def test_learns_the_cluster_structure(self, fitted: LightGCN) -> None:
        """u0 belongs to cluster 0, so block 0 must outscore the rest."""
        own = np.mean(fitted.score("u0", [f"i{i}" for i in range(ITEMS_PER_CLUSTER)]))
        other = np.mean(
            fitted.score(
                "u0", [f"i{i}" for i in range(ITEMS_PER_CLUSTER, CLUSTERS * ITEMS_PER_CLUSTER)]
            )
        )
        assert own > other

    def test_refuses_a_non_lightgcn_bundle(self) -> None:
        with pytest.raises(DataError):
            LightGCN().fit({"edges": []})

    def test_refuses_empty_interactions(self, clustered_data: LightGCNFitData) -> None:
        empty = LightGCNFitData(
            edges=clustered_data.edges.iloc[:0],
            num_users=1,
            num_items=1,
            internal_to_external_item={},
            external_to_internal_user={},
        )
        with pytest.raises(DataError):
            LightGCN().fit(empty)

    def test_is_deterministic_under_a_fixed_seed(self, clustered_data: LightGCNFitData) -> None:
        def train() -> list[str]:
            model = LightGCN(
                LightGCNConfig(embedding_dim=8, num_layers=1, max_epochs=5, seed=3), device="cpu"
            )
            model.fit(clustered_data)
            return model.recommend_batch(["u0", "u1"], 5)["u0"]

        assert train() == train()


class TestRecommendation:
    def test_unfitted_model_refuses_to_recommend(self) -> None:
        with pytest.raises(ModelNotFittedError):
            LightGCN().recommend("u0", 5)

    def test_unknown_user_returns_nothing(self, fitted: LightGCN) -> None:
        """A collaborative model has no embedding for an unseen user."""
        assert fitted.recommend("stranger", 5) == []

    def test_unknown_item_scores_zero(self, fitted: LightGCN) -> None:
        assert fitted.score("u0", ["not-an-item"]) == [0.0]

    def test_seen_items_are_filtered_by_default(self, fitted: LightGCN) -> None:
        recommended = fitted.recommend_batch([f"u{u}" for u in range(USERS)], 10)
        for user, items in recommended.items():
            seen = set(fitted._seen_by_user[int(user[1:])].tolist())
            assert not {int(item[1:]) for item in items} & seen

    def test_batching_matches_one_at_a_time(self, fitted: LightGCN) -> None:
        """The memory-bounded path must not change the answer."""
        users = [f"u{u}" for u in range(USERS)]
        batched = fitted.recommend_batch(users, 10)
        naive = {u: [c.item_id for c in fitted.recommend(u, 10)] for u in users}
        assert batched == naive

    def test_candidates_carry_provenance(self, fitted: LightGCN) -> None:
        candidate = fitted.recommend("u0", 1)[0]
        assert candidate.sources == ("lightgcn",)
        assert candidate.source_scores["lightgcn"] == candidate.score

    def test_rejects_non_positive_k(self, fitted: LightGCN) -> None:
        with pytest.raises(DataError):
            fitted.recommend("u0", 0)


class TestPersistence:
    def test_round_trip_preserves_recommendations(self, fitted: LightGCN, tmp_path) -> None:
        fitted.save(tmp_path / "model")
        loaded = LightGCN.load(tmp_path / "model", device="cpu")
        users = [f"u{u}" for u in range(USERS)]
        assert loaded.recommend_batch(users, 10) == fitted.recommend_batch(users, 10)

    def test_round_trip_preserves_scores_exactly(self, fitted: LightGCN, tmp_path) -> None:
        fitted.save(tmp_path / "model")
        loaded = LightGCN.load(tmp_path / "model", device="cpu")
        items = [f"i{i}" for i in range(CLUSTERS * ITEMS_PER_CLUSTER)]
        assert loaded.score("u0", items) == fitted.score("u0", items)

    def test_missing_files_are_reported_not_crashed(self, tmp_path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(ArtifactValidationError):
            LightGCN.load(tmp_path / "empty")

    def test_rejects_an_artifact_from_another_model(self, fitted: LightGCN, tmp_path) -> None:
        fitted.save(tmp_path / "model")
        config = tmp_path / "model" / "config.json"
        config.write_text(config.read_text().replace('"lightgcn"', '"something-else"', 1))
        with pytest.raises(ArtifactValidationError):
            LightGCN.load(tmp_path / "model")

    def test_mapping_checksum_mismatch_is_refused(self, fitted: LightGCN) -> None:
        """A different mapping means every recommended id resolves to the wrong item."""
        fitted.require_mapping("checksum-xyz")
        with pytest.raises(ArtifactValidationError):
            fitted.require_mapping("a-different-checksum")

    def test_graph_checksum_mismatch_is_refused(self, fitted: LightGCN) -> None:
        """Propagated embeddings encode the adjacency they were trained on."""
        fitted.require_graph(fitted.metadata()["graph_checksum"])
        with pytest.raises(ArtifactValidationError):
            fitted.require_graph("a-different-graph")


class TestExport:
    def test_item_embeddings_are_float32_for_indexing(self, fitted: LightGCN) -> None:
        embeddings = fitted.item_embeddings()
        assert embeddings.shape == (CLUSTERS * ITEMS_PER_CLUSTER, 16)
        assert embeddings.dtype == np.float32

    def test_user_embeddings_cover_every_user(self, fitted: LightGCN) -> None:
        assert fitted.user_embeddings().shape == (USERS, 16)

    def test_metadata_records_what_is_needed_to_reproduce(self, fitted: LightGCN) -> None:
        metadata = fitted.metadata()
        assert metadata["model"] == "lightgcn"
        assert metadata["config"]["num_layers"] == 2
        assert metadata["graph_checksum"]
        assert metadata["loss_history"]
