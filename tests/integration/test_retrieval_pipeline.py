"""End-to-end Phase 4 retrieval workflow over a small deterministic fixture.

    fixture data -> fit LightGCN + SASRec -> blend them -> measure candidate
                 recall and source overlap -> build a FAISS index over the
                 embeddings -> verify the index against brute force
                 -> save -> load -> re-run -> compare identical

The point is that the *seams* hold: a model's embeddings feed an index, two
models feed an aggregator, and everything survives a persistence round trip.
Each component is unit-tested on its own; what this catches is the mismatch
between them -- a dimension that only lines up by accident, a catalogue that
narrows silently, an index built over the padding row.

Offline, CPU-only, seconds. No PixelRec download, no GPU, no database, no
MLflow server, no multimodal vectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="Retrieval pipeline requires the 'retrieval' extra")
pytest.importorskip("faiss", reason="Retrieval pipeline requires the 'retrieval' extra")

from omnirank.models.lightgcn import LightGCN, LightGCNConfig, LightGCNFitData  # noqa: E402
from omnirank.models.sasrec import SASRec, SASRecConfig, SASRecFitData  # noqa: E402
from omnirank.retrieval.aggregation import build_aggregator  # noqa: E402
from omnirank.retrieval.blended import BlendedRetriever  # noqa: E402
from omnirank.retrieval.diagnostics import candidate_recall, source_overlap  # noqa: E402
from omnirank.retrieval.faiss_index import (  # noqa: E402
    FaissVectorIndex,
    brute_force_top_k,
)

pytestmark = pytest.mark.integration

USERS = 40
ITEMS = 30
HISTORY = 8


@pytest.fixture
def interactions() -> pd.DataFrame:
    """Each user prefers one block of ten items, with an ordered history."""
    rng = np.random.default_rng(11)
    rows = []
    for user in range(USERS):
        block = user % 3
        for position in range(HISTORY):
            rows.append(
                {
                    "internal_user_id": user,
                    "internal_item_id": block * 10 + int(rng.integers(0, 10)),
                    "interaction_order": position,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def mappings() -> dict[str, dict]:
    return {
        "items": {index: f"i{index}" for index in range(ITEMS)},
        "users": {f"u{user}": user for user in range(USERS)},
    }


@pytest.fixture
def lightgcn(interactions: pd.DataFrame, mappings: dict[str, dict]) -> LightGCN:
    model = LightGCN(
        LightGCNConfig(
            embedding_dim=16,
            num_layers=2,
            max_epochs=15,
            batch_size=64,
            learning_rate=0.05,
            early_stopping_patience=20,
            seed=7,
        ),
        device="cpu",
    )
    model.fit(
        LightGCNFitData(
            edges=interactions[["internal_user_id", "internal_item_id"]].drop_duplicates(),
            num_users=USERS,
            num_items=ITEMS,
            internal_to_external_item=mappings["items"],
            external_to_internal_user=mappings["users"],
            mapping_checksum="fixture-checksum",
        )
    )
    return model


@pytest.fixture
def sasrec(interactions: pd.DataFrame, mappings: dict[str, dict]) -> SASRec:
    rows = []
    for user, group in interactions.groupby("internal_user_id"):
        items = group.sort_values("interaction_order")["internal_item_id"].tolist()
        for cut in range(2, len(items)):
            rows.append((int(user), items[:cut], items[cut]))
    model = SASRec(
        SASRecConfig(
            maximum_sequence_length=HISTORY,
            embedding_dim=16,
            num_blocks=1,
            num_heads=2,
            dropout=0.0,
            max_epochs=15,
            batch_size=64,
            early_stopping_patience=20,
            seed=7,
        ),
        device="cpu",
    )
    model.fit(
        SASRecFitData(
            sequences=pd.DataFrame(
                rows, columns=["internal_user_id", "item_sequence", "target_item"]
            ),
            num_users=USERS,
            num_items=ITEMS,
            internal_to_external_item=mappings["items"],
            external_to_internal_user=mappings["users"],
            mapping_checksum="fixture-checksum",
        )
    )
    return model


@pytest.fixture
def users() -> list[str]:
    return [f"u{user}" for user in range(USERS)]


class TestModelsAgreeOnTheCatalogue:
    """Both models were fitted on the same data through different paths."""

    def test_both_expose_the_same_mapping(self, lightgcn: LightGCN, sasrec: SASRec) -> None:
        lightgcn.require_mapping("fixture-checksum")
        sasrec.require_mapping("fixture-checksum")

    def test_embeddings_exclude_padding_and_match_the_catalogue(
        self, lightgcn: LightGCN, sasrec: SASRec
    ) -> None:
        """SASRec has an extra padding row internally; it must not be exported."""
        assert lightgcn.item_embeddings().shape[0] == ITEMS
        assert sasrec.item_embeddings().shape[0] == ITEMS


class TestBlending:
    @pytest.fixture
    def blend(self, lightgcn: LightGCN, sasrec: SASRec) -> BlendedRetriever:
        return BlendedRetriever(
            {"lightgcn": lightgcn, "sasrec": sasrec},
            build_aggregator("reciprocal_rank_fusion"),
            name="phase4_blend",
        )

    def test_blend_returns_a_full_list(self, blend: BlendedRetriever, users: list[str]) -> None:
        """Over-retrieval must fill k even when the two models agree."""
        recommended = blend.recommend_batch(users, 5)
        assert all(len(items) == 5 for items in recommended.values())

    def test_blend_catalogue_is_the_union(
        self, blend: BlendedRetriever, lightgcn: LightGCN, sasrec: SASRec
    ) -> None:
        assert blend.fit_item_catalogue == (lightgcn.fit_item_catalogue | sasrec.fit_item_catalogue)

    def test_blend_is_deterministic(self, blend: BlendedRetriever, users: list[str]) -> None:
        assert blend.recommend_batch(users, 5) == blend.recommend_batch(users, 5)

    def test_provenance_survives_the_whole_pipeline(self, blend: BlendedRetriever) -> None:
        sources = {source for c in blend.recommend("u0", 10) for source in c.sources}
        assert sources <= {"lightgcn", "sasrec"}
        assert sources


class TestDiagnostics:
    def test_candidate_recall_is_measurable_end_to_end(
        self, lightgcn: LightGCN, sasrec: SASRec, users: list[str]
    ) -> None:
        blend = BlendedRetriever(
            {"lightgcn": lightgcn, "sasrec": sasrec},
            build_aggregator("reciprocal_rank_fusion"),
        )
        pools = blend.recommend_batch(users, 10)
        # Each user's own block is what a correct model should retrieve.
        targets = {user: {f"i{(int(user[1:]) % 3) * 10 + n}" for n in range(10)} for user in users}
        result = candidate_recall(pools, targets, depth=10)
        assert result.users_evaluated == USERS
        assert result.recall > 0.5

    def test_source_overlap_is_measurable_end_to_end(
        self, lightgcn: LightGCN, sasrec: SASRec, users: list[str]
    ) -> None:
        overlap = source_overlap(
            {
                "lightgcn": lightgcn.recommend_batch(users, 10),
                "sasrec": sasrec.recommend_batch(users, 10),
            },
            depth=10,
        )
        assert 0.0 <= overlap.pairwise_jaccard["lightgcn|sasrec"] <= 1.0
        assert 1.0 <= overlap.mean_sources_per_item <= 2.0


class TestIndexOverModelEmbeddings:
    """The seam most likely to be silently wrong."""

    def test_index_reproduces_the_model_ranking(self, lightgcn: LightGCN) -> None:
        """An index over a model's embeddings must agree with the model itself."""
        embeddings = lightgcn.item_embeddings()
        index = FaissVectorIndex()
        index.build(embeddings)
        queries = lightgcn.user_embeddings()[:5]
        found, _ = index.search(queries, 5)
        expected, _ = brute_force_top_k(embeddings, queries, 5)
        assert found == expected.tolist()

    def test_index_over_sasrec_queries(self, sasrec: SASRec, users: list[str]) -> None:
        """SASRec's query vectors must line up with its own item embeddings."""
        embeddings = sasrec.item_embeddings()
        index = FaissVectorIndex()
        index.build(embeddings)
        queries = sasrec.query_embeddings(users[:5])
        assert queries.shape[1] == embeddings.shape[1]
        found, _ = index.search(queries, 5)
        assert all(len(row) == 5 for row in found)

    def test_index_refuses_the_wrong_model(self, lightgcn: LightGCN) -> None:
        index = FaissVectorIndex()
        index.build(lightgcn.item_embeddings())
        index.attach_metadata(
            model_name="lightgcn",
            model_version="v1",
            item_mapping_checksum="fixture-checksum",
            build_timestamp="2026-01-01T00:00:00Z",
        )
        from omnirank.core.exceptions import ArtifactValidationError

        with pytest.raises(ArtifactValidationError):
            index.require_compatible(
                model_name="sasrec",
                model_version="v1",
                item_mapping_checksum="fixture-checksum",
            )


class TestPersistenceRoundTrip:
    def test_models_and_index_survive_a_round_trip_identically(
        self, lightgcn: LightGCN, sasrec: SASRec, users: list[str], tmp_path
    ) -> None:
        before = BlendedRetriever(
            {"lightgcn": lightgcn, "sasrec": sasrec},
            build_aggregator("reciprocal_rank_fusion"),
        ).recommend_batch(users, 5)

        lightgcn.save(tmp_path / "lightgcn")
        sasrec.save(tmp_path / "sasrec")
        index = FaissVectorIndex()
        index.build(lightgcn.item_embeddings())
        index.save(tmp_path / "index")

        reloaded_lightgcn = LightGCN.load(tmp_path / "lightgcn", device="cpu")
        reloaded_sasrec = SASRec.load(tmp_path / "sasrec", device="cpu")
        after = BlendedRetriever(
            {"lightgcn": reloaded_lightgcn, "sasrec": reloaded_sasrec},
            build_aggregator("reciprocal_rank_fusion"),
        ).recommend_batch(users, 5)

        assert after == before

        reloaded_index = FaissVectorIndex.load(tmp_path / "index")
        queries = reloaded_lightgcn.user_embeddings()[:5]
        assert reloaded_index.search(queries, 5) == index.search(queries, 5)
