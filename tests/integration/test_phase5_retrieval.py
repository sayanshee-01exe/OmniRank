"""End-to-end Phase 5 retrieval over a deterministic synthetic fixture.

    features -> train -> catalogue -> embeddings -> exact FAISS
             -> retrieve -> five-source RRF -> evaluate warm and cold
             -> save/load -> verify identical

**The cold-item test is the mandatory one.** The fixture contains an item that
appears in no user's history and is no user's training target, so the
collaborative fixtures cannot return it at any depth. The two-tower catalogue
contains it, the index contains it, and after training it is retrieved. That
chain is Phase 5's entire justification, and every link is asserted rather than
inferred from an aggregate metric.

Offline, CPU-only, seconds. No PixelRec download, no pretrained weights, no GPU,
no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="Phase 5 requires the 'retrieval' extra")
pytest.importorskip("faiss", reason="Phase 5 requires the 'retrieval' extra")

from omnirank.features.multimodal_store import MultimodalFeatureStore  # noqa: E402
from omnirank.models.base import Candidate  # noqa: E402
from omnirank.models.two_tower import (  # noqa: E402
    MultimodalTwoTower,
    TwoTowerConfig,
    TwoTowerRetriever,
    TwoTowerTrainer,
    TwoTowerTrainingDataset,
)
from omnirank.retrieval.aggregation import build_aggregator  # noqa: E402
from omnirank.retrieval.two_tower_index import (  # noqa: E402
    build_two_tower_index,
    load_item_embeddings,
    verify_index_against_brute_force,
    write_item_embeddings,
)

pytestmark = pytest.mark.integration

ITEMS = 60
USERS = 30
TEXT_DIM = 12
IMAGE_DIM = 8
TAGS = 3
#: In no history and no training target. Collaborative models cannot reach it.
COLD_ITEM = ITEMS - 1


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, object]:
    """Clustered content plus block-preferring users, with one truly cold item."""
    rng = np.random.default_rng(11)
    features = tmp_path / "features"
    features.mkdir(parents=True)

    tags = np.arange(ITEMS) % TAGS
    text_centres = rng.normal(size=(TAGS, TEXT_DIM)).astype("float32") * 4.0
    image_centres = rng.normal(size=(TAGS, IMAGE_DIM)).astype("float32") * 4.0
    text = (text_centres[tags] + rng.normal(size=(ITEMS, TEXT_DIM)) * 0.2).astype("float32")
    image = (image_centres[tags] + rng.normal(size=(ITEMS, IMAGE_DIM)) * 0.2).astype("float32")
    np.save(features / "text_features.npy", text)
    np.save(features / "image_features.npy", image)

    pd.DataFrame(
        {
            "internal_item_id": range(ITEMS),
            "external_item_id": [f"i{index}" for index in range(ITEMS)],
            "has_text_feature": np.ones(ITEMS, dtype=bool),
            "has_image_feature": np.ones(ITEMS, dtype=bool),
        }
    ).to_parquet(features / "modality_mask.parquet", index=False)

    (features / "multimodal_feature_manifest.json").write_text(
        json.dumps(
            {
                "feature_version": "1",
                "item_mapping_checksum": "phase5-fixture",
                "catalogue_items": ITEMS,
                "modalities": {
                    "text": {
                        "available": True,
                        "matrix_file": "text_features.npy",
                        "dimension": TEXT_DIM,
                        "coverage": 1.0,
                        "rows_matched": ITEMS,
                    },
                    "image": {
                        "available": True,
                        "matrix_file": "image_features.npy",
                        "dimension": IMAGE_DIM,
                        "coverage": 1.0,
                        "rows_matched": ITEMS,
                    },
                },
            }
        )
    )

    rows = []
    for user in range(USERS):
        block = user % TAGS
        pool = [item for item in range(ITEMS - 1) if tags[item] == block]
        chain = rng.permutation(pool)[:5].tolist()
        for cut in range(2, 5):
            rows.append((user, chain[:cut], chain[cut]))
    sequences = pd.DataFrame(rows, columns=["internal_user_id", "item_sequence", "target_item"])
    return {
        "store": MultimodalFeatureStore(features),
        "sequences": sequences,
        "tags": tags,
        "root": tmp_path,
    }


@pytest.fixture
def retriever(workspace) -> TwoTowerRetriever:
    """A trained two-tower model wrapped in its retrieval surface."""
    dataset = TwoTowerTrainingDataset(
        workspace["sequences"],
        workspace["store"],
        num_items=ITEMS,
        num_users=USERS,
        maximum_history_length=4,
        item_tags=workspace["tags"],
        num_tags=TAGS,
    )
    config = TwoTowerConfig(
        embedding_dim=16,
        hidden_dims=(32,),
        dropout=0.0,
        maximum_history_length=4,
        batch_size=16,
        max_epochs=30,
        early_stopping_patience=30,
        learning_rate=0.01,
        temperature=0.2,
        # The locked Phase 5 finding: the identity residual zeroes cold
        # retrieval, so the selected configuration does not use it.
        use_item_id_residual=False,
        use_user_id_embedding=False,
        device="cpu",
        seed=7,
    )
    torch.manual_seed(0)
    network = MultimodalTwoTower(
        config,
        text_dim=TEXT_DIM,
        image_dim=IMAGE_DIM,
        num_items=ITEMS,
        num_users=USERS,
        num_tags=TAGS,
    )
    TwoTowerTrainer(network, config, device="cpu").fit(dataset, dataset)
    network.eval()

    histories: dict[int, list[int]] = {}
    for index in range(len(dataset)):
        example = dataset[index]
        combined = [*example["history_item_ids"].tolist(), example["positive_item_id"]]
        user = example["internal_user_id"]
        if user not in histories or len(combined) > len(histories[user]):
            histories[user] = combined

    return TwoTowerRetriever(
        network,
        workspace["store"],
        internal_to_external_item={index: f"i{index}" for index in range(ITEMS)},
        external_to_internal_user={f"u{user}": user for user in range(USERS)},
        item_tags=workspace["tags"],
        histories=histories,
        warm_items=dataset.warm_mask,
        device="cpu",
        mapping_checksum="phase5-fixture",
    )


class TestColdItemIsUnreachableCollaboratively:
    """Establishes the premise the mandatory test depends on."""

    def test_the_cold_item_has_no_interactions(self, workspace) -> None:
        sequences = workspace["sequences"]
        in_history = any(COLD_ITEM in list(row) for row in sequences["item_sequence"])
        assert not in_history
        assert COLD_ITEM not in set(sequences["target_item"])

    def test_a_collaborative_catalogue_would_exclude_it(self, workspace) -> None:
        """What BPR or LightGCN would be able to return."""
        sequences = workspace["sequences"]
        observed = set(sequences["target_item"])
        for row in sequences["item_sequence"]:
            observed.update(list(row))
        assert COLD_ITEM not in observed


class TestColdItemRetrieval:
    """The mandatory Phase 5 test."""

    def test_the_catalogue_contains_the_cold_item(self, retriever) -> None:
        assert COLD_ITEM in retriever.cold_item_catalogue

    def test_the_index_contains_the_cold_item(self, retriever) -> None:
        embeddings = retriever.export_item_embeddings()
        _, metadata = build_two_tower_index(
            embeddings,
            retriever.catalogue,
            model_version="fixture",
            model_checksum="c",
            mapping_checksum="phase5-fixture",
            feature_version="1",
            feature_manifest_checksum="f",
            normalization="l2",
        )
        assert metadata["cold_item_count"] >= 1

    def test_the_cold_item_is_retrieved_for_a_matching_user(self, retriever, workspace) -> None:
        """The whole justification: content reaches what collaboration cannot."""
        tags = workspace["tags"]
        # Users in the cold item's own tag block should surface it.
        block = int(tags[COLD_ITEM])
        candidates = [f"u{user}" for user in range(USERS) if user % TAGS == block]
        recommended = retriever.recommend_batch(candidates, 20)
        retrieved = sum(f"i{COLD_ITEM}" in items for items in recommended.values())
        assert retrieved > 0, "the cold item was never retrieved by any matching user"

    def test_the_cold_item_is_scoreable(self, retriever) -> None:
        assert retriever.score("u0", [f"i{COLD_ITEM}"]) != [0.0]


class TestIndexExactness:
    def test_faiss_matches_brute_force(self, retriever) -> None:
        embeddings = retriever.export_item_embeddings()
        index, _ = build_two_tower_index(
            embeddings,
            retriever.catalogue,
            model_version="fixture",
            model_checksum="c",
            mapping_checksum="phase5-fixture",
            feature_version="1",
            feature_manifest_checksum="f",
            normalization="l2",
        )
        queries = retriever.encode_users([f"u{user}" for user in range(10)])
        report = verify_index_against_brute_force(index, embeddings, queries, k=10)
        assert report["matches_brute_force"]
        assert report["max_score_difference"] < 1e-4

    def test_embeddings_round_trip(self, retriever, tmp_path) -> None:
        embeddings = retriever.export_item_embeddings()
        write_item_embeddings(
            tmp_path / "emb",
            embeddings,
            retriever.catalogue,
            model_version="fixture",
            model_checksum="c",
            mapping_checksum="phase5-fixture",
            feature_version="1",
            feature_manifest_checksum="f",
            normalization="l2",
        )
        loaded, catalogue, manifest = load_item_embeddings(tmp_path / "emb")
        assert np.array_equal(loaded, embeddings)
        assert catalogue.checksum() == retriever.catalogue.checksum()
        assert manifest["cold_items"] == retriever.catalogue.cold_count


class TestFiveSourceFusion:
    """The two-tower alongside four collaborative fixtures."""

    @staticmethod
    def _stub(source: str, items: list[str]) -> list[Candidate]:
        return [
            Candidate(
                item_id=item,
                score=float(len(items) - position),
                sources=(source,),
                source_scores={source: float(len(items) - position)},
            )
            for position, item in enumerate(items)
        ]

    @pytest.fixture
    def sources(self, retriever):
        """Four collaborative stubs that cannot see the cold item, plus the model."""
        warm = [f"i{item}" for item in sorted(retriever.catalogue.internal_ids)[:12]]
        return {
            "popularity": self._stub("popularity", warm[:6]),
            "matrix_factorization": self._stub("matrix_factorization", warm[2:8]),
            "lightgcn": self._stub("lightgcn", warm[4:10]),
            "sasrec": self._stub("sasrec", warm[6:12]),
            "two_tower": retriever.recommend("u0", 6),
        }

    def test_five_source_fusion_preserves_provenance(self, sources) -> None:
        result = build_aggregator("reciprocal_rank_fusion").aggregate(sources, limit=20)
        seen = {source for candidate in result.candidates for source in candidate.sources}
        assert "two_tower" in seen
        assert len(seen) > 1

    def test_two_tower_contributes_unique_candidates(self, sources) -> None:
        """Its value may be reach rather than rank, so the unique set is measured."""
        others = {
            candidate.item_id
            for source, candidates in sources.items()
            if source != "two_tower"
            for candidate in candidates
        }
        unique = {candidate.item_id for candidate in sources["two_tower"]} - others
        assert unique, "the two-tower proposed nothing the other four did not"

    def test_fusion_is_deterministic(self, sources) -> None:
        aggregator = build_aggregator("reciprocal_rank_fusion")
        first = [c.item_id for c in aggregator.aggregate(sources, limit=10).candidates]
        second = [c.item_id for c in aggregator.aggregate(sources, limit=10).candidates]
        assert first == second

    def test_a_missing_source_is_reported_not_fatal(self, sources) -> None:
        degraded = {**sources, "sasrec": []}
        result = build_aggregator("reciprocal_rank_fusion").aggregate(degraded, limit=10)
        assert result.degraded_sources == ("sasrec",)
        assert result.candidates

    def test_five_sources_reach_more_than_four(self, sources) -> None:
        """The comparison the phase exists to make."""
        aggregator = build_aggregator("reciprocal_rank_fusion")
        four = {k: v for k, v in sources.items() if k != "two_tower"}
        pool_four = {c.item_id for c in aggregator.aggregate(four, limit=50).candidates}
        pool_five = {c.item_id for c in aggregator.aggregate(sources, limit=50).candidates}
        assert pool_five > pool_four


class TestPersistenceRoundTrip:
    def test_recommendations_survive_save_and_load(self, retriever, workspace, tmp_path) -> None:
        users = [f"u{user}" for user in range(USERS)]
        before = retriever.recommend_batch(users, 10)
        retriever.save(tmp_path / "retriever")
        loaded = TwoTowerRetriever.load(
            tmp_path / "retriever", store=workspace["store"], device="cpu"
        )
        assert loaded.recommend_batch(users, 10) == before

    def test_the_cold_item_stays_retrievable_after_reload(
        self, retriever, workspace, tmp_path
    ) -> None:
        """A cold-start guarantee that dies on reload is not a guarantee."""
        retriever.save(tmp_path / "retriever")
        loaded = TwoTowerRetriever.load(
            tmp_path / "retriever", store=workspace["store"], device="cpu"
        )
        assert COLD_ITEM in loaded.cold_item_catalogue
        assert loaded.score("u0", [f"i{COLD_ITEM}"]) != [0.0]
