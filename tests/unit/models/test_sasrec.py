"""SASRec sequential retrieval.

Causality is the property worth most of this file. If a position could attend
to the future, SASRec would learn to copy the answer out of its own input: the
training loss would fall faster, offline metrics would improve, and every one
of them would be meaningless. Nothing in a loss curve reveals that, so it is
asserted directly -- perturb a later position and require that earlier hidden
states do not move at all.

Padding is the second. It is ``num_items``, one past the last valid id, and it
must never be attended to, never contribute to the loss, and never be
recommended.

Skipped wholesale when torch is absent, so the core suite still runs without
the ``retrieval`` extra installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="SASRec requires the 'retrieval' extra")

from omnirank.core.exceptions import (  # noqa: E402
    ArtifactValidationError,
    DataError,
    ModelNotFittedError,
)
from omnirank.models.sasrec import (  # noqa: E402
    SASRec,
    SASRecConfig,
    SASRecFitData,
    SASRecNetwork,
    encode_sequences,
)

NUM_ITEMS = 20
USERS = 120
CHAIN = 6


@pytest.fixture
def successor_data() -> SASRecFitData:
    """Item *i* is always followed by item *i+1* (mod catalogue).

    Order is the only signal: every user's item *set* is an arbitrary window of
    the cycle, so a model that ignores sequence cannot do better than chance.
    That makes "did it learn the ordering?" a real assertion rather than a
    restatement of popularity.
    """
    rng = np.random.default_rng(0)
    rows = []
    for user in range(USERS):
        start = int(rng.integers(0, NUM_ITEMS))
        chain = [(start + step) % NUM_ITEMS for step in range(CHAIN)]
        for cut in range(2, CHAIN):
            rows.append((user, chain[:cut], chain[cut]))
    frame = pd.DataFrame(rows, columns=["internal_user_id", "item_sequence", "target_item"])
    return SASRecFitData(
        sequences=frame,
        num_users=USERS,
        num_items=NUM_ITEMS,
        internal_to_external_item={index: f"i{index}" for index in range(NUM_ITEMS)},
        external_to_internal_user={f"u{user}": user for user in range(USERS)},
        mapping_checksum="checksum-xyz",
    )


@pytest.fixture
def fitted(successor_data: SASRecFitData) -> SASRec:
    """A small model trained to convergence on the successor pattern."""
    model = SASRec(
        SASRecConfig(
            maximum_sequence_length=CHAIN,
            embedding_dim=32,
            num_blocks=2,
            num_heads=2,
            dropout=0.0,
            learning_rate=0.01,
            batch_size=64,
            max_epochs=60,
            early_stopping_patience=60,
            negatives_per_positive=4,
            seed=7,
        ),
        device="cpu",
    )
    model.fit(successor_data)
    return model


@pytest.fixture
def network() -> SASRecNetwork:
    torch.manual_seed(0)
    model = SASRecNetwork(
        SASRecConfig(maximum_sequence_length=6, embedding_dim=16, num_heads=2), NUM_ITEMS
    )
    model.eval()
    return model


class TestSequenceEncoding:
    def test_sequences_are_left_padded_and_right_aligned(self) -> None:
        """Right-aligned so the newest item always sits in the final column."""
        encoded = encode_sequences([[1, 2, 3]], maximum_length=5, padding_id=99)
        assert encoded[0].tolist() == [99, 99, 1, 2, 3]

    def test_truncation_keeps_the_newest_items(self) -> None:
        """Dropping recent history would defeat the point of a sequential model."""
        encoded = encode_sequences([[1, 2, 3, 4, 5]], maximum_length=3, padding_id=99)
        assert encoded[0].tolist() == [3, 4, 5]

    def test_empty_sequence_is_all_padding(self) -> None:
        encoded = encode_sequences([[]], maximum_length=3, padding_id=99)
        assert encoded[0].tolist() == [99, 99, 99]

    def test_output_is_int64_for_embedding_lookup(self) -> None:
        assert encode_sequences([[1]], maximum_length=2, padding_id=9).dtype == np.int64


class TestCausality:
    """A position must never see the future. This is the correctness property."""

    def test_a_later_item_cannot_change_an_earlier_hidden_state(
        self, network: SASRecNetwork
    ) -> None:
        base = torch.tensor([[1, 2, 3, 4, 5, 6]])
        changed = torch.tensor([[1, 2, 3, 4, 5, 17]])  # only the final position differs
        with torch.no_grad():
            first, second = network(base), network(changed)
        assert torch.equal(first[:, :-1, :], second[:, :-1, :])

    def test_the_changed_position_itself_does_move(self, network: SASRecNetwork) -> None:
        """Guards against the previous test passing because nothing propagates."""
        base = torch.tensor([[1, 2, 3, 4, 5, 6]])
        changed = torch.tensor([[1, 2, 3, 4, 5, 17]])
        with torch.no_grad():
            first, second = network(base), network(changed)
        assert (first[:, -1, :] - second[:, -1, :]).abs().max() > 1e-6

    def test_a_mid_sequence_change_affects_only_that_point_onwards(
        self, network: SASRecNetwork
    ) -> None:
        base = torch.tensor([[1, 2, 3, 4, 5, 6]])
        changed = torch.tensor([[1, 2, 3, 9, 5, 6]])  # position 3 differs
        with torch.no_grad():
            first, second = network(base), network(changed)
        assert torch.equal(first[:, :3, :], second[:, :3, :])
        assert (first[:, 3:, :] - second[:, 3:, :]).abs().max() > 1e-6

    def test_mask_blocks_exactly_the_strict_upper_triangle(self, network: SASRecNetwork) -> None:
        mask = network.causal_mask(4, torch.device("cpu"))
        assert mask.dtype == torch.bool
        assert mask.sum() == 6  # the strict upper triangle of a 4x4 is blocked
        assert not mask[3, 0]  # the last position may attend to the first
        assert mask[0, 3]  # but not the reverse


class TestPadding:
    def test_padding_id_is_one_past_the_last_real_item(self, network: SASRecNetwork) -> None:
        """Reusing 0 would collide with a real item under the Phase 2 mappings."""
        assert network.padding_id == NUM_ITEMS
        assert network.item_embedding.num_embeddings == NUM_ITEMS + 1

    def test_padding_embedding_stays_zero(self, network: SASRecNetwork) -> None:
        assert network.item_embedding.weight[network.padding_id].abs().sum() == 0.0

    def test_padding_is_never_recommended(self, fitted: SASRec) -> None:
        recommended = fitted.recommend_batch([f"u{u}" for u in range(USERS)], 5)
        assert not any(f"i{NUM_ITEMS}" in items for items in recommended.values())


class TestConfig:
    def test_embedding_dim_must_divide_across_heads(self) -> None:
        with pytest.raises(DataError):
            SASRecConfig(embedding_dim=10, num_heads=4)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("maximum_sequence_length", 0),
            ("num_blocks", 0),
            ("num_heads", 0),
            ("dropout", 1.0),
            ("learning_rate", 0.0),
            ("batch_size", 0),
            ("max_epochs", 0),
        ],
    )
    def test_rejects_invalid_values(self, field: str, value: float) -> None:
        with pytest.raises(DataError):
            SASRecConfig(**{field: value})


class TestFitting:
    def test_loss_decreases(self, fitted: SASRec) -> None:
        history = fitted.loss_history
        assert len(history) > 1
        assert history[-1] < history[0]

    def test_learns_the_successor_pattern(self, fitted: SASRec) -> None:
        """The true next item should land in the top 3 for most users."""
        hits = 0
        for user in range(USERS):
            recommended = fitted.recommend_batch([f"u{user}"], 3)[f"u{user}"]
            last = fitted._user_histories[user][-1]
            if f"i{(last + 1) % NUM_ITEMS}" in recommended:
                hits += 1
        assert hits / USERS > 0.5

    def test_refuses_a_non_sasrec_bundle(self) -> None:
        with pytest.raises(DataError):
            SASRec().fit({"sequences": []})

    def test_refuses_empty_sequences(self, successor_data: SASRecFitData) -> None:
        empty = SASRecFitData(
            sequences=successor_data.sequences.iloc[:0],
            num_users=1,
            num_items=1,
            internal_to_external_item={},
            external_to_internal_user={},
        )
        with pytest.raises(DataError):
            SASRec().fit(empty)

    def test_missing_columns_are_reported(self) -> None:
        with pytest.raises(DataError):
            SASRecFitData(
                sequences=pd.DataFrame({"internal_user_id": [0]}),
                num_users=1,
                num_items=1,
                internal_to_external_item={},
                external_to_internal_user={},
            )

    def test_is_deterministic_under_a_fixed_seed(self, successor_data: SASRecFitData) -> None:
        def train() -> list[str]:
            model = SASRec(
                SASRecConfig(
                    maximum_sequence_length=CHAIN,
                    embedding_dim=16,
                    num_blocks=1,
                    num_heads=2,
                    dropout=0.0,
                    max_epochs=5,
                    seed=3,
                ),
                device="cpu",
            )
            model.fit(successor_data)
            return model.recommend_batch(["u0"], 5)["u0"]

        assert train() == train()


class TestRecommendation:
    def test_unfitted_model_refuses_to_recommend(self) -> None:
        with pytest.raises(ModelNotFittedError):
            SASRec().recommend("u0", 5)

    def test_unknown_user_returns_nothing(self, fitted: SASRec) -> None:
        """There is no sequence to encode, and inventing one produces confident noise."""
        assert fitted.recommend("stranger", 5) == []

    def test_unknown_user_scores_zero(self, fitted: SASRec) -> None:
        assert fitted.score("stranger", ["i1", "i2"]) == [0.0, 0.0]

    def test_unknown_item_scores_zero(self, fitted: SASRec) -> None:
        assert fitted.score("u0", ["not-an-item"]) == [0.0]

    def test_seen_items_are_filtered_by_default(self, fitted: SASRec) -> None:
        recommended = fitted.recommend_batch([f"u{u}" for u in range(USERS)], 5)
        for user, items in recommended.items():
            seen = fitted._seen_by_user[int(user[1:])]
            assert not {int(item[1:]) for item in items} & seen

    def test_batching_matches_one_at_a_time(self, fitted: SASRec) -> None:
        users = [f"u{u}" for u in range(USERS)]
        assert fitted.recommend_batch(users, 5) == {
            user: [c.item_id for c in fitted.recommend(user, 5)] for user in users
        }

    def test_k_beyond_the_catalogue_is_padded_not_crashed(self, fitted: SASRec) -> None:
        assert len(fitted.recommend("u0", 500)) <= NUM_ITEMS

    def test_candidates_carry_provenance(self, fitted: SASRec) -> None:
        candidate = fitted.recommend("u0", 1)[0]
        assert candidate.sources == ("sasrec",)
        assert candidate.source_scores["sasrec"] == candidate.score

    def test_rejects_non_positive_k(self, fitted: SASRec) -> None:
        with pytest.raises(DataError):
            fitted.recommend("u0", 0)


class TestPersistence:
    def test_round_trip_preserves_recommendations(self, fitted: SASRec, tmp_path) -> None:
        fitted.save(tmp_path / "model")
        loaded = SASRec.load(tmp_path / "model", device="cpu")
        users = [f"u{u}" for u in range(USERS)]
        assert loaded.recommend_batch(users, 5) == fitted.recommend_batch(users, 5)

    def test_round_trip_preserves_histories(self, fitted: SASRec, tmp_path) -> None:
        """Inference needs the sequence, not just the weights."""
        fitted.save(tmp_path / "model")
        loaded = SASRec.load(tmp_path / "model", device="cpu")
        assert loaded.history_length("u0") == fitted.history_length("u0")

    def test_round_trip_preserves_scores_exactly(self, fitted: SASRec, tmp_path) -> None:
        fitted.save(tmp_path / "model")
        loaded = SASRec.load(tmp_path / "model", device="cpu")
        items = [f"i{i}" for i in range(NUM_ITEMS)]
        assert loaded.score("u0", items) == fitted.score("u0", items)

    def test_missing_files_are_reported_not_crashed(self, tmp_path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(ArtifactValidationError):
            SASRec.load(tmp_path / "empty")

    def test_rejects_an_artifact_from_another_model(self, fitted: SASRec, tmp_path) -> None:
        fitted.save(tmp_path / "model")
        config = tmp_path / "model" / "config.json"
        config.write_text(config.read_text().replace('"sasrec"', '"something-else"', 1))
        with pytest.raises(ArtifactValidationError):
            SASRec.load(tmp_path / "model")

    def test_mapping_checksum_mismatch_is_refused(self, fitted: SASRec) -> None:
        fitted.require_mapping("checksum-xyz")
        with pytest.raises(ArtifactValidationError):
            fitted.require_mapping("a-different-checksum")


class TestExport:
    def test_item_embeddings_exclude_padding(self, fitted: SASRec) -> None:
        """An index built over the padding row could return it as a neighbour."""
        embeddings = fitted.item_embeddings()
        assert embeddings.shape == (NUM_ITEMS, 32)
        assert embeddings.dtype == np.float32

    def test_query_embeddings_match_the_index_dimension(self, fitted: SASRec) -> None:
        queries = fitted.query_embeddings(["u0", "u1"])
        assert queries.shape == (2, 32)
        assert queries.dtype == np.float32

    def test_query_embeddings_skip_unknown_users(self, fitted: SASRec) -> None:
        assert fitted.query_embeddings(["stranger"]).shape == (0, 32)

    def test_metadata_records_what_is_needed_to_reproduce(self, fitted: SASRec) -> None:
        metadata = fitted.metadata()
        assert metadata["model"] == "sasrec"
        assert metadata["padding_id"] == NUM_ITEMS
        assert metadata["config"]["num_blocks"] == 2
        assert metadata["loss_history"]
