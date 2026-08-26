"""Generated Phase 5 configurations must match the records they derive from.

Both files are derived, not authored. The failure they guard against is quiet:
somebody edits `phase5_selected.yaml` by hand, and the tracked configuration
stops describing the run that justified it while still looking authoritative.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_generate_phase5_configs", PROJECT_ROOT / "scripts" / "generate_phase5_configs.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()

SELECTED = PROJECT_ROOT / "configs/models/phase5_selected.yaml"
FEATURES = PROJECT_ROOT / "configs/features/pixelrec_published.yaml"
SELECTION_RECORD = PROJECT_ROOT / "reports/metrics/phase_05/selected_configuration.json"
FEATURE_MANIFEST = (
    PROJECT_ROOT / "data/processed/pixelrec50k/features/multimodal_feature_manifest.json"
)

needs_selection = pytest.mark.skipif(
    not SELECTION_RECORD.is_file(), reason="no locked selection record in this checkout"
)
needs_manifest = pytest.mark.skipif(
    not FEATURE_MANIFEST.is_file(), reason="no aligned feature manifest in this checkout"
)


class TestSelectedConfigProvenance:
    @needs_selection
    def test_the_tracked_file_matches_the_selection_record(self) -> None:
        """Regenerating must be a no-op. If it is not, the file was edited."""
        record = json.loads(SELECTION_RECORD.read_text())
        assert SELECTED.read_text() == GENERATOR.render_selected_config(record)

    @needs_selection
    def test_it_records_every_field_the_closure_spec_requires(self) -> None:
        block = yaml.safe_load(SELECTED.read_text())["models"]["candidate_generators"]["two_tower"]
        required = (
            "embedding_dim",
            "text_projection_dim",
            "image_projection_dim",
            "modality_fusion",
            "history_pooling",
            "temperature",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "max_epochs",
            "early_stopping_patience",
            "seed",
            "use_user_id_embedding",
            "use_item_id_residual",
            "l2_normalize",
        )
        missing = [name for name in required if name not in block]
        assert not missing, f"selected config omits: {missing}"

    @needs_selection
    def test_it_carries_a_configuration_hash_and_selection_reference(self) -> None:
        parsed = yaml.safe_load(SELECTED.read_text())
        selection = parsed["selection"]
        assert selection["configuration_hash"]
        assert selection["selected_by"]
        assert selection["fit_splits"]
        assert selection["target_split"]

    def test_the_configuration_hash_ignores_recorded_metrics(self) -> None:
        """A hash that moved when a metric was appended would be useless."""
        block = {"embedding_dim": 128, "temperature": 0.07, "seed": 42}
        first = GENERATOR.configuration_hash(block)
        second = GENERATOR.configuration_hash({**block, "ndcg@20": 0.5, "run_id": "abc"})
        assert first == second

    def test_the_configuration_hash_moves_when_a_hyperparameter_moves(self) -> None:
        base = {"embedding_dim": 128, "temperature": 0.07}
        assert GENERATOR.configuration_hash(base) != GENERATOR.configuration_hash(
            {**base, "embedding_dim": 256}
        )


class TestFeatureConfigProvenance:
    @needs_manifest
    def test_the_tracked_file_matches_the_feature_manifest(self) -> None:
        manifest = json.loads(FEATURE_MANIFEST.read_text())
        dataset_path = PROJECT_ROOT / "data/processed/pixelrec50k/dataset_manifest.json"
        dataset = json.loads(dataset_path.read_text()) if dataset_path.is_file() else {}
        assert FEATURES.read_text() == GENERATOR.render_feature_config(manifest, dataset)

    @needs_manifest
    def test_dimensions_and_coverage_come_from_the_manifest(self) -> None:
        manifest = json.loads(FEATURE_MANIFEST.read_text())
        parsed = yaml.safe_load(FEATURES.read_text())["features"]
        for modality in ("text", "image"):
            recorded = manifest["modalities"][modality]
            assert parsed[modality]["dimension"] == recorded["dimension"]
            assert parsed[modality]["items_matched"] == recorded["rows_matched"]

    @needs_manifest
    def test_the_encoder_is_not_given_a_name_the_source_does_not_document(self) -> None:
        """PixelRec ships these vectors without saying what produced them.

        Writing `clip` or `bert` here would be a fabricated provenance claim
        that every downstream comparison would silently inherit.
        """
        parsed = yaml.safe_load(FEATURES.read_text())["features"]
        for modality in ("text", "image"):
            assert parsed[modality]["encoder_identity"] == "unknown"
        lowered = FEATURES.read_text().lower()
        for name in ("clip", "bert", "sentencetransformer", "sentence-transformer", "resnet"):
            assert f"encoder_identity: {name}" not in lowered

    @needs_manifest
    def test_it_carries_all_three_identity_checksums(self) -> None:
        compatibility = yaml.safe_load(FEATURES.read_text())["features"]["compatibility"]
        for key in (
            "feature_manifest_checksum",
            "item_mapping_checksum",
            "dataset_manifest_checksum",
        ):
            assert compatibility.get(key), f"{key} is empty"

    @needs_manifest
    def test_normalization_is_stated_rather_than_assumed(self) -> None:
        parsed = yaml.safe_load(FEATURES.read_text())["features"]
        assert parsed["normalization"]["input_vectors_normalized"] is False


class TestDriftDetection:
    @needs_selection
    @needs_manifest
    def test_check_mode_passes_on_the_committed_files(self) -> None:
        assert GENERATOR.main(["--check"]) == 0

    @needs_selection
    def test_check_mode_fails_when_a_file_is_edited(self, tmp_path, monkeypatch) -> None:
        """The whole point of --check: hand edits must not survive review."""
        copy = tmp_path / "phase5_selected.yaml"
        copy.write_text(SELECTED.read_text() + "\n# hand-edited\n")
        monkeypatch.setattr(GENERATOR, "SELECTED_CONFIG", copy)
        assert GENERATOR.main(["--check"]) == GENERATOR.DRIFT_EXIT


class TestGeneratedYamlIsValid:
    """The generated files must parse. This is not hypothetical.

    `selected_by` recorded a two-stage selection whose description contained
    ": ", which YAML reads as a nested mapping. The tracked config was
    unparseable and nothing noticed, because every consumer read the *record*
    it was generated from rather than the config itself.
    """

    @pytest.mark.parametrize("path", [SELECTED, FEATURES])
    def test_the_tracked_file_parses(self, path: Path) -> None:
        if not path.is_file():
            pytest.skip(f"{path.name} not generated in this checkout")
        assert isinstance(yaml.safe_load(path.read_text()), dict)

    @pytest.mark.parametrize(
        "value",
        [
            "two stage: ablation screen, then folds",
            "trailing colon:",
            "# looks like a comment",
            "- looks like a list item",
            "true",
            "null",
            "  padded  ",
            'has "quotes" inside',
        ],
    )
    def test_awkward_strings_survive_a_round_trip(self, value: str) -> None:
        rendered = f"key: {GENERATOR._scalar(value)}"
        assert yaml.safe_load(rendered)["key"] == value

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(True, True), (False, False), (None, None), (128, 128), (0.07, 0.07)],
    )
    def test_non_strings_keep_their_type(self, value: object, expected: object) -> None:
        assert yaml.safe_load(f"key: {GENERATOR._scalar(value)}")["key"] == expected

    def test_a_plain_word_is_not_needlessly_quoted(self) -> None:
        """Quoting everything would work but would make the files unreadable."""
        assert GENERATOR._scalar("mean_pooling") == "mean_pooling"
