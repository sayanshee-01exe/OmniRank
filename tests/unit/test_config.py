"""Configuration loading, environment overrides, and validation failures."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnirank.core.config import (
    ENV_PREFIX,
    AppConfig,
    DatabaseConfig,
    load_config,
)
from omnirank.core.exceptions import (
    ConfigFileNotFoundError,
    ConfigurationError,
    ConfigValidationError,
)


class TestLoading:
    def test_loads_the_real_repository_configuration(self, config: AppConfig):
        assert config.project_name == "omnirank"
        assert config.environment == "local"
        assert config.data.domain == "ecommerce"

    def test_include_directive_merges_every_overlay(self, config: AppConfig):
        # One key from each of the four overlays named in base.yaml's `include`.
        assert config.data.dataset_name == "ecommerce-subset"  # data/
        assert config.models.index.backend == "faiss"  # models/
        assert config.evaluation.k_values == (5, 10, 20, 50)  # evaluation/
        assert config.api.port == 8000  # serving/

    def test_include_key_is_not_part_of_the_schema(self, config: AppConfig):
        assert not hasattr(config, "include")

    def test_missing_base_file_names_the_path(self, tmp_path: Path):
        with pytest.raises(ConfigFileNotFoundError) as exc:
            load_config(tmp_path, env={})
        assert "base.yaml" in str(exc.value)

    def test_missing_overlay_names_the_path(self, tmp_path: Path):
        (tmp_path / "base.yaml").write_text("include: [nope.yaml]\n")
        with pytest.raises(ConfigFileNotFoundError) as exc:
            load_config(tmp_path, env={})
        assert "nope.yaml" in str(exc.value)

    def test_malformed_yaml_is_reported_as_a_config_error(self, tmp_path: Path):
        (tmp_path / "base.yaml").write_text("project_name: [unclosed\n")
        with pytest.raises(ConfigValidationError):
            load_config(tmp_path, env={})

    def test_non_mapping_yaml_is_rejected(self, tmp_path: Path):
        (tmp_path / "base.yaml").write_text("- just\n- a\n- list\n")
        with pytest.raises(ConfigValidationError) as exc:
            load_config(tmp_path, env={})
        assert "mapping" in str(exc.value)


class TestEnvironmentOverrides:
    def test_scalar_override(self, config_dir: Path):
        config = load_config(
            config_dir,
            env={f"{ENV_PREFIX}SEED": "1234"},
            dotenv_path=config_dir / "__none__",
        )
        assert config.seed == 1234

    def test_nested_override(self, config_dir: Path):
        config = load_config(
            config_dir,
            env={f"{ENV_PREFIX}LOGGING__LEVEL": "DEBUG"},
            dotenv_path=config_dir / "__none__",
        )
        assert config.logging.level == "DEBUG"

    def test_deeply_nested_override(self, config_dir: Path):
        config = load_config(
            config_dir,
            env={f"{ENV_PREFIX}DATA__SPLITTING__EMBARGO_DAYS": "7"},
            dotenv_path=config_dir / "__none__",
        )
        assert config.data.splitting.embargo_days == 7

    def test_boolean_values_are_coerced_not_left_as_strings(self, config_dir: Path):
        config = load_config(
            config_dir,
            env={f"{ENV_PREFIX}API__RELOAD": "false"},
            dotenv_path=config_dir / "__none__",
        )
        assert config.api.reload is False

    def test_empty_value_becomes_none(self, config_dir: Path):
        config = load_config(
            config_dir,
            env={f"{ENV_PREFIX}REDIS__PASSWORD": ""},
            dotenv_path=config_dir / "__none__",
        )
        assert config.redis.password is None

    def test_unprefixed_variables_are_ignored(self, config_dir: Path):
        config = load_config(
            config_dir,
            env={"SEED": "999", "PATH": "/usr/bin"},
            dotenv_path=config_dir / "__none__",
        )
        assert config.seed == 42

    def test_dotenv_is_read_and_real_env_wins_over_it(self, tmp_path: Path, config_dir: Path):
        dotenv = tmp_path / ".env"
        dotenv.write_text(
            f"{ENV_PREFIX}SEED=111\n{ENV_PREFIX}LOGGING__LEVEL=WARNING\n# a comment\n\n"
        )
        config = load_config(config_dir, env={f"{ENV_PREFIX}SEED": "222"}, dotenv_path=dotenv)
        assert config.seed == 222  # real env wins
        assert config.logging.level == "WARNING"  # dotenv-only key still applies

    def test_unknown_key_from_env_is_rejected_not_ignored(self, config_dir: Path):
        with pytest.raises(ConfigValidationError):
            load_config(
                config_dir,
                env={f"{ENV_PREFIX}TYPOED_SECTION__VALUE": "1"},
                dotenv_path=config_dir / "__none__",
            )


class TestValidation:
    def _write(self, tmp_path: Path, config_dir: Path, mutate) -> Path:
        """Copy the real config tree into tmp_path, applying `mutate` to base."""
        merged: dict = {}
        base = yaml.safe_load((config_dir / "base.yaml").read_text())
        includes = base.pop("include", [])
        merged.update(base)
        for overlay in includes:
            merged.update(yaml.safe_load((config_dir / overlay).read_text()))
        mutate(merged)
        (tmp_path / "base.yaml").write_text(yaml.safe_dump(merged))
        return tmp_path

    def test_unknown_top_level_key_is_rejected(self, tmp_path, config_dir):
        directory = self._write(tmp_path, config_dir, lambda c: c.update(nonsense=1))
        with pytest.raises(ConfigValidationError):
            load_config(directory, env={}, dotenv_path=tmp_path / "__none__")

    def test_split_fractions_must_leave_a_training_window(self, tmp_path, config_dir):
        def mutate(cfg):
            cfg["data"]["splitting"]["validation_fraction"] = 0.6
            cfg["data"]["splitting"]["test_fraction"] = 0.5

        directory = self._write(tmp_path, config_dir, mutate)
        with pytest.raises(ConfigValidationError) as exc:
            load_config(directory, env={}, dotenv_path=tmp_path / "__none__")
        assert "training window" in str(exc.value)

    def test_positive_threshold_above_every_weight_is_rejected(self, tmp_path, config_dir):
        directory = self._write(
            tmp_path, config_dir, lambda c: c["data"].update(positive_event_threshold=99.0)
        )
        with pytest.raises(ConfigValidationError) as exc:
            load_config(directory, env={}, dotenv_path=tmp_path / "__none__")
        assert "positive" in str(exc.value).lower()

    def test_fallback_chain_must_end_with_global_popularity(self, tmp_path, config_dir):
        directory = self._write(
            tmp_path, config_dir, lambda c: c["fallback"].update(chain=["category_popularity"])
        )
        with pytest.raises(ConfigValidationError) as exc:
            load_config(directory, env={}, dotenv_path=tmp_path / "__none__")
        assert "global_popularity" in str(exc.value)

    def test_empty_fallback_chain_is_rejected(self, tmp_path, config_dir):
        directory = self._write(tmp_path, config_dir, lambda c: c["fallback"].update(chain=[]))
        with pytest.raises(ConfigValidationError):
            load_config(directory, env={}, dotenv_path=tmp_path / "__none__")

    def test_source_weights_must_reference_declared_generators(self, tmp_path, config_dir):
        directory = self._write(
            tmp_path,
            config_dir,
            lambda c: c["models"]["aggregation"]["source_weights"].update(ghost_model=1.0),
        )
        with pytest.raises(ConfigValidationError) as exc:
            load_config(directory, env={}, dotenv_path=tmp_path / "__none__")
        assert "ghost_model" in str(exc.value)

    def test_default_top_k_may_not_exceed_max(self, tmp_path, config_dir):
        directory = self._write(
            tmp_path, config_dir, lambda c: c["serving"].update(default_top_k=500, max_top_k=100)
        )
        with pytest.raises(ConfigValidationError):
            load_config(directory, env={}, dotenv_path=tmp_path / "__none__")

    def test_production_requires_a_database_password(self, config_dir: Path):
        with pytest.raises(ConfigValidationError) as exc:
            load_config(
                config_dir,
                env={
                    f"{ENV_PREFIX}ENVIRONMENT": "production",
                    f"{ENV_PREFIX}API__RELOAD": "false",
                    f"{ENV_PREFIX}API__HOST": "0.0.0.0",
                    f"{ENV_PREFIX}LOGGING__FORMAT": "json",
                },
                dotenv_path=config_dir / "__none__",
            )
        assert "database.password" in str(exc.value)

    def test_production_rejects_reload_and_console_logging(self, config_dir: Path):
        with pytest.raises(ConfigValidationError) as exc:
            load_config(
                config_dir,
                env={
                    f"{ENV_PREFIX}ENVIRONMENT": "production",
                    f"{ENV_PREFIX}DATABASE__PASSWORD": "s3cret",
                },
                dotenv_path=config_dir / "__none__",
            )
        message = str(exc.value)
        assert "api.reload" in message
        assert "logging.format" in message


class TestSecrets:
    def test_password_is_not_rendered_in_dumps(self, config_dir: Path):
        config = load_config(
            config_dir,
            env={f"{ENV_PREFIX}DATABASE__PASSWORD": "supersecret"},
            dotenv_path=config_dir / "__none__",
        )
        assert "supersecret" not in str(config.model_dump(mode="json"))
        assert "supersecret" not in repr(config)

    def test_password_is_retrievable_where_needed(self, config_dir: Path):
        config = load_config(
            config_dir,
            env={f"{ENV_PREFIX}DATABASE__PASSWORD": "supersecret"},
            dotenv_path=config_dir / "__none__",
        )
        assert "supersecret" in config.database.dsn()

    def test_dsn_without_a_password_fails_with_a_named_setting(self):
        with pytest.raises(ConfigurationError) as exc:
            DatabaseConfig().dsn()
        assert "OMNIRANK__DATABASE__PASSWORD" in str(exc.value)

    def test_config_hash_ignores_the_password(self, config_dir: Path):
        def build(password: str) -> str:
            return load_config(
                config_dir,
                env={f"{ENV_PREFIX}DATABASE__PASSWORD": password},
                dotenv_path=config_dir / "__none__",
            ).config_hash

        assert build("one") == build("two")


class TestHashing:
    def test_hashes_are_stable_across_loads(self, config_dir: Path):
        def build() -> AppConfig:
            return load_config(config_dir, env={}, dotenv_path=config_dir / "__none__")

        assert build().training_config_hash == build().training_config_hash

    def test_training_hash_changes_with_a_training_relevant_setting(self, config_dir: Path):
        base = load_config(config_dir, env={}, dotenv_path=config_dir / "__none__")
        changed = load_config(
            config_dir,
            env={f"{ENV_PREFIX}SEED": "7"},
            dotenv_path=config_dir / "__none__",
        )
        assert base.training_config_hash != changed.training_config_hash

    def test_training_hash_ignores_serving_only_settings(self, config_dir: Path):
        """Changing an API port must not invalidate a trained model."""
        base = load_config(config_dir, env={}, dotenv_path=config_dir / "__none__")
        changed = load_config(
            config_dir,
            env={f"{ENV_PREFIX}API__PORT": "9999"},
            dotenv_path=config_dir / "__none__",
        )
        assert base.training_config_hash == changed.training_config_hash
        assert base.config_hash != changed.config_hash


class TestDerivedValues:
    def test_positive_event_types_respects_the_threshold(self, config: AppConfig):
        # threshold 2.0 excludes 'view' (1.0) and includes everything else.
        assert "view" not in config.data.positive_event_types
        assert "purchase" in config.data.positive_event_types

    def test_enabled_generators_is_empty_in_phase_1(self, config: AppConfig):
        """No model is implemented, so none may be switched on."""
        assert config.models.enabled_generators == ()

    def test_max_k_is_the_largest_cutoff(self, config: AppConfig):
        assert config.evaluation.max_k == 50

    def test_paths_resolve_against_a_root(self, config: AppConfig, tmp_path: Path):
        resolved = config.paths.resolved(tmp_path)
        assert resolved["processed_dir"] == tmp_path / "data/processed"
        assert resolved["metadata_dir"].is_absolute()

    def test_config_is_immutable(self, config: AppConfig):
        with pytest.raises(ValueError):
            # setattr, not `config.seed = 1`: the frozen model makes the field
            # read-only to the type checker, and this test asserts the *runtime*
            # behaviour rather than the static one.
            setattr(config, "seed", 1)  # noqa: B010
