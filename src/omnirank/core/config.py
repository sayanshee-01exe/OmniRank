"""Centralised, validated configuration.

Resolution order, later winning over earlier:

1. ``configs/base.yaml``
2. every overlay named in that file's ``include:`` list, in order
3. overlays passed explicitly to :func:`load_config`
4. a ``.env`` file in the project root, if present
5. real environment variables

Environment keys use a double-underscore path under the ``OMNIRANK__`` prefix::

    OMNIRANK__LOGGING__LEVEL=DEBUG          -> logging.level
    OMNIRANK__DATA__VALIDATION__MIN_PRICE=1 -> data.validation.min_price

Every model forbids unknown keys, so a typo in YAML or an env var is a startup
failure with the offending path named, not a silently ignored setting.
Secrets are accepted only from the environment: no field that holds one has a
default in YAML.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from omnirank.core.device import DeviceType, resolve_device
from omnirank.core.exceptions import (
    ConfigFileNotFoundError,
    ConfigurationError,
    ConfigValidationError,
)

ENV_PREFIX = "OMNIRANK__"
ENV_DELIMITER = "__"
DEFAULT_CONFIG_DIR = Path("configs")
DEFAULT_BASE_FILE = "base.yaml"


# --------------------------------------------------------------------------- #
# Section models
# --------------------------------------------------------------------------- #
class _Section(BaseModel):
    """Shared behaviour: reject unknown keys, forbid mutation after load."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DeviceConfig(_Section):
    """Compute target selection. See :mod:`omnirank.core.device`."""

    preferred: DeviceType = DeviceType.AUTO
    allow_cuda: bool = False

    def resolve(self) -> str:
        """Return the concrete device string for this host."""
        return resolve_device(self.preferred, allow_cuda=self.allow_cuda)


class LoggingConfig(_Section):
    """Structured logging setup. See :mod:`omnirank.core.logging`."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["console", "json"] = "console"
    include_timestamp: bool = True
    redact_keys: tuple[str, ...] = ("password", "token", "secret", "authorization", "api_key")


class PathsConfig(_Section):
    """Filesystem layout. All paths are resolved against the project root."""

    data_root: Path = Path("data")
    artifact_root: Path = Path("artifacts")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    external_dir: Path = Path("data/external")
    mappings_dir: Path = Path("artifacts/mappings")
    models_dir: Path = Path("artifacts/models")
    embeddings_dir: Path = Path("artifacts/embeddings")
    indexes_dir: Path = Path("artifacts/indexes")
    metadata_dir: Path = Path("artifacts/metadata")

    def resolved(self, root: Path) -> dict[str, Path]:
        """Return every path made absolute against ``root``."""
        return {
            name: (root / value) if not value.is_absolute() else value
            for name, value in self.model_dump().items()
        }


# --- data ------------------------------------------------------------------ #
class ValidationRulesConfig(_Section):
    """Bounds enforced by :mod:`omnirank.data.validation`."""

    min_price: float = 0.0
    max_price: float = 1_000_000.0
    min_rating: float = 1.0
    max_rating: float = 5.0
    min_timestamp: datetime
    allow_future_timestamps: bool = False
    drop_duplicate_events: bool = True

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.min_price > self.max_price:
            raise ValueError("data.validation.min_price must not exceed max_price")
        if self.min_rating > self.max_rating:
            raise ValueError("data.validation.min_rating must not exceed max_rating")
        return self


class FilteringConfig(_Section):
    """k-core filtering applied once, before splitting.

    Applied to the whole log rather than per split: filtering the three splits
    separately would give them different item vocabularies, and every comparison
    between them would silently compare different datasets.
    """

    enabled: bool = True
    min_interactions_per_user: int = Field(default=5, ge=0)
    min_interactions_per_item: int = Field(default=5, ge=0)
    # Removing sparse users can push items below their threshold and vice versa,
    # so a single pass leaves the invariant unsatisfied. Kept configurable so a
    # run can measure the difference rather than assume it.
    iterative: bool = True


#: Strategies that cut the log at global instants, parameterised by fractions.
FRACTION_STRATEGIES = frozenset({"temporal_global", "temporal_per_user"})
#: Strategies that hold out a fixed number of trailing events per user.
LEAVE_LAST_N_STRATEGIES = frozenset({"leave_one_out", "per_user_leave_last_n"})


class SplittingConfig(_Section):
    """Split protocol. See ADR-002 and docs/data/temporal_splitting.md.

    Two families of strategy live here and use different parameters:

    * **fraction** (``temporal_global``, ``temporal_per_user``) cut the log at
      global instants chosen so a given fraction of events falls after them.
    * **leave-last-N** (``per_user_leave_last_n``, ``leave_one_out``) hold out a
      fixed number of each user's most recent events.

    Only the parameters belonging to the selected family are validated, so an
    unused ``test_fraction`` cannot fail a leave-last-N configuration - and,
    more importantly, cannot silently look meaningful when it is ignored.
    """

    strategy: Literal[
        "temporal_global", "temporal_per_user", "leave_one_out", "per_user_leave_last_n"
    ] = "temporal_global"

    # -- fraction-family parameters ----------------------------------------- #
    validation_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    test_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    embargo_days: int = Field(default=0, ge=0)

    # -- leave-last-N-family parameters ------------------------------------- #
    validation_interactions: int = Field(default=1, ge=0)
    test_interactions: int = Field(default=1, ge=1)
    # Training events a user must retain *after* the held-out tail is removed.
    # A user with fewer is ineligible and contributes training history only;
    # the alternative - evaluating a user whose entire history is the target -
    # measures nothing but popularity.
    minimum_history_before_validation: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check_family_parameters(self) -> Self:
        if (
            self.strategy in FRACTION_STRATEGIES
            and self.validation_fraction + self.test_fraction >= 1.0
        ):
            raise ValueError(
                "data.splitting.validation_fraction + test_fraction must leave a "
                "non-empty training window (their sum must be < 1.0)"
            )
        return self

    @property
    def held_out_per_user(self) -> int:
        """Events removed from each eligible user's tail by a leave-last-N split."""
        return self.validation_interactions + self.test_interactions

    @property
    def minimum_eligible_interactions(self) -> int:
        """Total events a user needs to be eligible for evaluation."""
        return self.held_out_per_user + self.minimum_history_before_validation


class SequenceConfig(_Section):
    """User-history sequence shaping for sequential models."""

    max_length: int = Field(default=50, ge=2)
    min_length: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _check_lengths(self) -> Self:
        if self.min_length > self.max_length:
            raise ValueError("data.sequences.min_length must not exceed max_length")
        return self


class DatasetPathsConfig(_Section):
    """Where one dataset's raw, interim, and processed files live."""

    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    # Recorded verbatim in the dataset manifest so provenance survives the
    # dataset leaving this machine.
    source_repository: str = ""
    licence: str = ""


class ProcessingConfig(_Section):
    """Execution knobs that affect cost and safety, never dataset content."""

    # Rows per read block. Bounds peak memory on inputs larger than RAM.
    chunk_size: int = Field(default=100_000, ge=1)
    validate_checksums: bool = True
    # Off by default: silently replacing a processed dataset that a training run
    # may already have consumed is not something a pipeline should do by itself.
    overwrite: bool = False
    # Development subset: keep only the first N users. None processes everything.
    subset_users: int | None = Field(default=None, ge=1)


class InteractionDefaultsConfig(_Section):
    """Defaults applied to sources with a single, unlabelled event type.

    ``default_event_type`` is ``None`` for sources that label their own events
    (an e-commerce log distinguishing cart from purchase needs no default).
    A source like PixelRec, which records one undifferentiated implicit signal,
    sets it to ``"interaction"``.
    """

    default_event_type: str | None = None
    default_weight: float = Field(default=1.0, ge=0.0)


class FeatureFilesConfig(_Section):
    """Pre-extracted multimodal feature files, when the source publishes them."""

    validate_text_features: bool = True
    validate_image_features: bool = True
    # PixelRec publishes raw encoder outputs and documents no normalisation, so
    # none is applied. Flipping this would silently change every vector.
    normalize_precomputed_features: bool = False
    text_feature_file: str = "text_feature.json"
    image_feature_file: str = "image_feature.json"
    # Asserted per vector when set; None accepts whatever the file declares.
    expected_dimension: int | None = Field(default=None, ge=1)
    # Named only when the source documents it - never guessed.
    text_encoder: str | None = None
    image_encoder: str | None = None


class SliceConfig(_Section):
    """Evaluation-slice thresholds."""

    # Items outside the head accounting for this share of training interactions
    # are long tail. Recorded in the report so no number is quoted without it.
    long_tail_quantile: float = Field(default=0.8, gt=0.0, lt=1.0)


class OutputConfig(_Section):
    """Processed-output format."""

    format: Literal["parquet"] = "parquet"


class DataConfig(_Section):
    """Domain profile: the only place a vertical is named. See ADR-001."""

    domain: str
    dataset_name: str
    dataset_version: str
    event_types: dict[str, float]
    positive_event_threshold: float
    validation: ValidationRulesConfig
    filtering: FilteringConfig = Field(default_factory=FilteringConfig)
    splitting: SplittingConfig = Field(default_factory=SplittingConfig)
    sequences: SequenceConfig = Field(default_factory=SequenceConfig)
    # -- Phase 2 additions -------------------------------------------------- #
    # Optional so the Phase 1 e-commerce profile, which has no dataset on disk,
    # remains valid. A pipeline run requires `dataset` and fails clearly without it.
    dataset: DatasetPathsConfig | None = None
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    interactions: InteractionDefaultsConfig = Field(default_factory=InteractionDefaultsConfig)
    features: FeatureFilesConfig = Field(default_factory=FeatureFilesConfig)
    slices: SliceConfig = Field(default_factory=SliceConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _check_event_types(self) -> Self:
        if not self.event_types:
            raise ValueError("data.event_types must declare at least one event type")
        if any(weight < 0 for weight in self.event_types.values()):
            raise ValueError("data.event_types weights must be non-negative")
        if self.positive_event_threshold > max(self.event_types.values()):
            raise ValueError(
                "data.positive_event_threshold exceeds every declared event weight, "
                "so no interaction could ever count as positive"
            )
        if (
            self.interactions.default_event_type is not None
            and self.interactions.default_event_type not in self.event_types
        ):
            raise ValueError(
                f"data.interactions.default_event_type "
                f"{self.interactions.default_event_type!r} is not declared in "
                "data.event_types, so every ingested interaction would be rejected"
            )
        return self

    @property
    def positive_event_types(self) -> tuple[str, ...]:
        """Event names whose weight meets the positive-label threshold."""
        return tuple(
            sorted(
                name
                for name, weight in self.event_types.items()
                if weight >= self.positive_event_threshold
            )
        )


# --- models ---------------------------------------------------------------- #
class GeneratorConfig(_Section):
    """One candidate generator's budget and hyperparameters.

    ``extra="allow"`` here (unlike every other section) because each generator
    owns its own hyperparameter names and Phase 2+ adds them without touching
    this file. ``enabled``/``phase``/``top_k`` are the contract all share.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    enabled: bool = False
    phase: int = Field(ge=1)
    top_k: int = Field(default=200, ge=1)


class AggregationConfig(_Section):
    """How per-generator candidate lists are merged. Component 11.

    The strategy names mirror the constants in
    ``omnirank.retrieval.aggregation``; they are repeated as literals rather
    than imported because the config layer must stay importable without the
    retrieval extra installed.
    """

    strategy: Literal[
        "weighted_round_robin", "reciprocal_rank_fusion", "normalized_score_union"
    ] = "weighted_round_robin"
    max_candidates: int = Field(default=1000, ge=1)
    source_weights: dict[str, float] = Field(default_factory=dict)
    #: Damping constant for reciprocal rank fusion. Must be positive: at zero,
    #: a source's rank-1 item would dominate every fused sum.
    rrf_constant: float = Field(default=60.0, gt=0.0)
    normalization: Literal["min_max", "z_score", "rank_percentile"] = "rank_percentile"
    #: Depth multiplier applied before fusion. At 1, fusing top-k lists and
    #: truncating back to k under-fills whenever the sources agree.
    over_retrieval_factor: int = Field(default=3, ge=1)


class RankerConfig(_Section):
    """Learning-to-rank stage. Component 12. See ADR-008."""

    model_config = ConfigDict(extra="allow", frozen=True)

    enabled: bool = False
    phase: int = Field(ge=1)
    implementation: Literal["lightgbm", "catboost"] = "lightgbm"


class RerankerConfig(_Section):
    """Diversity-aware reranking. Component 13."""

    enabled: bool = False
    phase: int = Field(ge=1)
    strategy: Literal["mmr"] = "mmr"
    lambda_relevance: float = Field(default=0.7, ge=0.0, le=1.0)


class IndexConfig(_Section):
    """Vector index backend. Component 15. See ADR-004 and ADR-006."""

    backend: Literal["faiss", "pgvector", "qdrant"] = "faiss"
    metric: Literal["inner_product", "l2"] = "inner_product"
    index_version: int = Field(default=1, ge=1)


class ModelsConfig(_Section):
    """Retrieval / ranking stage configuration."""

    candidate_generators: dict[str, GeneratorConfig]
    aggregation: AggregationConfig
    ranker: RankerConfig
    reranker: RerankerConfig
    index: IndexConfig

    @model_validator(mode="after")
    def _check_weights_reference_generators(self) -> Self:
        unknown = set(self.aggregation.source_weights) - set(self.candidate_generators)
        if unknown:
            raise ValueError(
                "models.aggregation.source_weights names generators that are not "
                f"declared under models.candidate_generators: {sorted(unknown)}"
            )
        return self

    @property
    def enabled_generators(self) -> tuple[str, ...]:
        """Names of generators currently switched on, in config order."""
        return tuple(name for name, cfg in self.candidate_generators.items() if cfg.enabled)


# --- evaluation ------------------------------------------------------------ #
class BootstrapConfig(_Section):
    """User-level bootstrap confidence intervals for the primary metrics.

    Deterministic given ``seed``: the same recommendations and ground truth
    always produce the same interval, so a reported interval is reproducible
    rather than a property of one lucky run.
    """

    enabled: bool = True
    samples: int = Field(default=1000, ge=1)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    seed: int = Field(default=42, ge=0)


class BeyondAccuracyConfig(_Section):
    """Coverage, novelty, and exposure-inequality reporting."""

    # Additive smoothing on the novelty probability, applied only where a zero
    # would make -log2(p) infinite. Recorded in the report so the number is
    # never quoted without the rule that produced it.
    novelty_smoothing: float = Field(default=1.0, ge=0.0)
    # Whether eligible-but-never-recommended items count toward the Gini
    # denominator. Excluding them flatters a model that ignores the tail.
    gini_includes_zero_exposure: bool = True


class EvaluationConfig(_Section):
    """Offline evaluation protocol. Component 9."""

    k_values: tuple[int, ...]
    metrics: tuple[str, ...]
    beyond_accuracy_metrics: tuple[str, ...] = ()
    filter_seen: bool = True
    protocol: Literal["full", "sampled"] = "full"
    num_sampled_negatives: int = Field(default=100, ge=1)
    min_user_coverage: float = Field(default=0.95, ge=0.0, le=1.0)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    beyond_accuracy: BeyondAccuracyConfig = Field(default_factory=BeyondAccuracyConfig)

    @model_validator(mode="after")
    def _check_k_values(self) -> Self:
        if not self.k_values:
            raise ValueError("evaluation.k_values must not be empty")
        if any(k < 1 for k in self.k_values):
            raise ValueError("evaluation.k_values must all be >= 1")
        if not self.metrics:
            raise ValueError("evaluation.metrics must not be empty")
        return self

    @property
    def max_k(self) -> int:
        """Largest cut-off, i.e. how deep recommendation lists must go."""
        return max(self.k_values)


# --- serving --------------------------------------------------------------- #
class ApiConfig(_Section):
    """HTTP surface."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = False
    title: str = "OmniRank API"
    version: str = "v1"
    log_requests: bool = True


class ServingConfig(_Section):
    """Request-time pipeline budgets. Component 16."""

    default_top_k: int = Field(default=20, ge=1)
    max_top_k: int = Field(default=200, ge=1)
    latency_budget_ms: int = Field(default=300, ge=1)
    filter_unavailable: bool = True

    @model_validator(mode="after")
    def _check_top_k(self) -> Self:
        if self.default_top_k > self.max_top_k:
            raise ValueError("serving.default_top_k must not exceed serving.max_top_k")
        return self


class FallbackConfig(_Section):
    """Ordered degradation chain. See the fallback flow in the architecture docs."""

    chain: tuple[str, ...]
    mark_in_response: bool = True

    @model_validator(mode="after")
    def _check_chain(self) -> Self:
        if not self.chain:
            raise ValueError(
                "fallback.chain must not be empty: the API must always have a way "
                "to produce a non-empty response"
            )
        if self.chain[-1] != "global_popularity":
            raise ValueError(
                "fallback.chain must end with 'global_popularity', the only stage "
                "guaranteed to answer for any user"
            )
        return self


class DatabaseConfig(_Section):
    """PostgreSQL connection settings. Component 19. See ADR-005."""

    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    name: str = "omnirank"
    user: str = "omnirank"
    password: SecretStr | None = None
    pool_size: int = Field(default=5, ge=1)
    pool_timeout_seconds: int = Field(default=10, ge=1)
    echo_sql: bool = False

    def dsn(self, *, driver: str = "postgresql+psycopg") -> str:
        """Build a SQLAlchemy DSN.

        Raises:
            ConfigurationError: No password was supplied. Kept as a hard error
                so a missing secret surfaces at connect time with a named fix
                rather than as an opaque auth failure from the driver.
        """
        if self.password is None:
            raise ConfigurationError(
                "Database password is not set. Provide OMNIRANK__DATABASE__PASSWORD "
                "in your environment or .env file.",
                setting="database.password",
            )
        secret = self.password.get_secret_value()
        return f"{driver}://{self.user}:{secret}@{self.host}:{self.port}/{self.name}"


class RedisConfig(_Section):
    """Redis connection and cache policy. Component 18. See ADR-005."""

    host: str = "localhost"
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0)
    password: SecretStr | None = None
    key_prefix: str = "omnirank:"
    recommendation_ttl_seconds: int = Field(default=300, ge=1)
    socket_timeout_seconds: int = Field(default=2, ge=1)

    def url(self) -> str:
        """Build a redis:// URL. The password, if any, stays a SecretStr until here."""
        auth = f":{self.password.get_secret_value()}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #
class AppConfig(_Section):
    """The fully merged, validated OmniRank configuration."""

    project_name: str = "omnirank"
    environment: Literal["local", "ci", "staging", "production"] = "local"
    seed: int = Field(default=42, ge=0)

    device: DeviceConfig = Field(default_factory=DeviceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig
    models: ModelsConfig
    evaluation: EvaluationConfig
    api: ApiConfig = Field(default_factory=ApiConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    fallback: FallbackConfig
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)

    @model_validator(mode="after")
    def _check_deployed_environments(self) -> Self:
        """Fail fast on settings that are fine locally but wrong when deployed."""
        if self.environment in ("staging", "production"):
            problems: list[str] = []
            if self.database.password is None:
                problems.append("database.password must be set (OMNIRANK__DATABASE__PASSWORD)")
            if self.api.reload:
                problems.append("api.reload must be false outside local development")
            if self.logging.format != "json":
                problems.append("logging.format must be 'json' so logs stay machine-parseable")
            if self.api.host == "127.0.0.1":
                problems.append("api.host must bind beyond loopback to serve traffic")
            if problems:
                raise ValueError(
                    f"invalid configuration for environment={self.environment!r}: "
                    + "; ".join(problems)
                )
        return self

    # -- hashing ------------------------------------------------------------ #
    def _hashable(self, sections: Iterable[str]) -> str:
        """Canonical JSON for the named sections, with secrets excluded."""
        payload = self.model_dump(mode="json", include=set(sections))
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def config_hash(self) -> str:
        """SHA-256 over the whole non-secret configuration.

        ``model_dump`` renders ``SecretStr`` as a mask, so no secret value can
        reach the digest - two deployments differing only in password produce
        the same hash, which is the intended behaviour.
        """
        sections = list(type(self).model_fields)
        return hashlib.sha256(self._hashable(sections).encode()).hexdigest()

    @property
    def training_config_hash(self) -> str:
        """SHA-256 over only the sections that change a trained artifact.

        Recorded as ``configuration_hash`` in artifact metadata (ADR-006), so
        that changing an API port does not invalidate a model.
        """
        sections = ["seed", "device", "data", "models", "evaluation"]
        return hashlib.sha256(self._hashable(sections).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, with file-level errors named precisely."""
    if not path.is_file():
        raise ConfigFileNotFoundError(f"Configuration file not found: {path}", path=str(path))
    try:
        loaded = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigValidationError(
            f"Configuration file is not valid YAML: {path}", path=str(path), reason=str(exc)
        ) from exc
    if not isinstance(loaded, dict):
        raise ConfigValidationError(
            f"Configuration file must contain a mapping at the top level: {path}",
            path=str(path),
            found_type=type(loaded).__name__,
        )
    return loaded


def _parse_env_value(raw: str) -> Any:
    """Coerce an env-var string to a Python scalar using YAML scalar rules.

    ``"true"`` -> ``True``, ``"42"`` -> ``42``, ``"[1, 2]"`` -> ``[1, 2]``,
    ``""`` -> ``None``. Anything unparseable stays a string.
    """
    if raw == "":
        return None
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    """Turn ``OMNIRANK__A__B=v`` variables into ``{"a": {"b": v}}``."""
    overrides: dict[str, Any] = {}
    for key, raw in env.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = [part.lower() for part in key[len(ENV_PREFIX) :].split(ENV_DELIMITER) if part]
        if not path:
            continue
        cursor = overrides
        for part in path[:-1]:
            nxt = cursor.setdefault(part, {})
            if not isinstance(nxt, dict):
                raise ConfigValidationError(
                    f"Environment variable {key} conflicts with another variable that "
                    f"already set {'.'.join(path[:-1])!r} to a scalar.",
                    variable=key,
                )
            cursor = nxt
        cursor[path[-1]] = _parse_env_value(raw)
    return overrides


def _read_dotenv(path: Path) -> dict[str, str]:
    """Parse a minimal ``KEY=value`` .env file. Blank lines and ``#`` ignored."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


#: Overlays under this directory declare a vertical. Exactly one may apply.
DOMAIN_PROFILE_DIR = "data"


def load_config(
    config_dir: Path | str = DEFAULT_CONFIG_DIR,
    *,
    base_file: str = DEFAULT_BASE_FILE,
    overlays: Iterable[Path | str] = (),
    data_profile: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    dotenv_path: Path | str | None = None,
) -> AppConfig:
    """Load, merge, and validate the full configuration.

    Args:
        config_dir: Directory holding ``base.yaml`` and its overlay tree.
        base_file: Entry-point file inside ``config_dir``.
        overlays: Extra YAML files merged after the ones in ``include:``.
            Relative paths resolve against ``config_dir``.
        data_profile: A domain profile that *replaces* whichever profile
            ``base.yaml`` includes, rather than merging on top of it. Passing
            one is how a run selects its vertical.
        env: Environment mapping to read overrides from. Defaults to
            ``os.environ``. Injectable so tests never mutate real process env.
        dotenv_path: ``.env`` file merged *under* ``env``. Defaults to
            ``<config_dir>/../.env``; pass a non-existent path to skip.

    Returns:
        A frozen, validated :class:`AppConfig`.

    Raises:
        ConfigFileNotFoundError: A referenced YAML file is missing.
        ConfigValidationError: The merged result violates the schema.

    Note:
        Domain profiles *replace* rather than merge because the merge is a deep
        one: stacking a single-event-type profile on top of a six-event-type one
        would leave the union declared, and the new domain would silently accept
        events it does not have.
    """
    directory = Path(config_dir)
    merged = _read_yaml(directory / base_file)

    # `include:` is a loader directive, not part of the schema.
    includes = merged.pop("include", []) or []
    if not isinstance(includes, list):
        raise ConfigValidationError(
            "'include' must be a list of overlay paths", found_type=type(includes).__name__
        )

    if data_profile is not None:
        includes = [
            overlay for overlay in includes if Path(overlay).parent.name != DOMAIN_PROFILE_DIR
        ]
        includes.append(data_profile)

    for overlay in [*includes, *overlays]:
        overlay_path = Path(overlay)
        if not overlay_path.is_absolute():
            overlay_path = directory / overlay_path
        merged = _deep_merge(merged, _read_yaml(overlay_path))

    dotenv_file = Path(dotenv_path) if dotenv_path is not None else directory.parent / ".env"
    effective_env: dict[str, str] = dict(_read_dotenv(dotenv_file))
    # Real environment wins over the .env file.
    effective_env.update(os.environ if env is None else env)

    merged = _deep_merge(merged, _env_overrides(effective_env))

    try:
        return AppConfig.model_validate(merged)
    except ValueError as exc:
        raise ConfigValidationError(
            f"Configuration failed validation. Fix the paths listed below, then retry.\n{exc}",
            config_dir=str(directory),
        ) from exc


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Process-wide cached configuration, for use by long-lived services.

    The cache exists so that a request handler can call this without re-reading
    YAML. Tests and CLI entrypoints that need a specific configuration should
    call :func:`load_config` directly instead of clearing this cache.
    """
    return load_config()


__all__ = [
    "DOMAIN_PROFILE_DIR",
    "ENV_PREFIX",
    "FRACTION_STRATEGIES",
    "LEAVE_LAST_N_STRATEGIES",
    "AggregationConfig",
    "ApiConfig",
    "AppConfig",
    "BeyondAccuracyConfig",
    "BootstrapConfig",
    "DataConfig",
    "DatabaseConfig",
    "DatasetPathsConfig",
    "DeviceConfig",
    "EvaluationConfig",
    "FallbackConfig",
    "FeatureFilesConfig",
    "FilteringConfig",
    "GeneratorConfig",
    "IndexConfig",
    "InteractionDefaultsConfig",
    "LoggingConfig",
    "ModelsConfig",
    "OutputConfig",
    "PathsConfig",
    "ProcessingConfig",
    "RankerConfig",
    "RedisConfig",
    "RerankerConfig",
    "SequenceConfig",
    "ServingConfig",
    "SliceConfig",
    "SplittingConfig",
    "ValidationRulesConfig",
    "get_config",
    "load_config",
]
