"""Configuration for the multimodal two-tower retriever.

Every field is validated at construction. A silently-accepted bad value here
does not crash -- it trains a slightly different model than the one recorded in
the selection file, which is the kind of drift that only surfaces when someone
tries to reproduce a number months later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from omnirank.core.exceptions import DataError

MEAN_POOLING: Final = "mean"
RECENCY_WEIGHTED_POOLING: Final = "recency_weighted_mean"
POOLING_STRATEGIES: Final = (MEAN_POOLING, RECENCY_WEIGHTED_POOLING)

CONCAT_FUSION: Final = "concat"
GATED_FUSION: Final = "gated"
FUSION_STRATEGIES: Final = (CONCAT_FUSION, GATED_FUSION)

GELU: Final = "gelu"
RELU: Final = "relu"
ACTIVATIONS: Final = (GELU, RELU)


@dataclass(frozen=True, slots=True)
class TwoTowerConfig:
    """Hyperparameters for the multimodal two-tower retriever."""

    #: Shared output width of both towers. Retrieval is a dot product between
    #: them, so they cannot differ.
    embedding_dim: int = 128
    user_id_embedding_dim: int = 64
    item_id_embedding_dim: int = 64
    tag_embedding_dim: int = 32
    text_projection_dim: int = 128
    image_projection_dim: int = 128
    hidden_dims: tuple[int, ...] = (256, 128)
    activation: str = GELU

    history_pooling: str = RECENCY_WEIGHTED_POOLING
    #: Geometric decay applied to older history positions under
    #: recency-weighted pooling. 1.0 makes it identical to mean pooling.
    recency_decay: float = 0.9
    modality_fusion: str = GATED_FUSION

    #: Modality switches. Turning one off is how the ablations are run, so they
    #: are configuration rather than separate model classes.
    use_text: bool = True
    use_image: bool = True
    use_tag: bool = True
    use_user_id_embedding: bool = True
    #: The warm-item identity residual. Gated by a warm mask at encode time, so
    #: enabling it never gives a cold item an untrained embedding.
    use_item_id_residual: bool = True

    l2_normalize: bool = True
    temperature: float = 0.07
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    maximum_history_length: int = 50
    max_epochs: int = 50
    early_stopping_patience: int = 5
    #: Max global gradient norm. Contrastive losses at low temperature produce
    #: occasional very large gradients; clipping keeps one bad batch from
    #: destroying weights that took an epoch to learn.
    gradient_clip_norm: float = 1.0
    #: "auto" | "cpu" | "mps". Never selects CUDA.
    device: str = "auto"
    #: Users scored per batch during full-catalogue retrieval. Bounds peak
    #: memory at batch x catalogue floats rather than users x items.
    evaluation_user_batch_size: int = 256
    #: Mask in-batch negatives that are known positives for the same user.
    mask_false_negatives: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        positive = (
            ("embedding_dim", self.embedding_dim),
            ("user_id_embedding_dim", self.user_id_embedding_dim),
            ("item_id_embedding_dim", self.item_id_embedding_dim),
            ("tag_embedding_dim", self.tag_embedding_dim),
            ("text_projection_dim", self.text_projection_dim),
            ("image_projection_dim", self.image_projection_dim),
            ("batch_size", self.batch_size),
            ("maximum_history_length", self.maximum_history_length),
            ("max_epochs", self.max_epochs),
            ("early_stopping_patience", self.early_stopping_patience),
            ("evaluation_user_batch_size", self.evaluation_user_batch_size),
        )
        problems = [name for name, value in positive if value < 1]
        if self.temperature <= 0:
            problems.append("temperature")
        if not 0.0 <= self.dropout < 1.0:
            problems.append("dropout")
        if self.learning_rate <= 0:
            problems.append("learning_rate")
        if self.weight_decay < 0:
            problems.append("weight_decay")
        if self.seed < 0:
            problems.append("seed")
        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            problems.append("hidden_dims")
        if not 0.0 < self.recency_decay <= 1.0:
            problems.append("recency_decay")
        if self.gradient_clip_norm <= 0:
            problems.append("gradient_clip_norm")
        if problems:
            raise DataError("Invalid two-tower configuration", invalid=sorted(problems))

        if self.activation not in ACTIVATIONS:
            raise DataError(
                "Unknown activation",
                activation=self.activation,
                available=list(ACTIVATIONS),
            )
        if self.device not in ("auto", "cpu", "mps"):
            raise DataError(
                "Unknown device. CUDA is never selected implicitly.",
                device=self.device,
                available=["auto", "cpu", "mps"],
            )
        if self.history_pooling not in POOLING_STRATEGIES:
            raise DataError(
                "Unknown history pooling strategy",
                history_pooling=self.history_pooling,
                available=list(POOLING_STRATEGIES),
            )
        if self.modality_fusion not in FUSION_STRATEGIES:
            raise DataError(
                "Unknown modality fusion strategy",
                modality_fusion=self.modality_fusion,
                available=list(FUSION_STRATEGIES),
            )
        if not (self.use_text or self.use_image or self.use_tag or self.use_item_id_residual):
            raise DataError(
                "The item tower needs at least one input. With every content "
                "source disabled and no identity residual it has nothing to "
                "encode, and every item would receive the same vector.",
            )

    @property
    def content_enabled(self) -> bool:
        """Whether any content source feeds the item tower.

        A model with none of them cannot represent a cold item, whatever its
        warm metrics look like. Used to refuse cold-item claims rather than to
        silently return an empty cold catalogue.
        """
        return self.use_text or self.use_image or self.use_tag

    @property
    def label(self) -> str:
        """Compact identifier for experiment tables."""
        modalities = "".join(
            flag
            for flag, enabled in (
                ("T", self.use_text),
                ("I", self.use_image),
                ("G", self.use_tag),
                ("U", self.use_user_id_embedding),
                ("R", self.use_item_id_residual),
            )
            if enabled
        )
        pooling = "rw" if self.history_pooling == RECENCY_WEIGHTED_POOLING else "mean"
        return f"d{self.embedding_dim}_{modalities or 'none'}_{pooling}_t{self.temperature:g}"

    def to_dict(self) -> dict[str, Any]:
        """Serialisable payload."""
        return {
            "embedding_dim": self.embedding_dim,
            "user_id_embedding_dim": self.user_id_embedding_dim,
            "item_id_embedding_dim": self.item_id_embedding_dim,
            "tag_embedding_dim": self.tag_embedding_dim,
            "text_projection_dim": self.text_projection_dim,
            "image_projection_dim": self.image_projection_dim,
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation,
            "history_pooling": self.history_pooling,
            "recency_decay": self.recency_decay,
            "modality_fusion": self.modality_fusion,
            "use_text": self.use_text,
            "use_image": self.use_image,
            "use_tag": self.use_tag,
            "use_user_id_embedding": self.use_user_id_embedding,
            "use_item_id_residual": self.use_item_id_residual,
            "l2_normalize": self.l2_normalize,
            "temperature": self.temperature,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "maximum_history_length": self.maximum_history_length,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "gradient_clip_norm": self.gradient_clip_norm,
            "device": self.device,
            "evaluation_user_batch_size": self.evaluation_user_batch_size,
            "mask_false_negatives": self.mask_false_negatives,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TwoTowerConfig:
        """Rebuild from :meth:`to_dict`, tolerating list-typed hidden_dims."""
        data = dict(payload)
        if "hidden_dims" in data:
            data["hidden_dims"] = tuple(data["hidden_dims"])
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in known})


__all__ = [
    "CONCAT_FUSION",
    "FUSION_STRATEGIES",
    "GATED_FUSION",
    "MEAN_POOLING",
    "POOLING_STRATEGIES",
    "RECENCY_WEIGHTED_POOLING",
    "TwoTowerConfig",
]
