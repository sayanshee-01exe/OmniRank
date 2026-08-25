"""The multimodal two-tower network.

Two encoders that must land in the same space, because retrieval is a dot
product between them:

    user history + user id  -> user tower  -> u
    item content + item id  -> item tower  -> v
    score(u, i)             = u . v

**The item tower is what makes Phase 5 worth doing.** Every retriever before it
represents an item by who interacted with it, so an item nobody has interacted
with has no representation at all. This tower represents an item by what it
*is* -- its text vector, its image vector, its category -- which is available
the moment the item exists. That is the whole cold-start argument, and §5.3
below is where it either holds or quietly fails.

**Missing modalities are masked, never faked.** A zero vector is not "no text",
it is a specific point in text space that every text-less item would share.
Each modality carries a learned missing-token and an availability mask, so
"absent" is a state the model can represent rather than a coincidence it has to
infer.
"""

from __future__ import annotations

from typing import Final, cast

import torch
from torch import nn

from omnirank.core.exceptions import DataError
from omnirank.models.two_tower.config import (
    GATED_FUSION,
    GELU,
    RECENCY_WEIGHTED_POOLING,
    TwoTowerConfig,
)

#: Guards the division when a normalised vector is all zeros, which happens for
#: an empty history before pooling.
NORMALIZE_EPSILON: Final = 1e-8


def _activation(name: str) -> nn.Module:
    """Build the configured activation."""
    return nn.GELU() if name == GELU else nn.ReLU()


def _mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    *,
    activation: str,
    dropout: float,
) -> nn.Sequential:
    """Standard projection stack shared by both towers."""
    layers: list[nn.Module] = []
    width = input_dim
    for hidden in hidden_dims:
        layers += [nn.Linear(width, hidden), _activation(activation), nn.Dropout(dropout)]
        width = hidden
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class ModalityEncoder(nn.Module):
    """Projects one modality, substituting a learned token when it is absent.

    The missing-token is the point of this class. Projecting a zero vector would
    put every item lacking the modality at whatever point the projection maps
    zero to -- a location that carries no meaning but is nonetheless shared, so
    the model learns spurious similarity between items whose only common
    property is an absent feature.
    """

    def __init__(self, input_dim: int, output_dim: int, *, activation: str) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = _activation(activation)
        self.missing = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        """Encode, replacing unavailable rows with the learned missing token.

        Args:
            features: ``(..., input_dim)``.
            available: ``(...)`` boolean; ``False`` rows use the missing token.
        """
        projected = cast("torch.Tensor", self.activation(self.norm(self.projection(features))))
        mask = available.unsqueeze(-1).to(projected.dtype)
        return projected * mask + self.missing * (1.0 - mask)


class ItemTower(nn.Module):
    """Encodes an item from its content, plus an identity residual when warm."""

    def __init__(
        self,
        config: TwoTowerConfig,
        *,
        text_dim: int,
        image_dim: int,
        num_items: int,
        num_tags: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_items = num_items

        fused_width = 0
        if config.use_text:
            self.text_encoder = ModalityEncoder(
                text_dim, config.text_projection_dim, activation=config.activation
            )
            fused_width += config.text_projection_dim
        if config.use_image:
            self.image_encoder = ModalityEncoder(
                image_dim, config.image_projection_dim, activation=config.activation
            )
            fused_width += config.image_projection_dim
        if config.use_tag:
            self.tag_embedding = nn.Embedding(max(num_tags, 1), config.tag_embedding_dim)
            fused_width += config.tag_embedding_dim

        if config.modality_fusion == GATED_FUSION and fused_width:
            # A scalar gate per modality slot, so the model can learn to ignore a
            # modality that is usually absent instead of averaging noise into it.
            self.gate = nn.Sequential(nn.Linear(fused_width, fused_width), nn.Sigmoid())
        else:
            self.gate = None  # type: ignore[assignment]

        self.content_mlp = _mlp(
            max(fused_width, 1),
            config.hidden_dims,
            config.embedding_dim,
            activation=config.activation,
            dropout=config.dropout,
        )
        if config.use_item_id_residual:
            # Sized to the output space directly: it is added to the content
            # embedding, not concatenated before it, so that zeroing it for a
            # cold item leaves a well-formed content vector behind.
            self.item_id_embedding = nn.Embedding(num_items, config.embedding_dim)
            nn.init.normal_(self.item_id_embedding.weight, std=0.01)

    def encode_content(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        text_available: torch.Tensor,
        image_available: torch.Tensor,
        tag_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Content-only embedding. This is what a cold item gets."""
        parts: list[torch.Tensor] = []
        if self.config.use_text:
            parts.append(self.text_encoder(text, text_available))
        if self.config.use_image:
            parts.append(self.image_encoder(image, image_available))
        if self.config.use_tag:
            parts.append(self.tag_embedding(tag_ids))
        if not parts:
            raise DataError("Item tower has no enabled content inputs")

        fused = torch.cat(parts, dim=-1)
        if self.gate is not None:
            fused = fused * self.gate(fused)
        return cast("torch.Tensor", self.content_mlp(fused))

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        text_available: torch.Tensor,
        image_available: torch.Tensor,
        tag_ids: torch.Tensor,
        item_ids: torch.Tensor | None = None,
        warm_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full item embedding: content plus a warm-gated identity residual.

        ``final = content + warm_mask * id_residual``

        The gate is the entire cold-start guarantee. An item id that never
        appeared in training has an embedding still at its random
        initialisation; adding it would inject noise of the same magnitude as
        the learned signal, and the resulting vector would be neither content
        nor identity. Multiplying by ``warm_mask`` -- zero for a cold item --
        means a cold item's representation is *exactly* its content embedding,
        which is verifiable rather than hoped for.
        """
        embedding = self.encode_content(text, image, text_available, image_available, tag_ids)
        if self.config.use_item_id_residual and item_ids is not None:
            residual = self.item_id_embedding(item_ids)
            gate = (
                torch.ones_like(residual[..., :1])
                if warm_mask is None
                else warm_mask.unsqueeze(-1).to(residual.dtype)
            )
            embedding = embedding + residual * gate
        return embedding


class UserTower(nn.Module):
    """Encodes a user from their history, optionally plus an identity embedding."""

    def __init__(self, config: TwoTowerConfig, item_tower: ItemTower, *, num_users: int) -> None:
        super().__init__()
        self.config = config
        # Shared, not duplicated: a history item and a candidate item are the
        # same object, so encoding them with different weights would put the
        # query and the key in subtly different spaces.
        self.item_tower = item_tower
        self.num_users = num_users

        width = config.embedding_dim
        if config.use_user_id_embedding:
            self.user_id_embedding = nn.Embedding(num_users, config.user_id_embedding_dim)
            nn.init.normal_(self.user_id_embedding.weight, std=0.01)
            width += config.user_id_embedding_dim
        # History length, bucketed coarsely and passed as a scalar, so the model
        # can distinguish a confident long history from a one-item guess.
        width += 1

        self.fusion_mlp = _mlp(
            width,
            config.hidden_dims,
            config.embedding_dim,
            activation=config.activation,
            dropout=config.dropout,
        )

    def pool(
        self, encoded: torch.Tensor, padding_mask: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Pool per-item history embeddings into one vector.

        Padded positions contribute nothing: they are zeroed before summing and
        excluded from the divisor. A mean that divided by the padded width would
        shrink short histories towards the origin, making history length a
        hidden magnitude signal.
        """
        keep = (~padding_mask).unsqueeze(-1).to(encoded.dtype)
        masked = encoded * keep

        if self.config.history_pooling == RECENCY_WEIGHTED_POOLING:
            width = encoded.shape[1]
            # Histories are right-aligned, so position width-1 is the most
            # recent and gets weight 1.
            exponents = torch.arange(width - 1, -1, -1, device=encoded.device, dtype=encoded.dtype)
            weights = (self.config.recency_decay**exponents).unsqueeze(0).unsqueeze(-1)
            weights = weights * keep
        else:
            weights = keep

        total = weights.sum(dim=1)
        pooled = (masked * weights).sum(dim=1) / total.clamp(min=NORMALIZE_EPSILON)
        # An empty history pools to zero rather than to whatever the divisor
        # guard leaves behind.
        return pooled * (lengths > 0).unsqueeze(-1).to(pooled.dtype)

    def forward(
        self,
        history_text: torch.Tensor,
        history_image: torch.Tensor,
        history_text_available: torch.Tensor,
        history_image_available: torch.Tensor,
        history_tag_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        lengths: torch.Tensor,
        user_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of users from their histories.

        History items are encoded through the item tower's *content* path only.
        Using the identity residual here would make a user's query vector depend
        on item ids, and an unknown-user request built from a supplied history
        would then need those ids to be warm.
        """
        batch, width = padding_mask.shape
        flat_text = history_text.reshape(batch * width, -1)
        flat_image = history_image.reshape(batch * width, -1)
        encoded = self.item_tower.encode_content(
            flat_text,
            flat_image,
            history_text_available.reshape(-1),
            history_image_available.reshape(-1),
            history_tag_ids.reshape(-1),
        ).reshape(batch, width, -1)

        pooled = self.pool(encoded, padding_mask, lengths)
        parts = [pooled]
        if self.config.use_user_id_embedding:
            if user_ids is None:
                # An unknown user has no identity to look up. Zeros keep the
                # concatenation shape while contributing nothing, so the query
                # comes entirely from the supplied history.
                parts.append(
                    torch.zeros(
                        batch,
                        self.config.user_id_embedding_dim,
                        device=pooled.device,
                        dtype=pooled.dtype,
                    )
                )
            else:
                parts.append(self.user_id_embedding(user_ids))
        parts.append(torch.log1p(lengths.to(pooled.dtype)).unsqueeze(-1))
        return cast("torch.Tensor", self.fusion_mlp(torch.cat(parts, dim=-1)))


class MultimodalTwoTower(nn.Module):
    """User and item towers sharing one embedding space."""

    def __init__(
        self,
        config: TwoTowerConfig,
        *,
        text_dim: int,
        image_dim: int,
        num_items: int,
        num_users: int,
        num_tags: int = 1,
    ) -> None:
        super().__init__()
        if text_dim < 1 or image_dim < 1:
            raise DataError(
                "Feature dimensions must be positive",
                text_dim=text_dim,
                image_dim=image_dim,
            )
        if num_items < 1 or num_users < 1:
            raise DataError(
                "Catalogue and user counts must be positive",
                num_items=num_items,
                num_users=num_users,
            )
        self.config = config
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.num_items = num_items
        self.num_users = num_users
        self.num_tags = max(num_tags, 1)

        self.item_tower = ItemTower(
            config,
            text_dim=text_dim,
            image_dim=image_dim,
            num_items=num_items,
            num_tags=self.num_tags,
        )
        self.user_tower = UserTower(config, self.item_tower, num_users=num_users)

    def _normalize(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply the configured normalisation.

        With L2 on, a dot product is a cosine similarity, and FAISS's inner
        product means the same thing. The rule travels in the metadata because
        an index built under one convention and queried under the other returns
        confident nonsense.
        """
        if not self.config.l2_normalize:
            return vectors
        norms = cast("torch.Tensor", vectors.norm(dim=-1, keepdim=True))
        return vectors / norms.clamp(min=NORMALIZE_EPSILON)

    def encode_items(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        text_available: torch.Tensor,
        image_available: torch.Tensor,
        tag_ids: torch.Tensor,
        item_ids: torch.Tensor | None = None,
        warm_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode items. Cold items pass ``warm_mask=0`` and are content-only."""
        return self._normalize(
            self.item_tower(
                text, image, text_available, image_available, tag_ids, item_ids, warm_mask
            )
        )

    def encode_users(
        self,
        history_text: torch.Tensor,
        history_image: torch.Tensor,
        history_text_available: torch.Tensor,
        history_image_available: torch.Tensor,
        history_tag_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        lengths: torch.Tensor,
        user_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode users. ``user_ids=None`` builds the query from history alone."""
        return self._normalize(
            self.user_tower(
                history_text,
                history_image,
                history_text_available,
                history_image_available,
                history_tag_ids,
                padding_mask,
                lengths,
                user_ids,
            )
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a collated batch into aligned user and item vectors."""
        users = self.encode_users(
            batch["history_text_features"],
            batch["history_image_features"],
            batch["history_text_available"],
            batch["history_image_available"],
            batch["history_tag_ids"],
            batch["history_padding_mask"],
            batch["history_lengths"],
            batch.get("user_ids") if self.config.use_user_id_embedding else None,
        )
        items = self.encode_items(
            batch["positive_text_features"],
            batch["positive_image_features"],
            batch["positive_text_available"],
            batch["positive_image_available"],
            batch["positive_tag_ids"],
            batch.get("positive_item_ids"),
            batch.get("positive_warm_mask"),
        )
        return users, items

    def similarity(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Dot-product scores between user and item vectors."""
        if users.shape[-1] != items.shape[-1]:
            raise DataError(
                "User and item embeddings must share a width",
                user_dim=int(users.shape[-1]),
                item_dim=int(items.shape[-1]),
            )
        return users @ items.T

    def modality_schema(self) -> dict[str, object]:
        """What this model expects from a feature store."""
        return {
            "text_dim": self.text_dim,
            "image_dim": self.image_dim,
            "uses_text": self.config.use_text,
            "uses_image": self.config.use_image,
            "uses_tag": self.config.use_tag,
            "num_tags": self.num_tags,
            "missing_modality_policy": "learned missing token plus availability mask",
            "cold_item_policy": (
                "content only; the item-id residual is multiplied by a warm mask "
                "that is zero for items with no fitting interaction"
            ),
            "normalization": "l2" if self.config.l2_normalize else "none",
        }


__all__ = [
    "NORMALIZE_EPSILON",
    "ItemTower",
    "ModalityEncoder",
    "MultimodalTwoTower",
    "UserTower",
]
