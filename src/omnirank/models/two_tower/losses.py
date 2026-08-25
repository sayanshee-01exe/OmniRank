"""Contrastive retrieval objective for the two-tower model.

The loss is in-batch softmax (InfoNCE): every other item in the batch acts as a
negative for every user. That is what makes two-tower training affordable --
one forward pass over B items yields B x B comparisons instead of requiring an
explicit negative sampler.

It also introduces the failure this module spends most of its effort on.

**In-batch negatives are not all negative.** If user *i* has already interacted
with the item that happens to be user *j*'s target, the loss actively pushes
user *i* away from an item they demonstrably like. On a catalogue of 69,347
items with a batch of 512 the collision rate is low, but it is not zero, and it
is systematically biased towards popular items -- exactly the items most likely
to appear as someone else's target. Left unmasked it trains the model to
under-rank popular relevant items, which no loss curve reveals.

So known positives are masked out of the negative set. The masked entries are
set to a large negative logit rather than removed, because the softmax
denominator has to keep its shape for the batched implementation to stay
vectorised.

**Duplicate targets are a special case of the same problem.** When two rows in
a batch share a target item, each is the other's positive, and the off-diagonal
copy would otherwise be scored as a negative for an item the user *does* want.
Handled by the same mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from omnirank.core.exceptions import DataError

#: Added to masked logits. Large enough to vanish under softmax in float32,
#: small enough not to produce inf/NaN when the row is entirely masked.
MASKED_LOGIT: Final = -1e4


@dataclass(frozen=True, slots=True)
class ContrastiveOutput:
    """Loss plus the diagnostics needed to tell a healthy batch from a broken one."""

    loss: torch.Tensor
    #: Fraction of off-diagonal entries removed as known positives.
    masked_fraction: float
    #: Fraction of rows whose highest logit is the true positive.
    in_batch_accuracy: float
    #: Rows whose target duplicated another row's target.
    duplicate_targets: int


def build_false_negative_mask(
    target_items: torch.Tensor,
    user_positive_sets: list[set[int]] | None = None,
) -> torch.Tensor:
    """Mark in-batch pairs that must not be treated as negatives.

    Args:
        target_items: ``(batch,)`` internal item ids, one per row.
        user_positive_sets: Per row, the items that row's user is known to like.
            ``None`` masks only duplicate targets, which is the subset that can
            be detected without user history.

    Returns:
        ``(batch, batch)`` boolean where ``True`` means "exclude this pair from
        the negatives". The diagonal is always ``False``: a row's own positive
        is the label, not a negative, and masking it would remove the only
        term the numerator has.
    """
    if target_items.ndim != 1:
        raise DataError(
            "target_items must be a 1-D batch of item ids", shape=list(target_items.shape)
        )
    batch = target_items.shape[0]
    # Same item appearing as another row's target.
    mask = target_items.unsqueeze(0) == target_items.unsqueeze(1)

    if user_positive_sets is not None:
        if len(user_positive_sets) != batch:
            raise DataError(
                "user_positive_sets must align with the batch",
                rows=len(user_positive_sets),
                batch=batch,
            )
        targets = target_items.tolist()
        extra = torch.zeros_like(mask)
        for row, positives in enumerate(user_positive_sets):
            if not positives:
                continue
            for column, item in enumerate(targets):
                if row != column and item in positives:
                    extra[row, column] = True
        mask = mask | extra

    # The diagonal is the label. Whatever the rules above concluded, it stays.
    mask.fill_diagonal_(False)
    return mask


def in_batch_contrastive_loss(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    *,
    temperature: float,
    false_negative_mask: torch.Tensor | None = None,
    target_items: torch.Tensor | None = None,
) -> ContrastiveOutput:
    """InfoNCE over in-batch negatives.

    ``S[i][j] = (u_i . v_j) / temperature``, and row *i*'s label is column *i*.

    Args:
        user_embeddings: ``(batch, dim)``.
        item_embeddings: ``(batch, dim)``, positionally aligned with the users.
        temperature: Softmax sharpness. Lower concentrates probability on the
            nearest items; too low saturates and stalls learning.
        false_negative_mask: ``(batch, batch)`` pairs to exclude, from
            :func:`build_false_negative_mask`.
        target_items: Used only to count duplicate targets for the diagnostics.

    Raises:
        DataError: Shapes disagree, the batch is empty, or temperature is not
            positive.
    """
    if temperature <= 0:
        raise DataError("Temperature must be positive", temperature=temperature)
    if user_embeddings.shape != item_embeddings.shape:
        raise DataError(
            "User and item embeddings must have the same shape. Retrieval is a "
            "dot product between the towers, so their widths cannot differ.",
            user_shape=list(user_embeddings.shape),
            item_shape=list(item_embeddings.shape),
        )
    if user_embeddings.ndim != 2 or user_embeddings.shape[0] == 0:
        raise DataError(
            "Contrastive loss needs a non-empty 2-D batch",
            shape=list(user_embeddings.shape),
        )

    batch = user_embeddings.shape[0]
    logits = (user_embeddings @ item_embeddings.T) / temperature

    masked_fraction = 0.0
    if false_negative_mask is not None:
        if false_negative_mask.shape != (batch, batch):
            raise DataError(
                "False-negative mask must be (batch, batch)",
                mask_shape=list(false_negative_mask.shape),
                batch=batch,
            )
        logits = logits.masked_fill(false_negative_mask, MASKED_LOGIT)
        off_diagonal = batch * (batch - 1)
        masked_fraction = float(false_negative_mask.sum()) / off_diagonal if off_diagonal else 0.0

    labels = torch.arange(batch, device=logits.device)
    # cross_entropy applies log_softmax internally, which is the numerically
    # stable form; computing softmax and taking a log separately overflows for
    # the small temperatures this objective uses.
    loss = torch.nn.functional.cross_entropy(logits, labels)

    with torch.no_grad():
        accuracy = float((logits.argmax(dim=1) == labels).float().mean())
        duplicates = int(batch - len(set(target_items.tolist()))) if target_items is not None else 0

    return ContrastiveOutput(
        loss=loss,
        masked_fraction=masked_fraction,
        in_batch_accuracy=accuracy,
        duplicate_targets=duplicates,
    )


__all__ = [
    "MASKED_LOGIT",
    "ContrastiveOutput",
    "build_false_negative_mask",
    "in_batch_contrastive_loss",
]
