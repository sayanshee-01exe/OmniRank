"""Train and score a two-tower configuration on one rolling fold.

Selection on a single train-to-validation boundary measures one week. Phase 3
ended with a model ordering that reversed between validation and test, which one
origin cannot distinguish from a real difference. Everything here exists so that
a configuration's margin can be compared against its own variation across folds
and seeds.

Scoring is done here rather than through
:func:`omnirank.models.baselines.runner.run_experiment` for one reason: that
driver scores against a *split*, and a fold target is not a split. The metric
definitions are the same -- single-positive Recall@K and NDCG@K over the same
cutoffs -- so fold numbers stay comparable with each other, but they are not
interchangeable with split numbers and are never written to the same file.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from omnirank.core.exceptions import OmniRankError
from omnirank.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnirank.data.rolling import RollingFold

logger = get_logger(__name__)

#: Cutoffs shared with the split evaluator, so the two are read the same way.
CUTOFFS = (5, 10, 20, 50, 100, 200)

#: Retrieval depth for the candidate-recall ceiling.
CANDIDATE_DEPTH = 200

#: The ablation grid, keyed by label. Each entry is a delta against the tracked
#: development configuration, never a full configuration: a variant that
#: restated every field would silently stop tracking changes to the baseline.
ABLATION_OVERRIDES: dict[str, dict[str, Any]] = {
    "text_only": {
        "use_text": True,
        "use_image": False,
        "use_tag": False,
        "use_item_id_residual": False,
    },
    "image_only": {
        "use_text": False,
        "use_image": True,
        "use_tag": False,
        "use_item_id_residual": False,
    },
    "text_image": {
        "use_text": True,
        "use_image": True,
        "use_tag": False,
        "use_item_id_residual": False,
    },
    "text_image_tag": {
        "use_text": True,
        "use_image": True,
        "use_tag": True,
        "use_item_id_residual": False,
    },
    "full_with_id_residual": {
        "use_text": True,
        "use_image": True,
        "use_tag": True,
        "use_item_id_residual": True,
    },
    "full_no_user_id": {
        "use_text": True,
        "use_image": True,
        "use_tag": True,
        "use_item_id_residual": False,
        "use_user_id_embedding": False,
    },
    "mean_pooling": {
        "use_text": True,
        "use_image": True,
        "use_tag": True,
        "use_item_id_residual": False,
        "history_pooling": "mean",
    },
    # The control the multimodal vectors are measured against. Tags are a
    # single categorical id per item -- essentially free, and available for any
    # catalogue. If text and image cannot beat this, the published vectors are
    # not earning the 17 GB of storage and the alignment step they cost.
    #
    # A fully content-free control was attempted and is not constructible: the
    # item tower raises `Item tower has no enabled content inputs`, because a
    # tower with no content is not an item tower. The content-free comparison
    # in this phase is therefore LightGCN and BPR, which are genuinely
    # collaborative-only models, not a crippled two-tower.
    "tag_only": {
        "use_text": False,
        "use_image": False,
        "use_tag": True,
        "use_item_id_residual": False,
    },
    # Capacity probe. If a modality result is really a "too few dimensions to
    # hold both modalities" result, doubling the space is where it shows.
    "wide_embedding": {
        "use_text": True,
        "use_image": True,
        "use_tag": True,
        "use_item_id_residual": False,
        "embedding_dim": 256,
    },
}


def overrides_for(label: str) -> dict[str, Any]:
    """Configuration overrides for one ablation label."""
    if label not in ABLATION_OVERRIDES:
        raise OmniRankError(
            f"Unknown ablation label: {label}",
            known=sorted(ABLATION_OVERRIDES),
        )
    return dict(ABLATION_OVERRIDES[label])


def score_recommendations(
    recommended: dict[str, list[str]],
    targets: dict[str, str],
    cold_items: set[str],
) -> dict[str, Any]:
    """Recall/NDCG at :data:`CUTOFFS`, reported overall and on the cold subset.

    Each user has exactly one held-out target, so NDCG reduces to
    ``1 / log2(rank + 1)`` -- written out rather than routed through the general
    gain formula, because a reader checking this against the split evaluator
    should be able to see that they compute the same thing.
    """
    hits = dict.fromkeys(CUTOFFS, 0)
    gains = dict.fromkeys(CUTOFFS, 0.0)
    cold_hits = dict.fromkeys(CUTOFFS, 0)
    evaluated = cold_evaluated = 0

    for user, target in targets.items():
        items = recommended.get(user, [])
        if not items:
            # A user the model cannot answer for still counts as evaluated
            # elsewhere; here they are excluded, and the count is reported so
            # the denominator is never silently different between runs.
            continue
        evaluated += 1
        is_cold = target in cold_items
        cold_evaluated += int(is_cold)
        position = items.index(target) + 1 if target in items else None
        if position is None:
            continue
        for cut in CUTOFFS:
            if position <= cut:
                hits[cut] += 1
                gains[cut] += 1.0 / math.log2(position + 1)
                if is_cold:
                    cold_hits[cut] += 1

    divisor = max(evaluated, 1)
    payload: dict[str, Any] = {
        "users_evaluated": evaluated,
        "cold_users_evaluated": cold_evaluated,
    }
    for cut in CUTOFFS:
        payload[f"strict_recall@{cut}"] = round(hits[cut] / divisor, 8)
        payload[f"strict_ndcg@{cut}"] = round(gains[cut] / divisor, 8)
        # ``None``, not 0.0, when no target was cold. Within a rolling fold
        # every target is warm by construction -- the contrastive objective
        # uses targets as positives, so the model has seen them and they carry
        # an identity residual. A cold rate over zero cold users is undefined,
        # and writing 0.0 would read as "cold retrieval failed" when the truth
        # is that this protocol cannot measure it. Cold retrieval is measured
        # on the test split, where held-out items genuinely are unseen.
        payload[f"cold_recall@{cut}"] = (
            round(cold_hits[cut] / cold_evaluated, 8) if cold_evaluated else None
        )
    payload["candidate_recall@200"] = payload[f"strict_recall@{CANDIDATE_DEPTH}"]
    return payload


def evaluate_on_fold(
    label: str,
    fold: RollingFold,
    seed: int,
    *,
    dataset: Any,
    processed_root: Path | str,
    base_config: dict[str, Any],
    epochs: int,
    device: str = "cpu",
    subset_users: int | None = None,
) -> dict[str, Any]:
    """Fit ``label`` on one fold's history and score that fold's targets.

    Returns one flat record: identity, runtime, and metrics. Flat because it is
    written straight to CSV, and a nested record would have to be flattened at
    the point of writing, where the column names stop being reviewable.
    """
    from omnirank.models.two_tower import TwoTowerConfig, TwoTowerRetriever
    from omnirank.retrieval.runner import fit_two_tower, load_item_tags, sequences_from_fold

    raw = dict(base_config)
    raw.update(overrides_for(label))
    raw.update({"max_epochs": epochs, "seed": seed, "device": device})
    model_config = TwoTowerConfig(**raw)

    sequences = sequences_from_fold(
        fold, maximum_history_length=model_config.maximum_history_length
    )
    if subset_users is not None:
        sequences = sequences[sequences["internal_user_id"] < subset_users]
    if sequences.empty:
        raise OmniRankError(
            "Fold produced no training examples",
            fold=fold.name,
            label=label,
            subset_users=subset_users,
        )

    started = time.perf_counter()
    (network, store, history), measurement = fit_two_tower(
        dataset,
        ("train",),
        model_config,
        processed_root=processed_root,
        device=device,
        sequences=sequences,
    )
    train_seconds = time.perf_counter() - started

    tags, _ = load_item_tags(processed_root, dataset.num_items)
    histories: dict[int, list[int]] = {}
    warm = np.zeros(dataset.num_items, dtype=bool)
    for user, items, target in zip(
        sequences["internal_user_id"],
        sequences["item_sequence"],
        sequences["target_item"],
        strict=True,
    ):
        histories[int(user)] = [int(item) for item in items]
        warm[[int(item) for item in items]] = True
        # The target is warm because *this fold's* history period contained it
        # for somebody. Warmth is a property of the fold, not of the dataset.
        warm[int(target)] = True

    retriever = TwoTowerRetriever.from_trained(
        network, store, dataset, histories, warm, tags, device=device
    )
    retriever.export_item_embeddings()

    started = time.perf_counter()
    external_user = {
        internal: external for external, internal in dataset.external_to_internal_users().items()
    }
    external_item = dataset.internal_to_external_items()
    targets = {
        external_user[int(user)]: external_item[int(item)]
        for user, item in zip(sequences["internal_user_id"], sequences["target_item"], strict=True)
        if int(user) in external_user and int(item) in external_item
    }
    recommended = retriever.recommend_batch(sorted(targets), CANDIDATE_DEPTH)
    evaluation_seconds = time.perf_counter() - started

    cold_items = {
        external_item[item] for item in retriever.cold_item_catalogue if item in external_item
    }
    metrics = score_recommendations(recommended, targets, cold_items)

    record = {
        "experiment_id": f"{label}@{fold.name}@seed{seed}",
        "label": label,
        "fold": fold.name,
        "fold_offset": fold.offset,
        "fold_checksum": fold.checksum[:16],
        "seed": seed,
        "configuration_hash": model_config.label,
        "training_examples": len(sequences),
        "best_epoch": history.best_epoch,
        "stopped_early": history.stopped_early,
        "epochs_run": len(history.train_loss),
        "final_train_loss": round(history.train_loss[-1], 6) if history.train_loss else None,
        "train_seconds": round(train_seconds, 1),
        "evaluation_seconds": round(evaluation_seconds, 1),
        "peak_memory_mb": round(measurement.peak_memory_mb or 0.0, 1),
        "device": history.device,
        "warm_items": int(warm.sum()),
        "cold_items": len(cold_items),
        **metrics,
        **overrides_for(label),
    }
    logger.info(
        "fold_evaluation.complete",
        label=label,
        fold=fold.name,
        seed=seed,
        strict_ndcg20=record["strict_ndcg@20"],
        cold_recall20=record["cold_recall@20"],
    )
    return record


def summarise_folds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate fold records per configuration, spread alongside the mean.

    The worst fold is reported beside the mean because a configuration that
    wins on average by collapsing at one origin is not stable, and a mean alone
    hides exactly that. The standard deviation is the reference a claimed
    margin has to clear.
    """
    import statistics

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)

    summary: list[dict[str, Any]] = []
    for label, records in sorted(grouped.items()):
        ndcg = [record["strict_ndcg@20"] for record in records]
        recall = [record["strict_recall@20"] for record in records]
        # Absent where the fold had no cold target at all -- see
        # `score_recommendations`. Averaging those in as zeros would report a
        # failure that was never measured.
        cold = [
            record["cold_recall@20"]
            for record in records
            if record.get("cold_recall@20") is not None
        ]
        summary.append(
            {
                "label": label,
                "runs": len(records),
                "folds": "+".join(sorted({record["fold"] for record in records})),
                "seeds": "+".join(str(seed) for seed in sorted({r["seed"] for r in records})),
                "mean_strict_ndcg@20": round(statistics.mean(ndcg), 8),
                "worst_fold_strict_ndcg@20": round(min(ndcg), 8),
                # The worst *fold*, averaging over seeds within each fold
                # first. Unlike the worst single run, this does not punish a
                # configuration for having been measured more times: more runs
                # mean more chances to draw a low one, so `min` over runs
                # systematically favours whichever contender was measured least.
                "worst_fold_mean_strict_ndcg@20": round(
                    min(
                        statistics.mean(
                            [
                                record["strict_ndcg@20"]
                                for record in records
                                if record["fold"] == fold_name
                            ]
                        )
                        for fold_name in {record["fold"] for record in records}
                    ),
                    8,
                ),
                "stdev_strict_ndcg@20": round(statistics.stdev(ndcg), 8) if len(ndcg) > 1 else 0.0,
                "mean_strict_recall@20": round(statistics.mean(recall), 8),
                "cold_runs_measured": len(cold),
                "mean_cold_recall@20": round(statistics.mean(cold), 8) if cold else None,
                "worst_cold_recall@20": round(min(cold), 8) if cold else None,
                "mean_candidate_recall@200": round(
                    statistics.mean([record["candidate_recall@200"] for record in records]), 8
                ),
                "mean_train_seconds": round(
                    statistics.mean([record["train_seconds"] for record in records]), 1
                ),
                "peak_memory_mb": max(record["peak_memory_mb"] for record in records),
            }
        )
    return summary


__all__ = [
    "ABLATION_OVERRIDES",
    "CANDIDATE_DEPTH",
    "CUTOFFS",
    "evaluate_on_fold",
    "overrides_for",
    "score_recommendations",
    "summarise_folds",
]
