#!/usr/bin/env python
"""Four-source versus five-source candidate fusion.

    python scripts/compare_five_source_fusion.py --split test

The comparison Phase 5 exists to make: does adding a content-based retriever to
four collaborative ones reach targets they cannot?

**Registered artifacts are loaded, not refitted.** The four collaborative
sources already exist as final Phase 3-4 artifacts; refitting them here would
take roughly two hours and, worse, would compare *different models* to the ones
whose numbers are already reported. Loading is both cheaper and the only way the
fusion result stays comparable to the standalone results.

The two-tower's value may not be its standalone rank. A retriever that scores
below LightGCN alone can still be worth its place if it proposes items the other
four never do -- so unique contribution and cold-target reach are measured
alongside the aggregate, not instead of it.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from omnirank.core.config import load_config
from omnirank.core.exceptions import ConfigurationError, OmniRankError
from omnirank.core.logging import configure_logging, get_logger, run_context
from omnirank.data.processed import load_processed_dataset
from omnirank.evaluation.reporting import REPORT_ROOT, write_csv, write_json
from omnirank.models.baselines.runner import boundary_for_stage, run_experiment
from omnirank.retrieval.aggregation import build_aggregator
from omnirank.retrieval.blended import BlendedRetriever
from omnirank.retrieval.diagnostics import candidate_recall, source_overlap

CONFIG_ERROR_EXIT = 2
RUN_ERROR_EXIT = 3

PHASE_ROOT = REPORT_ROOT.parent / "phase_05"
PROCESSED = "data/processed/pixelrec50k"
COLLABORATIVE = ("popularity", "matrix_factorization", "lightgcn", "sasrec")
TWO_TOWER = "two_tower"

#: Candidate budgets §15 asks for. Depth is what a ranker inherits, so recall is
#: measured across it rather than at one arbitrary cut.
BUDGETS = (50, 100, 200, 500, 1200)

REGISTERED = {
    "popularity": "phase3-popularity-final",
    "matrix_factorization": "phase3-mf-final",
    "lightgcn": "phase4-lightgcn-final",
    "sasrec": "phase5-sasrec-final",
    "two_tower": "phase5-two-tower-final",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--data-config", default="configs/data/pixelrec50k.yaml")
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "mps"))
    parser.add_argument("--subset-users", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--diagnostic-depth", type=int, default=300)
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args(argv)


def _load_registered(name: str, config: Any, device: str) -> Any:
    """Load one registered collaborative retriever."""
    path = Path(config.paths.models_dir) / config.data.dataset_name / name / REGISTERED[name]
    if not path.is_dir():
        raise OmniRankError(f"{name} is not registered at {path}")
    if name == "popularity":
        from omnirank.models.baselines.popularity import PopularityRecommender

        return PopularityRecommender.load(path)
    if name == "matrix_factorization":
        from omnirank.models.baselines.bpr import BPRMatrixFactorization

        return BPRMatrixFactorization.load(path, device=device)
    if name == "lightgcn":
        from omnirank.models.lightgcn import LightGCN

        return LightGCN.load(path, device=device)
    from omnirank.models.sasrec import SASRec

    return SASRec.load(path, device=device)


def _load_two_tower(config: Any, args: argparse.Namespace) -> Any:
    """Load the registered final two-tower, beside its feature store.

    Loaded, not refitted, for the same reason the other four sources are: a
    fusion number and a single-source number in the same report must describe
    the same model. Refitting produced an instance that *should* have matched
    the registered one and, since the seeding fix, actually would -- but
    "should" is not "verified", and there is no reason to spend fifteen minutes
    reintroducing the doubt.
    """
    from omnirank.features.multimodal_store import MultimodalFeatureStore
    from omnirank.models.two_tower import TwoTowerRetriever

    path = (
        Path(config.paths.models_dir) / config.data.dataset_name / TWO_TOWER / REGISTERED[TWO_TOWER]
    )
    if not path.is_dir():
        raise OmniRankError(
            f"The final two-tower is not registered at {path}. "
            "Run: python scripts/compare_multimodal_retrievers.py --stage final"
        )
    store = MultimodalFeatureStore(Path(PROCESSED) / "features")
    retriever = TwoTowerRetriever.load(path, store=store, device=args.device)
    retriever.export_item_embeddings()
    get_logger("omnirank.fusion").info(
        "fusion.two_tower_loaded",
        path=str(path),
        warm=retriever.catalogue.warm_count,
        cold=retriever.catalogue.cold_count,
    )
    return retriever


def _score(
    system: str,
    model: Any,
    dataset: Any,
    config: Any,
    fit_splits: tuple[str, ...],
    target: str,
    kind: str,
    sources: str,
    per_user_out: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, Any]:
    """Evaluate one system through the shared harness.

    ``per_user_out`` collects the per-user metric maps. A paired bootstrap
    needs them, and recomputing them by scoring twice would risk the interval
    describing a different evaluation from the point estimate beside it.
    """
    started = time.perf_counter()
    result = run_experiment(
        model,
        dataset,
        config,
        model_name=system,
        model_version=f"fusion-{system}",
        fit_splits=fit_splits,
        target_split=target,
        fit_measurement=None,
        configuration={},
        bootstrap=False,
    )
    flat = result.strict.flat()
    if per_user_out is not None:
        per_user_out[system] = dict(result.strict.per_user)
    slices = {item.slice_name: item.to_dict() for item in result.slices}
    cold = slices.get("items_cold_start", {})
    return {
        "system": system,
        "kind": kind,
        "sources": sources,
        "ndcg@20": round(flat.get("ndcg@20", 0.0), 8),
        "recall@20": round(flat.get("recall@20", 0.0), 8),
        "coverage@20": round(flat.get("coverage@20", 0.0), 8),
        "novelty@20": round(flat.get("novelty@20", 0.0), 6),
        "cold_ndcg@20": round(cold.get("ndcg@20", 0.0), 8),
        "cold_recall@20": round(cold.get("recall@20", 0.0), 8),
        "cold_recall@50": round(cold.get("recall@50", 0.0), 8),
        "cold_users": cold.get("users", 0),
        "unreachable_cold_users": slices.get("targets_unreachable_cold", {}).get("users", 0),
        "evaluation_seconds": round(time.perf_counter() - started, 1),
    }


#: Comparisons the phase actually turns on. Each is (challenger, baseline).
BOOTSTRAP_PAIRS = (
    ("five_source_rrf", "four_source_rrf"),
    ("five_source_rrf", "lightgcn"),
    ("lightgcn_two_tower", "lightgcn"),
    ("two_tower", "lightgcn"),
)

#: Resamples. 1000 is the project default and is enough for a 95% interval
#: whose width is reported to three decimals.
BOOTSTRAP_SAMPLES = 1000


def _write_bootstrap(
    per_user: dict[str, dict[str, dict[str, float]]], logger: Any, run_id: str
) -> None:
    """Paired bootstrap intervals for the comparisons that carry the phase.

    Paired, over shared users, with the same resampled indices applied to both
    systems -- an unpaired interval over the same numbers would be far wider
    and would be measuring user variance rather than the difference.
    """
    from omnirank.evaluation.bootstrap import paired_bootstrap_delta

    rows: list[dict[str, Any]] = []
    for challenger, baseline in BOOTSTRAP_PAIRS:
        if challenger not in per_user or baseline not in per_user:
            logger.warning(
                "fusion.bootstrap_skipped",
                run_id=run_id,
                challenger=challenger,
                baseline=baseline,
                reason="one system was not scored in this run",
            )
            continue
        for metric in ("recall@20", "ndcg@20"):
            try:
                interval = paired_bootstrap_delta(
                    per_user[challenger],
                    per_user[baseline],
                    metric,
                    samples=BOOTSTRAP_SAMPLES,
                )
            except OmniRankError as exc:
                logger.warning(
                    "fusion.bootstrap_failed",
                    run_id=run_id,
                    challenger=challenger,
                    baseline=baseline,
                    metric=metric,
                    reason=str(exc),
                )
                continue
            rows.append(
                {
                    "challenger": challenger,
                    "baseline": baseline,
                    "metric": metric,
                    "delta": round(interval.point_estimate, 8),
                    "ci_lower": round(interval.lower, 8),
                    "ci_upper": round(interval.upper, 8),
                    "confidence_level": interval.confidence_level,
                    "samples": interval.samples,
                    "users": interval.users,
                    # The only reading that matters: an interval straddling
                    # zero is not evidence of a difference, however large the
                    # point estimate looks.
                    "excludes_zero": bool(interval.lower > 0 or interval.upper < 0),
                }
            )
            logger.info("fusion.bootstrap", run_id=run_id, **rows[-1])
    write_csv(rows, PHASE_ROOT / "bootstrap_deltas.csv")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    config_dir = Path(args.config_dir)
    profile = Path(args.data_config)
    with contextlib.suppress(ValueError):
        profile = profile.resolve().relative_to(config_dir.resolve())

    try:
        config = load_config(config_dir, data_profile=profile)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR_EXIT

    configure_logging(config.logging, force=True)
    logger = get_logger("omnirank.fusion")
    fit_splits, target = boundary_for_stage("final" if args.split == "test" else "selection")

    with run_context(stage="five_source_fusion") as run_id:
        try:
            dataset = load_processed_dataset(
                Path(config.data.dataset.processed_dir),
                Path(config.paths.mappings_dir) / config.data.dataset_name,
                verify_checksums=not args.skip_checksums,
            )
            models: dict[str, Any] = {
                name: _load_registered(name, config, args.device) for name in COLLABORATIVE
            }
            logger.info("fusion.loaded_registered", run_id=run_id, sources=sorted(models))
            models[TWO_TOWER] = _load_two_tower(config, args)
        except OmniRankError as exc:
            logger.error("fusion.setup_failed", run_id=run_id, reason=str(exc))
            return RUN_ERROR_EXIT

        rows: list[dict[str, Any]] = []
        per_user: dict[str, dict[str, dict[str, float]]] = {}
        for name, model in models.items():
            rows.append(
                _score(name, model, dataset, config, fit_splits, target, "single", name, per_user)
            )
            logger.info(
                "fusion.solo",
                run_id=run_id,
                **{k: rows[-1][k] for k in ("system", "ndcg@20", "cold_recall@20")},
            )
            write_csv(rows, PHASE_ROOT / "five_source_fusion_metrics.csv")

        blends = {
            "four_source_rrf": COLLABORATIVE,
            "five_source_rrf": (*COLLABORATIVE, TWO_TOWER),
            "lightgcn_two_tower": ("lightgcn", TWO_TOWER),
            "sasrec_two_tower": ("sasrec", TWO_TOWER),
        }
        for label, members in blends.items():
            blend = BlendedRetriever(
                {name: models[name] for name in members},
                build_aggregator("reciprocal_rank_fusion"),
                name=label,
            )
            rows.append(
                _score(
                    label,
                    blend,
                    dataset,
                    config,
                    fit_splits,
                    target,
                    "blend",
                    "+".join(members),
                    per_user,
                )
            )
            logger.info(
                "fusion.blend",
                run_id=run_id,
                **{k: rows[-1][k] for k in ("system", "ndcg@20", "cold_recall@20")},
            )
            write_csv(rows, PHASE_ROOT / "five_source_fusion_metrics.csv")

        _write_bootstrap(per_user, logger, run_id)

        # Diagnostics: does the two-tower reach what the others cannot?
        users = sorted(dataset.external_to_internal_users())
        depth = args.diagnostic_depth
        per_source = {name: model.recommend_batch(users, depth) for name, model in models.items()}
        overlap = source_overlap(per_source, depth=depth)
        write_csv(
            [
                {"pair": pair, "jaccard": value}
                for pair, value in sorted(overlap.pairwise_jaccard.items())
            ]
            + [
                {"pair": f"unique_contribution::{name}", "jaccard": value}
                for name, value in sorted(overlap.unique_contribution.items())
            ],
            PHASE_ROOT / "source_overlap.csv",
        )

        internal_to_external_user = {
            internal: external
            for external, internal in dataset.external_to_internal_users().items()
        }
        internal_to_external_item = dataset.internal_to_external_items()
        targets: dict[str, set[str]] = {}
        for row in dataset.split(target).itertuples():
            user = internal_to_external_user.get(int(row.internal_user_id))
            item = internal_to_external_item.get(int(row.internal_item_id))
            if user is not None and item is not None:
                targets.setdefault(user, set()).add(item)

        recall_rows = []
        for budget in BUDGETS:
            for label, members in (
                ("four_source", COLLABORATIVE),
                ("five_source", (*COLLABORATIVE, TWO_TOWER)),
            ):
                pool = {
                    user: sorted(
                        {
                            item
                            for name in members
                            for item in per_source[name].get(user, [])[:budget]
                        }
                    )
                    for user in users
                }
                measured = candidate_recall(pool, targets, depth=budget * len(members))
                recall_rows.append({"budget": budget, "sources": label, **measured.to_dict()})
        write_csv(recall_rows, PHASE_ROOT / "candidate_recall.csv")

        # Targets only the two-tower reached.
        others = {
            user: {
                item for name in COLLABORATIVE for item in per_source[name].get(user, [])[:depth]
            }
            for user in users
        }
        unique_hits = sum(
            1
            for user, wanted in targets.items()
            if wanted & set(per_source[TWO_TOWER].get(user, [])[:depth]) - others.get(user, set())
        )
        write_json(
            {
                "depth": depth,
                "targets_reached_only_by_two_tower": unique_hits,
                "mean_sources_per_item": overlap.mean_sources_per_item,
                "unique_contribution": overlap.unique_contribution,
                "pairwise_jaccard": overlap.pairwise_jaccard,
            },
            PHASE_ROOT / "two_tower_unique_contribution.json",
        )
        logger.info(
            "fusion.complete",
            run_id=run_id,
            systems=len(rows),
            targets_only_two_tower=unique_hits,
            mean_sources_per_item=round(overlap.mean_sources_per_item, 4),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
