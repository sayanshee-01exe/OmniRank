#!/usr/bin/env python
"""Train a recommendation model.

    python scripts/train.py --model popularity --config-dir configs

**Phase 1 status: no model is implemented.** This script validates its
arguments against the configured generator registry - so ``--model lightgcn``
correctly reports that LightGCN is a Phase 3 deliverable rather than failing with
an import error - and then exits non-zero.

Planned flow (Phase 2 onward): load the prepared dataset, construct the named
model behind ``omnirank.models.base.CandidateGenerator``, fit, evaluate on the
validation split, export the model plus any embeddings, and register the result
with full :class:`~omnirank.artifacts.metadata.ArtifactMetadata`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnirank.core.config import load_config
from omnirank.core.exceptions import ConfigurationError
from omnirank.core.logging import configure_logging, get_logger, run_context

NOT_IMPLEMENTED_EXIT = 3


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Train an OmniRank model.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument(
        "--model",
        required=True,
        help="Model key from models.candidate_generators, or 'ranker'.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the configured seed.")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config_dir)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging(config.logging, force=True)
    logger = get_logger("omnirank.train")

    known = {**config.models.candidate_generators}
    if args.model == "ranker":
        phase = config.models.ranker.phase
    elif args.model in known:
        phase = known[args.model].phase
    else:
        print(
            f"Unknown model {args.model!r}. Known: {sorted([*known, 'ranker'])}",
            file=sys.stderr,
        )
        return 2

    with run_context(stage="train", model=args.model) as run_id:
        logger.info(
            "train.configuration_ok",
            run_id=run_id,
            model=args.model,
            planned_phase=phase,
            seed=args.seed if args.seed is not None else config.seed,
            device=config.device.preferred.value,
            training_config_hash=config.training_config_hash[:16],
        )
        logger.error(
            "train.not_implemented",
            run_id=run_id,
            model=args.model,
            planned_phase=phase,
            detail=(
                f"{args.model!r} is a Phase {phase} deliverable. Phase 1 defines the "
                "CandidateGenerator/Ranker interfaces but implements no model."
            ),
        )
    return NOT_IMPLEMENTED_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
