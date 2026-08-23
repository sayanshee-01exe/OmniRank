#!/usr/bin/env python
"""Run the OmniRank API locally.

    python scripts/serve.py
    python scripts/serve.py --port 9000 --config-dir configs

Configuration comes from ``configs/base.yaml`` plus ``.env`` plus environment
variables; command-line flags override all three, which is the convention for
things you change per-invocation rather than per-environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/serve.py` from a checkout without installing first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnirank.core.config import load_config
from omnirank.core.exceptions import ConfigurationError
from omnirank.core.logging import configure_logging, get_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the OmniRank API server.")
    parser.add_argument("--config-dir", default="configs", help="Directory holding base.yaml.")
    parser.add_argument("--host", default=None, help="Override api.host.")
    parser.add_argument("--port", type=int, default=None, help="Override api.port.")
    parser.add_argument("--reload", action="store_true", help="Force auto-reload on code changes.")
    parser.add_argument(
        "--no-reload", action="store_true", help="Disable auto-reload regardless of config."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)

    try:
        config = load_config(args.config_dir)
    except ConfigurationError as exc:
        # Printed rather than logged: logging is configured *from* the config we
        # just failed to load, and a stack trace would bury the actual problem.
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging(config.logging, force=True)
    logger = get_logger("omnirank.serve")

    host = args.host or config.api.host
    port = args.port or config.api.port
    reload = config.api.reload
    if args.reload:
        reload = True
    if args.no_reload:
        reload = False

    logger.info(
        "serve.starting",
        host=host,
        port=port,
        reload=reload,
        environment=config.environment,
        docs_url=f"http://{host}:{port}/docs",
    )

    import uvicorn

    if reload:
        # Reload requires an import string so the worker can re-import the app.
        # The factory then loads configuration itself in the child process.
        uvicorn.run(
            "omnirank.api.app:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
            reload_dirs=["src"],
            log_config=None,  # keep our structlog handler
        )
    else:
        from omnirank.api.app import create_app

        uvicorn.run(create_app(config), host=host, port=port, log_config=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
