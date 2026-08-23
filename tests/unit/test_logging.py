"""Structured logging: configuration, correlation, and redaction."""

from __future__ import annotations

import json
import logging

import pytest

from omnirank.core.config import LoggingConfig
from omnirank.core.logging import (
    REDACTED,
    bound_context,
    configure_logging,
    get_logger,
    new_run_id,
    reset_context,
    run_context,
)


@pytest.fixture
def emit(capsys):
    """Emit an event through the real pipeline and return the parsed JSON.

    Deliberately not ``structlog.testing.capture_logs``: that helper replaces the
    processor chain, which is exactly where contextvar merging and redaction
    live. Testing through the real renderer is the only way to prove they run.
    """

    def _emit(event: str, **values: object) -> dict[str, object]:
        # Reconfigured inside the call phase on purpose: capsys swaps sys.stderr
        # between the setup and call phases, so a handler bound during setup
        # would write into a buffer this function never reads.
        configure_logging(LoggingConfig(format="json", level="DEBUG"), force=True)
        capsys.readouterr()
        get_logger("test").info(event, **values)
        line = capsys.readouterr().err.strip().splitlines()[-1]
        parsed: dict[str, object] = json.loads(line)
        return parsed

    return _emit


class TestConfiguration:
    def test_configures_the_root_logger(self):
        configure_logging(LoggingConfig(level="WARNING"), force=True)
        assert logging.getLogger().level == logging.WARNING
        assert logging.getLogger().handlers

    def test_is_idempotent_without_force(self):
        configure_logging(LoggingConfig(level="ERROR"), force=True)
        configure_logging(LoggingConfig(level="DEBUG"))  # no force: ignored
        assert logging.getLogger().level == logging.ERROR

    def test_force_reconfigures(self):
        configure_logging(LoggingConfig(level="ERROR"), force=True)
        configure_logging(LoggingConfig(level="DEBUG"), force=True)
        assert logging.getLogger().level == logging.DEBUG

    @pytest.mark.parametrize("fmt", ["console", "json"])
    def test_both_renderers_configure_cleanly(self, fmt):
        configure_logging(LoggingConfig(format=fmt), force=True)
        get_logger("test").info("configured")

    def test_json_output_is_parseable(self, capsys):
        configure_logging(LoggingConfig(format="json", level="INFO"), force=True)
        get_logger("test").info("machine_readable", answer=42)
        payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert payload["event"] == "machine_readable"
        assert payload["answer"] == 42

    def test_get_logger_works_before_configuration(self):
        """A module logging during startup must not silently drop its output."""
        import omnirank.core.logging as logging_module

        logging_module._configured = False
        try:
            get_logger("early").info("works")
            assert logging_module._configured is True
        finally:
            configure_logging(force=True)


class TestCorrelation:
    def test_bound_context_attaches_to_events(self, emit):
        with bound_context(run_id="abc123", stage="training"):
            event = emit("started")
        assert event["run_id"] == "abc123"
        assert event["stage"] == "training"

    def test_context_is_unbound_on_exit(self, emit):
        with bound_context(run_id="abc123"):
            pass
        assert "run_id" not in emit("after")

    def test_context_is_unbound_even_when_the_block_raises(self, emit):
        with pytest.raises(RuntimeError), bound_context(run_id="abc123"):
            raise RuntimeError("boom")
        assert "run_id" not in emit("after")

    def test_run_context_generates_and_yields_an_id(self, emit):
        with run_context() as run_id:
            event = emit("in_run")
        assert event["run_id"] == run_id
        assert len(run_id) == 12

    def test_run_context_accepts_an_explicit_id(self, emit):
        with run_context("fixed-id", stage="eval"):
            event = emit("in_run")
        assert event["run_id"] == "fixed-id"
        assert event["stage"] == "eval"

    def test_nested_contexts_compose(self, emit):
        with bound_context(run_id="outer"), bound_context(stage="inner"):
            event = emit("nested")
        assert event["run_id"] == "outer"
        assert event["stage"] == "inner"

    def test_run_ids_are_unique(self):
        assert len({new_run_id() for _ in range(200)}) == 200

    def test_reset_clears_everything(self, emit):
        from structlog.contextvars import bind_contextvars

        bind_contextvars(leftover="x")
        reset_context()
        assert "leftover" not in emit("clean")


class TestRedaction:
    def test_top_level_sensitive_key_is_masked(self, capsys):
        configure_logging(LoggingConfig(format="json"), force=True)
        get_logger("test").info("login", password="hunter2")
        assert "hunter2" not in capsys.readouterr().err

    def test_nested_sensitive_key_is_masked(self, capsys):
        configure_logging(LoggingConfig(format="json"), force=True)
        get_logger("test").info("conn", db={"host": "localhost", "password": "hunter2"})
        captured = capsys.readouterr().err
        assert "hunter2" not in captured
        assert "localhost" in captured

    def test_masking_reaches_inside_lists(self, capsys):
        configure_logging(LoggingConfig(format="json"), force=True)
        get_logger("test").info("many", conns=[{"token": "abc"}, {"token": "def"}])
        captured = capsys.readouterr().err
        assert "abc" not in captured
        assert "def" not in captured

    def test_matching_is_case_insensitive(self, capsys):
        configure_logging(LoggingConfig(format="json"), force=True)
        get_logger("test").info("auth", Authorization="Bearer xyz")
        assert "xyz" not in capsys.readouterr().err

    def test_non_sensitive_values_survive(self, capsys):
        configure_logging(LoggingConfig(format="json"), force=True)
        get_logger("test").info("ok", user_count=5, host="db.internal")
        captured = capsys.readouterr().err
        assert "db.internal" in captured
        assert REDACTED not in captured

    def test_redact_keys_are_configurable(self, capsys):
        configure_logging(LoggingConfig(format="json", redact_keys=("custom_field",)), force=True)
        get_logger("test").info("event", custom_field="hide-me", password="not-listed")
        captured = capsys.readouterr().err
        assert "hide-me" not in captured
        # Only the configured keys are masked - the list is authoritative.
        assert "not-listed" in captured
