# OmniRank developer commands.
# Everything runs inside the project venv; no global installs, no activation needed.

PYTHON  := .venv/bin/python
PIP     := uv pip
VENV    := .venv

.DEFAULT_GOAL := help
.PHONY: help setup install lint format typecheck test test-unit test-integration check serve up down clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

$(VENV):
	uv venv --python 3.11

setup: $(VENV)  ## Create the venv, install the project with dev extras, seed .env
	$(PIP) install -e ".[dev]"
	@test -f .env || (cp .env.example .env && echo "created .env from .env.example")
	@echo "Ready. Run 'make test' or 'make serve'."

install:  ## Reinstall the project (after dependency changes)
	$(PIP) install -e ".[dev]"

lint:  ## Ruff check
	$(PYTHON) -m ruff check src tests scripts

format:  ## Ruff format (writes)
	$(PYTHON) -m ruff format src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

typecheck:  ## MyPy strict
	$(PYTHON) -m mypy

test:  ## Full test suite
	$(PYTHON) -m pytest

test-unit:  ## Unit tests only
	$(PYTHON) -m pytest tests/unit

test-integration:  ## Integration tests only
	$(PYTHON) -m pytest tests/integration -m integration

check: lint typecheck test  ## Lint + typecheck + test (what CI runs)

serve:  ## Run the API locally
	$(PYTHON) scripts/serve.py

up:  ## Start PostgreSQL and Redis (applies schema.sql on first start)
	docker compose up -d
	@echo "postgres :5432  redis :6379   (stop with 'make down')"

down:  ## Stop the backing services
	docker compose down

clean:  ## Remove caches and build output
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
