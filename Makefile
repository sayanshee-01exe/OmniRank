# OmniRank developer commands.
# Everything runs inside the project venv; no global installs, no activation needed.

PYTHON  := .venv/bin/python
PIP     := uv pip
VENV    := .venv

.DEFAULT_GOAL := help
.PHONY: help setup install install-baseline lint format typecheck test test-unit \
        test-integration test-baseline check serve up down clean download-data \
        prepare-data validate-data profile-data train-popularity train-mf \
        evaluate-popularity evaluate-mf compare-baselines

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

install-baseline:  ## Install the baseline modelling extra (adds PyTorch)
	$(PIP) install -e ".[baseline,dev]"

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

test-baseline:  ## Evaluation + baseline model tests (needs the baseline extra)
	$(PYTHON) -m pytest tests/unit/evaluation tests/unit/models \
	    tests/integration/test_baseline_pipeline.py

check: lint typecheck test  ## Lint + typecheck + test (what CI runs)

download-data:  ## Download PixelRec50K (51 MB; --with-features adds 17.3 GB)
	$(PYTHON) scripts/download_pixelrec50k.py

prepare-data:  ## Build the processed PixelRec50K dataset
	$(PYTHON) scripts/prepare_data.py --config configs/data/pixelrec50k.yaml --overwrite

validate-data:  ## Check the raw source files exist and match the expected schema
	$(PYTHON) scripts/prepare_data.py --config configs/data/pixelrec50k.yaml --validate-only

profile-data:  ## Profile the raw dataset only, then stop
	$(PYTHON) scripts/prepare_data.py --config configs/data/pixelrec50k.yaml --profile-only

train-popularity:  ## Fit + register the popularity baseline (selection stage)
	$(PYTHON) scripts/train.py --model popularity \
	    --data-config configs/data/pixelrec50k.yaml \
	    --stage selection --version phase3-popularity-selection

train-mf:  ## Fit + register the BPR baseline (selection stage)
	$(PYTHON) scripts/train.py --model matrix_factorization \
	    --data-config configs/data/pixelrec50k.yaml \
	    --stage selection --version phase3-mf-selection

evaluate-popularity:  ## Evaluate the registered popularity model on validation
	$(PYTHON) scripts/evaluate.py --model popularity \
	    --version phase3-popularity-selection --split validation --protocol full

evaluate-mf:  ## Evaluate the registered BPR model on validation
	$(PYTHON) scripts/evaluate.py --model matrix_factorization \
	    --version phase3-mf-selection --split validation --protocol full

compare-baselines:  ## Full Phase 3 comparison: selection, lock, final, reports
	$(PYTHON) scripts/compare_baselines.py \
	    --config-dir configs --data-config configs/data/pixelrec50k.yaml

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
