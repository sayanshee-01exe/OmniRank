# OmniRank developer commands.
# Everything runs inside the project venv; no global installs, no activation needed.

PYTHON  := .venv/bin/python
PIP     := uv pip
VENV    := .venv

.DEFAULT_GOAL := help
.PHONY: help setup install install-baseline install-retrieval lint format typecheck \
        test test-unit test-integration test-baseline test-retrieval check serve up \
        down clean download-data prepare-data validate-data profile-data \
        train-popularity train-mf evaluate-popularity evaluate-mf compare-baselines \
        train-lightgcn train-sasrec evaluate-lightgcn evaluate-sasrec \
        build-lightgcn-index build-sasrec-index compare-retrievers \
        compare-aggregation benchmark-index

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

install-retrieval:  ## Install the retrieval extra (adds FAISS on top of baseline)
	$(PIP) install -e ".[retrieval,dev]"

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

test-retrieval:  ## LightGCN, SASRec, aggregation and index tests (needs the retrieval extra)
	$(PYTHON) -m pytest tests/unit/retrieval tests/unit/models/test_lightgcn.py \
	    tests/unit/models/test_sasrec.py tests/unit/data/test_rolling.py

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

train-lightgcn:  ## Fit + register LightGCN (selection stage)
	$(PYTHON) scripts/train.py --model lightgcn \
	    --data-config configs/data/pixelrec50k.yaml \
	    --stage selection --version phase4-lightgcn-selection

train-sasrec:  ## Fit + register SASRec (selection stage)
	$(PYTHON) scripts/train.py --model sasrec \
	    --data-config configs/data/pixelrec50k.yaml \
	    --stage selection --version phase4-sasrec-selection

evaluate-lightgcn:  ## Evaluate the registered LightGCN model on validation
	$(PYTHON) scripts/evaluate.py --model lightgcn \
	    --version phase4-lightgcn-selection --split validation --protocol full

evaluate-sasrec:  ## Evaluate the registered SASRec model on validation
	$(PYTHON) scripts/evaluate.py --model sasrec \
	    --version phase4-sasrec-selection --split validation --protocol full

build-lightgcn-index:  ## Build a FAISS index over LightGCN item embeddings
	$(PYTHON) scripts/build_index.py --model lightgcn \
	    --version phase4-lightgcn-selection --index-type flat_ip --verify-exact

build-sasrec-index:  ## Build a FAISS index over SASRec item embeddings
	$(PYTHON) scripts/build_index.py --model sasrec \
	    --version phase4-sasrec-selection --index-type flat_ip --verify-exact

compare-retrievers:  ## Full Phase 4 comparison: selection, rolling, lock, final, reports
	$(PYTHON) scripts/compare_retrievers.py \
	    --config-dir configs --data-config configs/data/pixelrec50k.yaml

compare-aggregation:  ## Fit every source once, then score each source and every blend
	$(PYTHON) scripts/compare_aggregation.py \
	    --config-dir configs --data-config configs/data/pixelrec50k.yaml --stage selection

benchmark-index:  ## Measure every FAISS index type against exact search
	$(PYTHON) scripts/benchmark_index.py \
	    --model lightgcn --version phase4-lightgcn-final

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
