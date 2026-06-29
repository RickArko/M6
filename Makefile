# M6 Forecasting Uncertainty — canonical entrypoint.
# Linux/macOS/WSL only. `make help` to see everything.

.DEFAULT_GOAL := help
.PHONY: help bootstrap install activate lint fmt typecheck test test-smoke test-unit \
        test-integration test-fast cov check \
        download prep cv-naive cv-historical cv-gaussian cv-ensemble cv-adaptive cv-csp cv-recipe \
        score score-all submit all notebook viz clean clean-all

UV       ?= uv
VENV     ?= .venv
HORIZON  ?= 20
WINDOWS  ?= 6
MODEL    ?= adaptive
MODELS   ?= naive historical gaussian ensemble adaptive csp
REPORT   ?= reports
RUN_ID   ?= latest

export UV_PROJECT_ENVIRONMENT := $(VENV)
export PYTHONWARNINGS := ignore::SyntaxWarning

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "; \
	             printf "\nM6 Forecasting — make targets\n\nUsage: make <target> [VAR=value]\n\n"} \
	     /^[a-zA-Z_.-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nVariables (override on CLI): HORIZON=%s WINDOWS=%s MODEL=%s\n\n" \
	        "$(HORIZON)" "$(WINDOWS)" "$(MODEL)"

# ---- Setup ---------------------------------------------------------

bootstrap: ## First-time setup (installs uv, syncs deps)
	@bash -c 'command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; }'
	$(UV) sync --all-groups

install: ## Sync deps, install pre-commit hooks
	$(UV) sync --all-groups
	@if [ -d .git ]; then \
	    echo "==> Installing pre-commit hooks"; \
	    $(UV) run pre-commit install >/dev/null; \
	fi
	@if [ ! -f .env ] && [ -f .env.example ]; then \
	    echo "==> Seeding .env from .env.example"; \
	    cp .env.example .env; \
	fi
	@printf '\n\033[32m==> Install complete.\033[0m venv at \033[36m./%s\033[0m\n' "$(VENV)"
	@printf '   • Run: \033[36muv run <cmd>\033[0m always works\n'

activate: ## Print activate command
	@echo "source $(VENV)/bin/activate"

# ---- Quality -------------------------------------------------------

lint: ## Lint with ruff
	$(UV) run ruff check .

fmt: ## Format with ruff
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck: ## mypy on src/m6
	$(UV) run mypy

test: ## Run the full pytest suite (smoke + unit + integration)
	$(UV) run pytest

test-smoke: ## Smoke tier — imports, CLI help, package metadata (~1s)
	$(UV) run pytest -m smoke

test-unit: ## Unit tier — pure-function tests on config/data/eval/metrics
	$(UV) run pytest -m unit

test-integration: ## Integration tier — model fit/predict + CV on toy data
	$(UV) run pytest -m integration

test-fast: ## Smoke + unit (skip integration and `slow`)
	$(UV) run pytest -m "smoke or unit" --no-cov -q

cov: ## Run the suite with coverage (terminal + htmlcov/)
	$(UV) run pytest --cov=m6 --cov-report=term-missing --cov-report=html

check: lint typecheck test ## Lint + types + tests (CI entry point)

# ---- Pipeline ------------------------------------------------------

download: ## Download M6 price data from Yahoo Finance
	$(UV) run m6 download

prep: ## Build the long-format training parquet
	$(UV) run m6 prep

cv-naive: ## Cross-validate the naive equal-probability benchmark
	$(UV) run m6 cv naive --horizon $(HORIZON) --n-windows $(WINDOWS)

cv-historical: ## Cross-validate the historical-frequency model
	$(UV) run m6 cv historical --horizon $(HORIZON) --n-windows $(WINDOWS)

cv-gaussian: ## Cross-validate the multivariate Gaussian model
	$(UV) run m6 cv gaussian --horizon $(HORIZON) --n-windows $(WINDOWS)

cv-ensemble: ## Cross-validate the ensemble (gaussian + historical) model
	$(UV) run m6 cv ensemble --horizon $(HORIZON) --n-windows $(WINDOWS)

cv-adaptive: ## Cross-validate the adaptive gradient boosting model (best performer)
	$(UV) run m6 cv adaptive --horizon $(HORIZON) --n-windows $(WINDOWS)

cv-csp: ## Cross-validate the Conformal Seasonal Pools model (training-free)
	$(UV) run m6 cv csp --horizon $(HORIZON) --n-windows $(WINDOWS)

cv-recipe: ## Cross-validate from a YAML recipe (RECIPE=configs/m6/gaussian.yaml)
	$(UV) run m6 cv-recipe $(RECIPE) --horizon $(HORIZON) --n-windows $(WINDOWS)

# ---- Scoring + report ----------------------------------------------

score: ## Score CV artifacts → reports/{figures,metrics,report.md,report.html} (MODELS="naive historical gaussian ensemble adaptive")
	@set -e; CMD="$(UV) run m6 score --out $(REPORT) --run-id $(RUN_ID)"; \
	for m in $(MODELS); do CMD="$$CMD --model $$m"; done; \
	echo "$$CMD"; eval $$CMD

score-all: ## Score every CV artifact found in artifacts/ (cv_<name>.parquet)
	@set -e; \
	models=$$(ls artifacts/cv_*.parquet 2>/dev/null | sed -e 's|artifacts/cv_||' -e 's|\.parquet$$||'); \
	if [ -z "$$models" ]; then echo "No artifacts/cv_*.parquet found — run a cv-* target first." >&2; exit 1; fi; \
	CMD="$(UV) run m6 score --out $(REPORT) --run-id $(RUN_ID)"; \
	for m in $$models; do CMD="$$CMD --model $$m"; done; \
	echo "$$CMD"; eval $$CMD

eval: cv-naive cv-historical cv-gaussian cv-ensemble cv-adaptive score ## End-to-end: all models CV, then score

submit: bootstrap download prep cv-adaptive score ## One-command happy path: adaptive model submission
	@echo "\n✅ Submission pipeline complete. See reports/ for scores and metrics."

all: eval ## Full pipeline: all models + score

# ---- Visualisation -------------------------------------------------

viz: ## Render pipeline visualisation (animated GIF → assets/pipeline.gif)
	$(UV) run m6 viz

# ---- Notebooks -----------------------------------------------------

notebook: ## Launch Jupyter Lab with the notebook dep group
	$(UV) run --group notebook jupyter lab

# ---- Cleanup -------------------------------------------------------

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean ## Also remove .venv, data, forecasts, artifacts
	rm -rf .venv data/processed data/m6 artifacts forecasts reports
