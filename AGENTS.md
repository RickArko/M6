# M6 Forecasting Uncertainty

## Overview
M6 forecasting competition focused on probabilistic (quintile) forecasting for 100 financial assets. This project provides a reproducible pipeline for downloading data, running cross-validation, and scoring probability forecasts using the Ranked Probability Score (RPS).

## Quick Start
```bash
make bootstrap     # Install uv + sync deps
make download      # Download 100 M6 assets from Yahoo Finance
make prep          # Build long frame with forward returns + quintiles
make cv-adaptive   # CV for adaptive gradient boosting model (best performer)
make score         # Score all CV artifacts
```

## Key Architecture
- `src/m6/` — main package (config, data, evaluation, scoring, models, cv, cli)
- `tests/` — smoke/unit/integration tiers via pytest markers
- `configs/m6/` — YAML recipes for model definitions
- `Makefile` — single entrypoint wrapping `uv run m6 ...`

## Evaluation Metric
**RPS (Ranked Probability Score)**: strictly proper scoring rule for ordered categorical outcomes (5 quintiles). RPS = 0.16 for naive equal-probability forecast, lower is better. Top solutions achieved ~0.1564-0.1565.

## Models
- `naive` — equal probability 0.2 (M6 baseline)
- `historical` — empirical quintile frequencies per asset
- `gaussian` — multivariate normal with shrinkage covariance + Monte Carlo
- `ensemble` — simple average of gaussian + historical
- `adaptive` — cross-sectional gradient boosting with financial features, calibration, and historical blending (best performer)

## Development
```bash
make lint        # ruff check
make fmt         # ruff format
make typecheck   # mypy
make test        # pytest (all tiers)
make test-fast   # smoke + unit only
```

## Verification
Always run `make check` (lint + typecheck + test) before committing.

## CI
See `.github/workflows/ci.yml` for the GitHub Actions workflow.
