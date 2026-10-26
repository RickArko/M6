# M6 Forecasting Uncertainty

[![CI](https://github.com/RickArko/M6/actions/workflows/ci.yml/badge.svg)](https://github.com/RickArko/M6/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Reproducible pipeline for the [M6 Forecasting Competition](https://www.unic.ac.cy/iff/research/forecasting/m-competitions/m6/):
probabilistic quintile forecasting for 100 financial assets, scored by **Ranked Probability Score (RPS)**.

- **Naive benchmark:** RPS = 0.16000
- **Top published:** RPS ≈ 0.15645
- **Best model here:** adaptive gradient boosting + calibration → ~0.1565–0.1575

---

## Happy Path

```bash
make submit
```

Bootstraps the environment, downloads data, builds the training frame, runs the adaptive model, and scores the result.

For the full pipeline (all models including CSP variants):

```bash
make all
```

Core baselines only (no CSP):

```bash
make eval
```

---

## Quick Reference

```bash
make bootstrap      # Install uv + sync deps
make download       # Download 100 M6 assets from Yahoo Finance
make prep           # Build long-format training frame
make cv-adaptive    # Run adaptive model CV (best performer)
make score          # Score available CV artifacts (auto-discovers)
make check          # lint + typecheck + test (CI entry point)
```

`make score` scores whatever `artifacts/cv_*.parquet` files exist. Pin a subset with `make score MODELS="naive adaptive"`.

---

## Architecture

- `src/m6/models/` — `naive`, `historical`, `gaussian`, `ensemble`, `adaptive`, `csp`, `csp_copula`
- `configs/m6/` — YAML recipes
- `Makefile` — canonical entrypoint wrapping `uv run m6 ...`

---

## License

MIT
