"""Shared fixtures and per-directory markers.

Layout:
    tests/smoke/        marker: smoke
    tests/unit/         marker: unit
    tests/integration/  marker: integration
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_TIER_MARKERS = {
    "smoke": pytest.mark.smoke,
    "unit": pytest.mark.unit,
    "integration": pytest.mark.integration,
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    tests_root = Path(__file__).parent
    for item in items:
        try:
            rel = Path(item.fspath).relative_to(tests_root)
        except ValueError:
            continue
        tier = rel.parts[0] if rel.parts else ""
        marker = _TIER_MARKERS.get(tier)
        if marker is not None:
            item.add_marker(marker)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def toy_returns(rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic daily returns for 6 assets × 500 trading days.

    Creates a realistic test set with:
      - 3 stocks with N(0.0005, 0.015) daily returns
      - 3 ETFs with N(0.0002, 0.01) daily returns
    """
    n_days = 500
    tickers = ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ", "IWM"]
    categories = ["stock", "stock", "stock", "us_etf", "us_etf", "us_etf"]

    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    rows = []
    for idx, ticker in enumerate(tickers):
        mu = 0.0005 if idx < 3 else 0.0002
        sigma = 0.015 if idx < 3 else 0.01
        returns = rng.normal(mu, sigma, n_days)
        price = 100.0 * np.exp(np.cumsum(returns))
        for d, ret, p in zip(dates, returns, price):
            rows.append(
                {
                    "unique_id": ticker,
                    "ds": d,
                    "y": ret.astype(np.float32),
                    "price": p,
                    "category": categories[idx],
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def toy_long(toy_returns: pd.DataFrame) -> pd.DataFrame:
    """Full long frame with forward returns and quintiles."""
    from m6.data import assign_quintiles, compute_forward_returns

    df = compute_forward_returns(toy_returns, horizon=20)
    df = assign_quintiles(df)
    return df.dropna(subset=["quintile"]).reset_index(drop=True)


@pytest.fixture()
def toy_cv(toy_long: pd.DataFrame) -> pd.DataFrame:
    """Synthetic CV frame with naive and perfect predictions."""
    from m6.data import make_cv_cutoffs

    cutoffs = make_cv_cutoffs(toy_long, n_windows=2, horizon=20)
    folds = []
    for cutoff in cutoffs:
        fold = toy_long[toy_long["ds"] == cutoff].copy()
        if fold.empty:
            continue
        fold["cutoff"] = cutoff
        fold["naive_p1"] = 0.2
        fold["naive_p2"] = 0.2
        fold["naive_p3"] = 0.2
        fold["naive_p4"] = 0.2
        fold["naive_p5"] = 0.2

        q = fold["quintile"].round().astype(int).clip(1, 5)
        for i in range(1, 6):
            fold[f"perfect_p{i}"] = (q == i).astype(float)
        folds.append(fold)

    if not folds:
        return pd.DataFrame()

    result = pd.concat(folds, ignore_index=True)
    result = result.drop(columns=["y"], errors="ignore")
    result = result.rename(columns={"quintile": "y"})
    return result
