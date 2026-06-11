"""Multivariate Gaussian model — estimate mean returns and covariance,
Monte Carlo simulate forward returns, and compute quintile probabilities.

This is the approach used by the 1st-place forecasting team (Dan) and
the 3rd-place team (SebastianR). The key steps:

1. Estimate mean vector and covariance matrix of daily returns up to cutoff.
2. Apply shrinkage to the covariance matrix for numerical stability.
3. Monte Carlo sample forward-horizon return scenarios.
4. For each scenario, rank all assets and assign quintiles.
5. Average quintile assignments across scenarios → probability vector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from m6.config import SETTINGS


def _estimate_moments(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    shrinkage: float | None = None,
    min_periods: int = 252,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Estate mean vector, covariance, and ticker order from daily returns.

    Returns (tickers, mean_vec, cov_mat) or None if insufficient data.
    """
    hist = df[df[time_col] <= cutoff].copy()
    if hist.empty:
        return None

    pivot = hist.pivot_table(index=time_col, columns=id_col, values=target_col)
    pivot = pivot.dropna(how="any", axis=1)

    if pivot.shape[1] < 2:
        return None
    if pivot.shape[0] < min_periods:
        return None

    tickers = pivot.columns.to_numpy()
    mean_vec = pivot.mean().to_numpy()
    cov_mat = pivot.cov().to_numpy()

    shr = shrinkage if shrinkage is not None else SETTINGS.covariance_shrinkage
    target = np.diag(np.diag(cov_mat))
    cov_mat = (1 - shr) * cov_mat + shr * target

    return tickers, mean_vec, cov_mat


def predict_gaussian(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    h: int = 20,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    n_simulations: int | None = None,
    shrinkage: float | None = None,
) -> pd.DataFrame:
    """Predict quintile probabilities using a multivariate Gaussian model.

    Args:
        df: Full long frame with daily returns.
        cutoff: Forecast origin (trading date).
        h: Forecast horizon in trading days.
        id_col: Column identifying each asset.
        time_col: Column with trading dates.
        target_col: Column with daily returns.
        n_simulations: Number of Monte Carlo draws.
        shrinkage: Covariance shrinkage intensity (0 = sample, 1 = diagonal).

    Returns:
        DataFrame with columns unique_id, ds, p1-p5
    """
    n_sim = n_simulations or SETTINGS.n_monte_carlo
    result = _estimate_moments(
        df,
        cutoff,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        shrinkage=shrinkage,
    )
    if result is None:
        return pd.DataFrame()

    tickers, mean_vec, cov_mat = result
    n_assets = len(tickers)

    daily_mean = mean_vec
    daily_cov = cov_mat
    h_mean = daily_mean * h
    h_cov = daily_cov * h

    rng = np.random.default_rng(SETTINGS.seed)
    samples = rng.multivariate_normal(h_mean, h_cov, size=n_sim)

    quintile_counts = np.zeros((n_assets, 5), dtype=np.float64)
    for i in range(n_sim):
        ranks = np.argsort(np.argsort(samples[i]))
        q_idx = np.floor(ranks / n_assets * 5).astype(int).clip(0, 4)
        quintile_counts[np.arange(n_assets), q_idx] += 1.0

    probs = quintile_counts / quintile_counts.sum(axis=1, keepdims=True)

    rows = []
    for idx, ticker in enumerate(tickers):
        row = {"unique_id": ticker, time_col: cutoff}
        for qj in range(5):
            row[f"p{qj + 1}"] = probs[idx, qj]
        rows.append(row)
    return pd.DataFrame(rows)
