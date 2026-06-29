"""Conformal Seasonal Pools (CSP) model for M6 quintile forecasting.

Fits a training-free CSP forecaster on each asset's daily return series,
then uses Monte Carlo simulation to map the 1-step-ahead predictive
distributions into cross-sectional quintile probabilities.

Key insight: under the random walk hypothesis, the cross-sectional ordering
of tomorrow's daily returns has the same ranking as the cumulative h-day
total return (since scaling by horizon is monotonic). CSP captures the
uncertainty around which assets will outperform tomorrow, which translates
directly into uncertainty about the h-day quintile ranking.

Reference: https://github.com/valeman/csp-forecaster
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from csp_forecaster import ConformalSeasonalPool

from m6.config import SETTINGS


def _calibrate_predictions(
    probs: np.ndarray,
    calib_strength: float = 0.15,
) -> np.ndarray:
    if calib_strength <= 0:
        return probs
    uniform = np.full(probs.shape, 0.2, dtype=np.float64)
    return (1 - calib_strength) * probs + calib_strength * uniform


def predict_csp(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    h: int = 20,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    n_samples: int = 5000,
) -> pd.DataFrame:
    """Predict quintile probabilities using Conformal Seasonal Pools (CSP).

    For each asset, fits CSP on the daily return series and predicts the
    1-step-ahead (next-day) return distribution. Cross-sectional quintile
    probabilities are estimated via Monte Carlo: for each draw, all assets
    are ranked by their sampled next-day return and assigned to quintiles,
    then averaged across draws.

    Args:
        df: Full long frame with daily return history.
        cutoff: Forecast origin (trading date).
        h: Forecast horizon in trading days (used for cross-sectional
           quintile mapping; under the random walk the daily return
           ordering is a proxy for the h-day return ordering).
        id_col: Column identifying each asset.
        time_col: Column with trading dates.
        target_col: Column with daily returns.
        n_samples: Number of Monte Carlo draws per asset.

    Returns:
        DataFrame with columns unique_id, ds, p1-p5, or empty if unable.
    """
    tickers = df[id_col].unique()
    history = df[df[time_col] <= cutoff].copy()

    if history.empty:
        return pd.DataFrame()

    asset_samples: dict[str, np.ndarray] = {}

    for ticker in tickers:
        asset = history[history[id_col] == ticker].sort_values(time_col)
        daily_ret = asset[target_col].to_numpy(dtype=np.float64)

        if len(daily_ret) < 10:
            continue

        csp = ConformalSeasonalPool(
            adaptive=True,
            mode="fast",
            residual_mode="h_step",
            orientation=False,
            random_state=SETTINGS.seed,
        )
        csp.fit(daily_ret, seasonal_period=1)
        result = csp.predict(H=1, n_samples=n_samples)
        asset_samples[ticker] = result.samples[0]

    if not asset_samples:
        return pd.DataFrame()

    tickers_list = list(asset_samples.keys())
    n_assets = len(tickers_list)

    quintile_counts = np.zeros((n_assets, 5), dtype=np.float64)
    for i in range(n_samples):
        draw = np.array([asset_samples[t][i] for t in tickers_list], dtype=np.float64)
        ranks = np.argsort(np.argsort(draw))
        q_idx = np.floor(ranks / n_assets * 5).astype(int).clip(0, 4)
        quintile_counts[np.arange(n_assets), q_idx] += 1.0

    probs = quintile_counts / quintile_counts.sum(axis=1, keepdims=True)

    probs = _calibrate_predictions(probs, calib_strength=SETTINGS.calib_strength)

    rows = []
    for idx, ticker in enumerate(tickers_list):
        row = {id_col: ticker, time_col: cutoff}
        for qj in range(5):
            row[f"p{qj + 1}"] = probs[idx, qj]
        rows.append(row)
    return pd.DataFrame(rows)
