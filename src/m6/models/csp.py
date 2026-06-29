"""Conformal Seasonal Pools (CSP) model for M6 quintile forecasting.

Fits CSP on each asset's h-day forward return series. The point forecast
(center of the conformal distribution) is the historical mean forward return,
which is the best available unbiased signal. Temperature scaling calibrates
the cross-sectional confidence to match the true predictive power.

Result: RPS ~0.161 with temperature=5.0 (slightly above 0.160 naive).
The historical mean forward return has near-zero predictive power for 1-month
returns (ρ ≈ 0.02), so CSP cannot meaningfully differentiate assets and
converges to naive performance.

Reference: https://github.com/valeman/csp-forecaster
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from csp_forecaster import ConformalSeasonalPool

from m6.config import SETTINGS


def _temperature_scale(
    probs: np.ndarray,
    temperature: float = 5.0,
) -> np.ndarray:
    """Apply temperature scaling to reduce overconfidence.

    p_i = exp(log(raw_p_i) / T) / Z preserves ranking but compresses
    the probability distribution toward uniform as T increases.
    T → ∞  gives uniform [0.2, 0.2, 0.2, 0.2, 0.2].
    """
    if temperature <= 1.0:
        return probs
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    scaled = np.exp(logits / temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def predict_csp(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    h: int = 20,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    n_samples: int = 10000,
    temperature: float = 5.0,
) -> pd.DataFrame:
    """Predict quintile probabilities using CSP with temperature scaling.

    Args:
        df: Full long frame with price history.
        cutoff: Forecast origin (trading date).
        h: Forecast horizon in trading days.
        id_col: Column identifying each asset.
        time_col: Column with trading dates.
        target_col: Unused — present for interface compatibility.
        n_samples: Number of Monte Carlo draws per asset.
        temperature: Temperature scaling factor (higher = more uniform).

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
        prices = asset["price"].to_numpy(dtype=np.float64)

        if len(prices) < h + 5:
            continue

        fwd = prices[h:] / prices[:-h] - 1.0

        if len(fwd) < 10:
            continue

        mean_fwd = float(fwd.mean())

        csp = ConformalSeasonalPool(
            adaptive=True,
            mode="fast",
            residual_mode="h_step",
            orientation=False,
            random_state=SETTINGS.seed,
        )
        csp.fit(fwd, seasonal_period=1)
        csp.history[-1] = mean_fwd
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

    probs = _temperature_scale(probs, temperature=temperature)

    rows = []
    for idx, ticker in enumerate(tickers_list):
        row = {id_col: ticker, time_col: cutoff}
        for qj in range(5):
            row[f"p{qj + 1}"] = probs[idx, qj]
        rows.append(row)
    return pd.DataFrame(rows)
