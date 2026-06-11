"""Multivariate Gaussian model — estimate mean returns and covariance,
Monte Carlo simulate forward returns, and compute quintile probabilities.

Improvements over naive:
- Shrinkage covariance for numerical stability
- Probability calibration (shrink toward uniform to reduce overconfidence)
- Volatility-scaled momentum tilt
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
    """Estimate mean vector, covariance (LW), and ticker order.

    Returns (tickers, mean_vec, cov_mat) or None if insufficient data.
    """
    hist = df[df[time_col] <= cutoff].copy()
    if hist.empty:
        return None

    pivot = hist.pivot_table(index=time_col, columns=id_col, values=target_col)
    pivot = pivot.fillna(0.0)

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


def _vol_mom_signal(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    mom_window: int = 20,
    vol_window: int = 60,
) -> np.ndarray:
    """Volatility-scaled momentum (return / vol).

    :meth:`_compute_momentum` returns total return over ``mom_window`` days.
    The signal is divided by volatility (std of daily returns over ``vol_window``).
    This captures risk-adjusted momentum — high momentum with low volatility is
    more informative.
    """
    start = cutoff - pd.Timedelta(days=mom_window * 3)
    recent = df[(df[time_col] > start) & (df[time_col] <= cutoff)]
    momentum = recent.groupby(id_col)[target_col].sum()
    vol = recent.groupby(id_col)[target_col].std().clip(lower=1e-8)
    signal = momentum / vol

    tickers = df[id_col].unique()
    result = np.array([signal.get(t, 0.0) for t in tickers], dtype=np.float64)
    return result


def _calibrate_predictions(
    probs: np.ndarray,
    calib_strength: float = 0.15,
) -> np.ndarray:
    """Shrink predicted probabilities toward the uniform baseline.

    Mixes model predictions with uniform probabilities to reduce
    overconfidence.  ``calib_strength=0`` is pure model, ``calib_strength=1``
    is pure uniform (naive).
    """
    if calib_strength <= 0:
        return probs
    uniform = np.full(probs.shape, 0.2, dtype=np.float64)
    return (1 - calib_strength) * probs + calib_strength * uniform


def _tilt_probs(
    probs: np.ndarray,
    signal: np.ndarray,
    strength: float = 0.03,
) -> np.ndarray:
    """Gently tilt probabilities based on a signal.

    High-signal assets get a small probability shift toward higher quintiles.
    Low-signal assets shift toward lower quintiles.
    """
    n_assets, n_q = probs.shape
    q_center = (n_q + 1) / 2.0
    q_weights = np.arange(1, n_q + 1, dtype=np.float64)

    ranks = np.argsort(np.argsort(signal)).astype(np.float64)
    ranks_normalized = (ranks / (n_assets - 1) - 0.5) * 2

    shift = strength * ranks_normalized[:, None] * (q_weights - q_center) / (n_q - 1)
    tilted = probs + shift
    tilted = np.clip(tilted, 0, None)
    tilted /= tilted.sum(axis=1, keepdims=True)
    return tilted


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

    h_mean = mean_vec * h
    h_cov = cov_mat * h

    rng = np.random.default_rng(SETTINGS.seed)
    samples = rng.multivariate_normal(h_mean, h_cov, size=n_sim)

    quintile_counts = np.zeros((n_assets, 5), dtype=np.float64)
    for i in range(n_sim):
        ranks = np.argsort(np.argsort(samples[i]))
        q_idx = np.floor(ranks / n_assets * 5).astype(int).clip(0, 4)
        quintile_counts[np.arange(n_assets), q_idx] += 1.0

    probs = quintile_counts / quintile_counts.sum(axis=1, keepdims=True)

    probs = _calibrate_predictions(probs, calib_strength=SETTINGS.calib_strength)

    signal = _vol_mom_signal(
        df,
        cutoff,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        mom_window=SETTINGS.mom_window,
    )
    probs = _tilt_probs(probs, signal, strength=SETTINGS.tilt_strength)

    rows = []
    for idx, ticker in enumerate(tickers):
        row = {"unique_id": ticker, time_col: cutoff}
        for qj in range(5):
            row[f"p{qj + 1}"] = probs[idx, qj]
        rows.append(row)
    return pd.DataFrame(rows)
