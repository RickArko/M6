"""CSP marginals + Gaussian copula hybrid.

Uses CSP to model each asset's marginal forward-return distribution
(flexible, non-Gaussian tails), and a Gaussian copula to capture
cross-asset dependence. The copula correlation is estimated from
historical daily returns with Ledoit-Wolf shrinkage (same approach
as the Gaussian model).

This addresses the fundamental weakness of pure CSP: univariate models
cannot differentiate assets because the historical mean provides
near-zero cross-sectional signal. The copula provides the cross-asset
correlation structure that drives the MV Gaussian model's success,
while CSP marginals handle fat tails and asymmetry better than the
Gaussian assumption.

Key insight: the Gaussian copula's correlation, when all marginals
have nearly identical means, *reduces* CS differentiation (correlated
draws move together). To avoid this, we use the Gaussian model's
mean-and-covariance joint distribution to generate the correlated
uniforms, then map through the CSP empirical CDF to preserve the
non-Gaussian marginal shape.

Algorithm:
1. Fit CSP for each asset → sorted samples (empirical inverse CDF)
2. Compute mean vector and shrinkage covariance from daily returns (Gaussian)
3. Draw MVN samples from N(µ_h, Σ_h)
4. For each draw, compute standard-normal quantiles → uniform
5. Map uniforms through CSP inverse CDF → CSP-marginal returns
6. Rank cross-sectionally → quintile probabilities
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from csp_forecaster import ConformalSeasonalPool
from scipy.stats import norm

from m6.config import SETTINGS


def _estimate_moments(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    tickers_list: list[str],
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    h: int = 20,
    shrinkage: float = 0.3,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimate h-period mean vector and shrinkage covariance from daily returns.

    Returns (mean_vec, cov_mat) for the subset of tickers in tickers_list,
    or None if insufficient data.
    """
    hist = df[df[time_col] <= cutoff].copy()
    if hist.empty:
        return None

    pivot = hist.pivot_table(index=time_col, columns=id_col, values=target_col)
    pivot = pivot.fillna(0.0)

    available = [t for t in tickers_list if t in pivot.columns]
    if len(available) < 2:
        return None

    pivot = pivot[available]
    n = len(available)

    mean_daily = pivot.mean().to_numpy()
    cov_daily = pivot.cov().to_numpy()

    # Shrinkage
    target = np.eye(n) * np.trace(cov_daily) / n
    cov_mat = (1 - shrinkage) * cov_daily + shrinkage * target

    # Ensure positive-definite
    eigvals = np.linalg.eigvalsh(cov_mat)
    if eigvals.min() < 1e-8:
        cov_mat += np.eye(n) * (1e-8 - eigvals.min())

    # Scale to h-period
    mean_vec = mean_daily * h
    cov_mat = cov_mat * h

    return mean_vec, cov_mat


def predict_csp_copula(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    h: int = 20,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    n_samples: int = 10000,
    shrinkage: float = 0.3,
) -> pd.DataFrame:
    """Predict quintile probabilities using CSP marginals + Gaussian copula.

    Args:
        df: Full long frame with price history and daily returns.
        cutoff: Forecast origin (trading date).
        h: Forecast horizon in trading days.
        id_col: Column identifying each asset.
        time_col: Column with trading dates.
        target_col: Column with daily returns.
        n_samples: Number of Monte Carlo draws from the joint distribution.
        shrinkage: Covariance shrinkage strength.

    Returns:
        DataFrame with columns unique_id, ds, p1-p5, or empty if unable.
    """
    tickers = df[id_col].unique()
    history = df[df[time_col] <= cutoff].copy()

    if history.empty:
        return pd.DataFrame()

    # --- Step 1: Fit CSP marginals (sorted samples = empirical inverse CDF) ---
    csp_samples: dict[str, np.ndarray] = {}
    n_csp = max(n_samples, 2000)

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
        result = csp.predict(H=1, n_samples=n_csp)

        csp_samples[ticker] = np.sort(result.samples[0])

    if not csp_samples:
        return pd.DataFrame()

    tickers_list = list(csp_samples.keys())
    n_assets = len(tickers_list)

    # --- Step 2: Gaussian moments (mean + covariance) from daily returns ------
    moments = _estimate_moments(
        df,
        cutoff,
        tickers_list,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        h=h,
        shrinkage=shrinkage,
    )
    if moments is None:
        return pd.DataFrame()

    mean_vec, cov_mat = moments

    # Map from available order back to tickers_list order
    pivot = history.pivot_table(index=time_col, columns=id_col, values=target_col)
    available = [t for t in tickers_list if t in pivot.columns]
    avail_mask = [t in available for t in tickers_list]

    # --- Step 3: Draw from Gaussian joint distribution ------------------------
    rng = np.random.default_rng(SETTINGS.seed)
    Z = rng.multivariate_normal(mean_vec, cov_mat, size=n_samples)

    # For assets not in the covariance matrix (shouldn't happen), use independent
    # draws from their CSP marginal

    # --- Step 4: Transform through CSP marginals via probability integral -----
    quintile_counts = np.zeros((n_assets, 5), dtype=np.float64)

    for i in range(n_samples):
        returns = np.empty(n_assets, dtype=np.float64)
        z = Z[i]  # shape (n_avail,)
        avail_counter = 0
        for j in range(n_assets):
            sorted_s = csp_samples[tickers_list[j]]
            n_s = len(sorted_s)
            if avail_mask[j]:
                std = np.sqrt(cov_mat[avail_counter, avail_counter])
                u = norm.cdf((z[avail_counter] - mean_vec[avail_counter]) / max(std, 1e-12))
                idx_float = u * (n_s - 1)
                idx_low = int(np.floor(idx_float))
                idx_high = min(idx_low + 1, n_s - 1)
                frac = max(0.0, min(1.0, idx_float - idx_low))
                returns[j] = sorted_s[idx_low] * (1.0 - frac) + sorted_s[idx_high] * frac
                avail_counter += 1
            else:
                idx_i = int(rng.integers(0, n_s))
                returns[j] = sorted_s[idx_i]

        ranks = np.argsort(np.argsort(returns))
        q_idx = np.floor(ranks / n_assets * 5).astype(int).clip(0, 4)
        quintile_counts[np.arange(n_assets), q_idx] += 1.0

    probs = quintile_counts / n_samples

    uniform = np.full((n_assets, 5), 0.2, dtype=np.float64)
    probs = (1 - SETTINGS.calib_strength) * probs + SETTINGS.calib_strength * uniform

    rows = []
    for idx, ticker in enumerate(tickers_list):
        row = {id_col: ticker, time_col: cutoff}
        for qj in range(5):
            row[f"p{qj + 1}"] = probs[idx, qj]
        rows.append(row)
    return pd.DataFrame(rows)
