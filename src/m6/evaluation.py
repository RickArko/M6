"""RPS — Ranked Probability Score (M6 official metric).

The Ranked Probability Score (RPS) is a strictly proper scoring rule for
ordered categorical outcomes. For each asset the outcome is a quintile
(1 = worst, 5 = best), encoded as a 5-element one-hot (or fractional)
vector q. The forecast is a 5-element probability vector f (summing to 1).

Formula for a single asset:
    RPS_i = (1/5) * sum_{j=1}^{5} (CDF_q(j) - CDF_f(j))^2
where CDF is the cumulative distribution (prefix sum).

Reference: https://github.com/Mcompetitions/M6-methods
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# RPS of the naive equal-probability benchmark
NAIVE_RPS: float = 0.16

# Known top-3 published RPS scores for reference
PUBLISHED_RPS: dict[str, float] = {
    "Dan (1st)": 0.15645,
    "FinQBoost (2nd)": 0.15648,
    "SebastianR (3rd)": 0.15649,
    "Naive benchmark": NAIVE_RPS,
}

N_QUINTILES = 5


def _quintile_to_onehot(q: float, n: int = N_QUINTILES) -> np.ndarray:
    """Convert a (possibly fractional) quintile to a one-hot or fractional vector."""
    if q == int(q) and 1 <= q <= n:
        oh = np.zeros(n)
        oh[int(q) - 1] = 1.0
        return oh
    lower = int(np.floor(q))
    upper = int(np.ceil(q))
    if lower == upper:
        oh = np.zeros(n)
        oh[lower - 1] = 1.0
        return oh
    ratio = upper - q
    oh = np.zeros(n)
    oh[lower - 1] = ratio
    oh[upper - 1] = 1.0 - ratio
    return oh


def _rps_single(
    actual: np.ndarray,
    forecast: np.ndarray,
    n: int = N_QUINTILES,
) -> float:
    """RPS for a single asset."""
    cdf_act = np.cumsum(actual)
    cdf_fc = np.cumsum(forecast)
    return float(np.mean((cdf_act - cdf_fc) ** 2))


def rps(
    actual_q: pd.Series | pd.DataFrame,
    forecast_probs: pd.DataFrame,
) -> float:
    if forecast_probs is None or forecast_probs.empty or forecast_probs.shape[1] == 0:
        return float("nan")

    q_vals = np.asarray(actual_q).ravel()
    f_arr = forecast_probs.to_numpy()
    n_obs = min(len(q_vals), f_arr.shape[0])
    n_classes = min(forecast_probs.shape[1], N_QUINTILES)

    if n_obs == 0:
        return float("nan")

    total = 0.0
    for i in range(n_obs):
        oh = _quintile_to_onehot(float(q_vals[i]))
        f = f_arr[i, :n_classes]
        if len(oh) > n_classes:
            oh = oh[:n_classes]
        total += _rps_single(oh, f)
    return total / n_obs


@dataclass
class RPSComponents:
    """Cached per-series metadata. For M6 this is lightweight — we
    mainly need it for interface compatibility with the scoring module."""

    n_assets: int
    tickers: pd.Index


def compute_components(long_train: pd.DataFrame) -> RPSComponents:
    """Build RPS components from the training frame.

    For M6, this primarily identifies the asset universe and validates data.
    We don't need per-series weights like WRMSSE because RPS is unweighted.
    """
    tickers = long_train["unique_id"].unique()
    return RPSComponents(
        n_assets=len(tickers),
        tickers=pd.Index(tickers),
    )


def rps_for_models(
    actual: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pd.Series:
    fcst_cols = [c for c in forecasts.columns if c not in (target_col,)]
    merged = actual[[id_col, time_col, target_col]].merge(
        forecasts[fcst_cols],
        on=[id_col, time_col],
        how="inner",
    )
    prob_cols = [c for c in merged.columns if c not in (id_col, time_col, target_col, "cutoff")]
    model_names = sorted(set(c.rsplit("_p", 1)[0] for c in prob_cols if "_p" in c))

    scores: dict[str, float] = {}
    for m in model_names:
        pcols = [f"{m}_p{i}" for i in range(1, N_QUINTILES + 1)]
        missing = [c for c in pcols if c not in merged.columns]
        if missing:
            continue
        merged_fc = merged[pcols].fillna(1.0 / N_QUINTILES)
        scores[m] = rps(merged[target_col], merged_fc)
    return pd.Series(scores, name="rps").sort_values()


def make_submission(
    preds: pd.DataFrame,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    prob_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Pivot a long probability frame into the M6 submission layout.

    Each row is an asset-period with columns: unique_id, ds, p1-p5.
    """
    if prob_cols is None:
        prob_cols = [c for c in preds.columns if c.startswith("p") and c[1:].isdigit()]
    sub = preds[[id_col, time_col, *prob_cols]].copy()
    return sub.sort_values([id_col, time_col]).reset_index(drop=True)
