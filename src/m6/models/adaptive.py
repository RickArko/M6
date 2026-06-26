"""Adaptive gradient boosting model — cross-sectional ML with financial features.

Key ideas:
- Cross-sectional features (momentum, vol, sharpe) per asset
- Conservative gradient boosting (shallow trees, low learning rate)
- Strong shrinkage toward uniform to avoid overconfidence
- Blend with historical frequencies for robustness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from m6.config import SETTINGS
from m6.features import build_feature_frame
from m6.models.historical import _compute_quintile_frequencies


def _add_extra_features(
    df: pd.DataFrame,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pd.DataFrame:
    """Add cross-sectional and regime features on top of the base feature frame."""
    result = df.copy()

    # Reversal: short-term vs medium-term momentum
    if "rmean_5" in result.columns and "rmean_21" in result.columns:
        result["reversal_5_21"] = result["rmean_5"] / (result["rmean_21"].abs() + 1e-9)

    # Cross-sectional momentum ranks
    if "mom_21" in result.columns:
        result["mom_rank_21"] = result.groupby(time_col, observed=True)["mom_21"].rank(pct=True)
    if "mom_63" in result.columns:
        result["mom_rank_63"] = result.groupby(time_col, observed=True)["mom_63"].rank(pct=True)

    # Volatility regime: market-wide average vol per date
    if "rstd_21" in result.columns:
        result["vol_regime"] = result.groupby(time_col, observed=True)["rstd_21"].transform("mean")

    return result


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return stable list of feature column names."""
    base = [
        "rmean_5",
        "rmean_21",
        "rmean_63",
        "rmean_252",
        "rstd_5",
        "rstd_21",
        "rstd_63",
        "rstd_252",
        "rsharpe_5",
        "rsharpe_21",
        "rsharpe_63",
        "rsharpe_252",
        "mom_21",
        "mom_63",
        "mom_126",
        "mom_252",
        "vol_rank_21",
        "vol_zscore_21",
        "vol_rank_63",
        "vol_zscore_63",
    ]
    extra = [
        "reversal_5_21",
        "mom_rank_21",
        "mom_rank_63",
        "vol_regime",
    ]
    available = [c for c in base + extra if c in df.columns]
    return available


def _calibrate_predictions(
    probs: np.ndarray,
    calib_strength: float = 0.2,
) -> np.ndarray:
    """Shrink predicted probabilities toward the uniform baseline."""
    if calib_strength <= 0:
        return probs
    uniform = np.full(probs.shape, 0.2, dtype=np.float64)
    return (1 - calib_strength) * probs + calib_strength * uniform


def _blend_with_historical(
    ml_probs: np.ndarray,
    hist_df: pd.DataFrame,
    tickers: np.ndarray,
    alpha: float = 0.65,
) -> np.ndarray:
    """Blend ML probabilities with historical frequencies per asset."""
    result = np.zeros_like(ml_probs)
    for i, ticker in enumerate(tickers):
        row = hist_df[hist_df["unique_id"] == ticker]
        if row.empty:
            result[i] = ml_probs[i]
            continue
        hist_vec = np.array([row[f"p{j}"].iloc[0] for j in range(1, 6)], dtype=np.float64)
        result[i] = alpha * ml_probs[i] + (1 - alpha) * hist_vec
    return result


def predict_adaptive(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    h: int = 20,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pd.DataFrame:
    """Predict quintile probabilities using an adaptive gradient boosting model.

    Args:
        df: Full long frame with daily returns, price, and quintile assignments.
        cutoff: Forecast origin (trading date).
        h: Forecast horizon in trading days (unused directly, but kept for API).
        id_col: Column identifying each asset.
        time_col: Column with trading dates.
        target_col: Column with daily returns (used for feature engineering).

    Returns:
        DataFrame with columns unique_id, ds, p1-p5
    """
    # Build feature frame
    feat_df = build_feature_frame(
        df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        price_col="price",
    )
    feat_df = _add_extra_features(feat_df, id_col=id_col, time_col=time_col, target_col=target_col)
    feature_cols = _get_feature_cols(feat_df)

    if not feature_cols:
        return pd.DataFrame()

    # Training data: all historical dates up to cutoff with valid quintiles
    train = feat_df[(feat_df[time_col] <= cutoff) & feat_df["quintile"].notna()].copy()
    if len(train) < 200:
        return pd.DataFrame()

    # Drop constant or all-NaN features (HistGradientBoostingClassifier requires >=2 distinct values)
    valid_feature_cols = []
    for col in feature_cols:
        n_unique = train[col].nunique(dropna=True)
        if n_unique >= 2:
            valid_feature_cols.append(col)

    if not valid_feature_cols:
        return pd.DataFrame()

    X = train[valid_feature_cols].to_numpy()
    y = train["quintile"].astype(int).to_numpy()

    # Conservative GBM to avoid overfitting
    clf = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.04,
        max_depth=3,
        min_samples_leaf=80,
        random_state=SETTINGS.seed,
    )
    # Limit OpenMP threads to avoid deadlocks in some environments
    with threadpool_limits(limits=1, user_api="openmp"):
        clf.fit(X, y)

    # Prediction data: current cross-section at cutoff
    pred = feat_df[feat_df[time_col] == cutoff].copy()
    if pred.empty:
        return pd.DataFrame()

    # Ensure we have all assets
    tickers = pred[id_col].unique()
    X_pred = pred[valid_feature_cols].to_numpy()
    probs = clf.predict_proba(X_pred)

    # Ensure probability column order matches p1-p5
    classes = clf.classes_
    ordered_probs = np.zeros((len(tickers), 5), dtype=np.float64)
    for qi, cls in enumerate([1, 2, 3, 4, 5]):
        if cls in classes:
            idx = np.where(classes == cls)[0][0]
            ordered_probs[:, qi] = probs[:, idx]
    # Renormalize in case a class was missing
    row_sums = ordered_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    ordered_probs = ordered_probs / row_sums

    # Calibration — strong shrink toward uniform (top M6 trick)
    ordered_probs = _calibrate_predictions(ordered_probs, calib_strength=0.25)

    # Blend with historical frequencies for robustness
    hist = _compute_quintile_frequencies(df, cutoff, id_col=id_col, time_col=time_col, target_col="quintile")
    if not hist.empty:
        ordered_probs = _blend_with_historical(ordered_probs, hist, tickers, alpha=0.6)

    # Build result
    rows = []
    for idx, ticker in enumerate(tickers):
        row = {id_col: ticker, time_col: cutoff}
        for qj in range(5):
            row[f"p{qj + 1}"] = ordered_probs[idx, qj]
        rows.append(row)
    return pd.DataFrame(rows)
