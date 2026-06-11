"""Financial feature engineering for the M6 asset universe.

Provides building blocks for enhanced probability models:
- Rolling volatility estimates
- Momentum signals
- Volume-based features (if volume data available)
- Correlation with market factors (SPY, sector ETFs)
"""

from __future__ import annotations

import pandas as pd


def add_rolling_features(
    df: pd.DataFrame,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    windows: tuple[int, ...] = (5, 21, 63, 252),
) -> pd.DataFrame:
    result = df.copy()
    result = result.sort_values([id_col, time_col])
    eps = 1e-9

    for w in windows:
        rolled = result.groupby(id_col, observed=True)[target_col]
        result[f"rmean_{w}"] = rolled.transform(
            lambda s, ww=w: s.rolling(ww, min_periods=max(5, ww // 4)).mean()
        )
        result[f"rstd_{w}"] = rolled.transform(
            lambda s, ww=w: s.rolling(ww, min_periods=max(5, ww // 4)).std()
        )
        result[f"rsharpe_{w}"] = result[f"rmean_{w}"] / (result[f"rstd_{w}"] + eps)

    return result


def add_momentum_features(
    df: pd.DataFrame,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    price_col: str = "price",
    windows: tuple[int, ...] = (21, 63, 126, 252),
) -> pd.DataFrame:
    """Add momentum (lagged total return) features for each asset."""
    result = df.copy()
    result = result.sort_values([id_col, time_col])

    for w in windows:
        result[f"mom_{w}"] = result.groupby(id_col, observed=True)[price_col].transform(
            lambda s, ww=w: s.pct_change(ww)
        )
    return result


def add_volatility_features(
    df: pd.DataFrame,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    windows: tuple[int, ...] = (21, 63),
) -> pd.DataFrame:
    """Add volatility regime features: rank within asset universe and z-score."""
    result = df.copy()

    for w in windows:
        vol_col = f"rstd_{w}"
        if vol_col not in result.columns:
            continue
        result[f"vol_rank_{w}"] = result.groupby(time_col, observed=True)[vol_col].rank(pct=True)
        result[f"vol_zscore_{w}"] = result.groupby(id_col, observed=True)[vol_col].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-9)
        )
    return result


def build_feature_frame(
    df: pd.DataFrame,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    price_col: str = "price",
) -> pd.DataFrame:
    """Build the full feature-augmented frame.

    Adds rolling stats, momentum, and volatility regime features.
    """
    result = add_rolling_features(df, id_col=id_col, time_col=time_col, target_col=target_col)
    result = add_momentum_features(result, id_col=id_col, time_col=time_col, price_col=price_col)
    result = add_volatility_features(result, id_col=id_col, time_col=time_col, target_col=target_col)
    return result
