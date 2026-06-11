"""Naive equal-probability benchmark — predicts 0.2 for every quintile.

This matches the M6 competition baseline. Its RPS is 0.16, and models
must beat it to demonstrate forecasting skill.
"""

from __future__ import annotations

import pandas as pd


def predict_naive(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    h: int = 20,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pd.DataFrame:
    """Predict equal probabilities (0.2) for all five quintiles.

    Args:
        df: Full long frame with price history up to and including cutoff.
        cutoff: Forecast origin (trading date).
        h: Forecast horizon in trading days.
        id_col: Column identifying each asset.
        time_col: Column with trading dates.

    Returns:
        DataFrame with columns unique_id, ds, p1-p5
        where ``ds`` is the cutoff date and p1-p5 all equal 0.2.
    """
    assets = df.loc[df[time_col] <= cutoff, id_col].unique()
    result = pd.DataFrame(
        {
            id_col: assets,
            time_col: cutoff,
            "p1": 0.2,
            "p2": 0.2,
            "p3": 0.2,
            "p4": 0.2,
            "p5": 0.2,
        }
    )
    return result
