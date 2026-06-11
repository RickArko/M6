"""Ensemble model — averages probability predictions from multiple models.

Simple averaging (equal weights) of gaussian + historical predictions.
"""

from __future__ import annotations

import pandas as pd

from m6.models.gaussian import predict_gaussian
from m6.models.historical import predict_historical


def predict_ensemble(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    h: int = 20,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> pd.DataFrame:
    """Predict quintile probabilities by averaging gaussian + historical.

    Args:
        df: Full long frame with daily returns and quintile assignments.
        cutoff: Forecast origin (trading date).
        h: Forecast horizon in trading days.

    Returns:
        DataFrame with columns unique_id, ds, p1-p5
    """
    gauss = predict_gaussian(df, cutoff, h, id_col=id_col, time_col=time_col)
    hist = predict_historical(df, cutoff, h, id_col=id_col, time_col=time_col)

    if gauss.empty and hist.empty:
        return pd.DataFrame()
    if gauss.empty:
        return hist
    if hist.empty:
        return gauss

    merged = gauss.merge(hist, on=[id_col, time_col], suffixes=("_g", "_h"))
    rows = []
    for _, row in merged.iterrows():
        entry = {id_col: row[id_col], time_col: cutoff}
        for qi in range(1, 6):
            entry[f"p{qi}"] = (row[f"p{qi}_g"] + row[f"p{qi}_h"]) / 2.0
        rows.append(entry)
    return pd.DataFrame(rows)
