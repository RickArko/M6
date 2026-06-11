"""Historical-frequency model — predicts quintile probabilities from past frequencies.

For each asset, the probability of landing in each quintile is estimated as
the historical frequency observed prior to the cutoff date. Recent observations
are weighted more heavily (EWMA decay).

A pooled variant uses all-asset frequencies when per-asset data is sparse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from m6.config import SETTINGS


def _compute_quintile_frequencies(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "quintile",
    use_pooled: bool = True,
    halflife: int | None = None,
) -> pd.DataFrame:
    """Compute per-asset (and optionally pooled) EWMA-weighted quintile frequencies.

    Returns a DataFrame with columns: unique_id, p1-p5
    If an asset has no history, the pooled frequencies are used.
    """
    hist = df[df[time_col] <= cutoff].copy()
    if hist.empty:
        return pd.DataFrame()

    hist = hist.dropna(subset=[target_col])
    if hist.empty:
        return pd.DataFrame()

    hl = halflife if halflife is not None else SETTINGS.ewma_halflife
    days_ago = (cutoff - hist[time_col]).dt.days
    hist["_w"] = np.exp(-days_ago / (hl * 5))
    hist["_w"] /= hist["_w"].sum()

    quintile_vals = hist[target_col].round().astype(int).clip(1, 5)

    temp = hist[[id_col]].assign(_q=quintile_vals, _w=hist["_w"])
    per_asset_rows = []
    for _, grp in temp.groupby(id_col, sort=False):
        w = grp["_w"].to_numpy()
        q = grp["_q"].to_numpy()
        counts = np.zeros(5, dtype=np.float64)
        for qi in range(1, 6):
            mask = q == qi
            counts[qi - 1] = w[mask].sum()
        prob = counts / counts.sum() if counts.sum() > 0 else np.full(5, 0.2)
        per_asset_rows.append(prob)
    per_asset = pd.DataFrame(
        per_asset_rows,
        index=temp[id_col].unique(),
        columns=[1, 2, 3, 4, 5],
    ).fillna(0.0)

    if use_pooled:
        all_w = hist["_w"].to_numpy()
        pooled = np.zeros(5, dtype=np.float64)
        for qi in range(1, 6):
            mask = quintile_vals == qi
            pooled[qi - 1] = all_w[mask].sum()
        pooled = pooled / pooled.sum() if pooled.sum() > 0 else np.full(5, 0.2)

        rows = []
        for uid in hist[id_col].unique():
            if uid in per_asset.index:
                probs = per_asset.loc[uid].to_numpy()
                min_obs = 10
                n_obs = len(hist[hist[id_col] == uid])
                if n_obs >= min_obs:
                    rows.append({"unique_id": uid, **{f"p{i + 1}": probs[i] for i in range(5)}})
                else:
                    alpha = n_obs / min_obs
                    blended = alpha * probs + (1 - alpha) * pooled
                    rows.append({"unique_id": uid, **{f"p{i + 1}": blended[i] for i in range(5)}})
            else:
                rows.append({"unique_id": uid, **{f"p{i + 1}": pooled[i] for i in range(5)}})
        return pd.DataFrame(rows)
    else:
        per_asset.columns = [f"p{i}" for i in range(1, 6)]
        per_asset = per_asset.reset_index()
        return per_asset.rename(columns={per_asset.index.name or id_col: id_col})


def predict_historical(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    h: int = 20,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "quintile",
) -> pd.DataFrame:
    """Predict quintile probabilities from EWMA-weighted historical frequencies.

    Args:
        df: Full long frame with price history and quintile assignments.
        cutoff: Forecast origin (trading date).
        h: Forecast horizon in trading days (unused by this model).
        id_col: Column identifying each asset.
        time_col: Column with trading dates.
        target_col: Column with quintile values.

    Returns:
        DataFrame with columns unique_id, ds, p1-p5
    """
    freqs = _compute_quintile_frequencies(df, cutoff, id_col=id_col, time_col=time_col, target_col=target_col)
    if freqs.empty:
        return pd.DataFrame()

    result = freqs.copy()
    result[time_col] = cutoff
    return result
