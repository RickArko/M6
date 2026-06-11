"""Historical-frequency model — predicts quintile probabilities from past frequencies.

For each asset, the probability of landing in each quintile is estimated as
the historical frequency observed prior to the cutoff date. This captures
asset-specific biases (e.g., high-volatility assets appear in extreme
quintiles more often).

A pooled variant uses all-asset frequencies when per-asset data is sparse.
"""

from __future__ import annotations

import pandas as pd


def _compute_quintile_frequencies(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "quintile",
    use_pooled: bool = True,
) -> pd.DataFrame:
    """Compute per-asset (and optionally pooled) quintile frequencies.

    Returns a DataFrame with columns: unique_id, p1-p5
    If an asset has no history, the pooled frequencies are used.
    """
    hist = df[df[time_col] <= cutoff].copy()
    if hist.empty:
        return pd.DataFrame()

    hist = hist.dropna(subset=[target_col])
    if hist.empty:
        return pd.DataFrame()

    quintiles = hist[target_col].round().astype(int).clip(1, 5)

    freq = (
        hist[[id_col]]
        .assign(_q=quintiles)
        .groupby(id_col)["_q"]
        .value_counts(normalize=True)
        .reset_index(name="prob")
    )
    per_asset = freq.pivot_table(index=id_col, columns="_q", values="prob", fill_value=0)

    for q in range(1, 6):
        if q not in per_asset.columns:
            per_asset[q] = 0.0
    per_asset = per_asset[sorted(per_asset.columns)]

    if use_pooled:
        pooled = quintiles.value_counts(normalize=True).reindex(range(1, 6), fill_value=0.2)
        pooled_probs = pooled.to_numpy()

        rows = []
        for uid in hist[id_col].unique():
            if uid in per_asset.index:
                probs = per_asset.loc[uid].to_numpy()
                min_obs = 10
                n_obs = (hist[hist[id_col] == uid][time_col] <= cutoff).sum()
                if n_obs >= min_obs:
                    rows.append({"unique_id": uid, **{f"p{i + 1}": probs[i] for i in range(5)}})
                else:
                    alpha = n_obs / min_obs
                    blended = alpha * probs + (1 - alpha) * pooled_probs
                    rows.append({"unique_id": uid, **{f"p{i + 1}": blended[i] for i in range(5)}})
            else:
                rows.append({"unique_id": uid, **{f"p{i + 1}": pooled_probs[i] for i in range(5)}})
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
    """Predict quintile probabilities from historical frequencies.

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
