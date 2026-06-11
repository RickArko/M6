"""Point metrics for probability forecasts on quintile outcomes.

These supplement the primary RPS metric with interpretable diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from m6.evaluation import N_QUINTILES, _quintile_to_onehot, rps

EPS = 1e-9

__all__ = [
    "aggregate_series_metrics",
    "brier_score",
    "log_loss",
    "per_series_metrics",
    "quintile_accuracy",
]


def quintile_accuracy(
    actual_q: pd.Series,
    predicted_q: pd.Series,
) -> float:
    """Fraction of times the predicted quintile (mode) matches the actual quintile."""
    correct = (actual_q.round().astype(int).clip(1, 5) == predicted_q.round().astype(int).clip(1, 5)).sum()
    return correct / max(len(actual_q), 1)


def log_loss(
    actual_q: pd.Series,
    forecast_probs: pd.DataFrame,
) -> float:
    """Multi-class log loss (cross-entropy).

    Lower is better. Infinite if any probability is exactly 0 for the
    correct class.
    """
    total = 0.0
    n = len(actual_q)
    for i in range(n):
        oh = _quintile_to_onehot(actual_q.iloc[i])
        f = forecast_probs.iloc[i].to_numpy().clip(EPS, 1 - EPS)
        total -= float(np.sum(oh * np.log(f)))
    return total / n if n > 0 else float("nan")


def brier_score(
    actual_q: pd.Series,
    forecast_probs: pd.DataFrame,
) -> float:
    """Multi-class Brier score (mean squared error between forecast probs and one-hot).

    Lower is better. This is the same as RPS for a single class, but
    without the CDF transform that RPS uses. RPS is more appropriate for
    ordered categories like quintiles.
    """
    total = 0.0
    n = len(actual_q)
    for i in range(n):
        oh = _quintile_to_onehot(actual_q.iloc[i])
        f = forecast_probs.iloc[i].to_numpy()
        total += float(np.sum((oh - f) ** 2))
    return total / (n * N_QUINTILES) if n > 0 else float("nan")


def per_series_metrics(
    actual: pd.DataFrame,
    forecast_probs: pd.DataFrame,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    prob_prefix: str = "",
) -> pd.DataFrame:
    """Per-asset metrics: RPS, accuracy, log-loss, Brier.

    Returns a DataFrame indexed by unique_id with columns for each metric.
    """
    merged = actual[[id_col, time_col, target_col]].merge(forecast_probs, on=[id_col, time_col], how="inner")
    if merged.empty:
        raise ValueError("No overlapping rows between actual and forecast.")

    pcols = [f"{prob_prefix}_p{i}" if prob_prefix else f"p{i}" for i in range(1, N_QUINTILES + 1)]
    pcols = [c for c in pcols if c in merged.columns]

    rows: list[dict] = []
    for uid, group in merged.groupby(id_col, observed=True):
        if len(pcols) == 0:
            continue
        g_actual = group[target_col]
        g_probs = group[pcols]
        rps_val = rps(g_actual, g_probs)
        if np.isnan(rps_val):
            continue
        acc = quintile_accuracy(
            g_actual,
            pd.Series(np.argmax(g_probs.to_numpy(), axis=1) + 1, index=group.index),
        )
        ll = log_loss(g_actual, g_probs)
        bs = brier_score(g_actual, g_probs)
        rows.append(
            {
                "unique_id": uid,
                "rps": rps_val,
                "accuracy": acc,
                "log_loss": ll,
                "brier": bs,
                "n_obs": len(group),
            }
        )
    return pd.DataFrame(rows).set_index("unique_id")


def aggregate_series_metrics(
    per_series: pd.DataFrame,
    *,
    weights: pd.Series | None = None,
) -> pd.Series:
    """Reduce per-series metrics to scalars; unweighted mean for RPS."""
    cols = [c for c in ("rps", "accuracy", "log_loss", "brier") if c in per_series.columns]
    if weights is None:
        return per_series[cols].mean()
    common = per_series.index.intersection(weights.index)
    w = weights.loc[common]
    w = w / w.sum() if w.sum() > 0 else w
    return pd.Series(
        {c: float((per_series.loc[common, c] * w).sum()) for c in cols},
        name="aggregate",
    )
