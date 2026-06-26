"""Rolling-origin cross-validation for M6 probability models.

Each CV fold:
  - Uses data up to a cutoff date as training
  - Predicts quintile probabilities for the forward h-trading-day window
  - Records the actual quintiles when they materialise

The result is a wide frame with columns:
    unique_id, ds (cutoff date), y (actual quintile, 1-5),
    y_return (actual forward return), <model>_p1 .. <model>_p5
"""

from __future__ import annotations

import pandas as pd

from m6.config import SETTINGS, set_global_seed
from m6.data import make_cv_cutoffs
from m6.evaluation import N_QUINTILES
from m6.logging import logger
from m6.models.adaptive import predict_adaptive
from m6.models.ensemble import predict_ensemble
from m6.models.gaussian import predict_gaussian
from m6.models.historical import predict_historical
from m6.models.naive import predict_naive


def _align_to_realized(
    df: pd.DataFrame,
    preds: pd.DataFrame,
    cutoff: pd.Timestamp,
    h: int,
    *,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pd.DataFrame:
    forward_window = cutoff + pd.Timedelta(days=h * 3)
    forward = df[(df[time_col] > cutoff) & (df[time_col] <= forward_window)].copy()

    if forward.empty:
        return pd.DataFrame()

    if "return_fwd" not in forward.columns or "quintile" not in forward.columns:
        from m6.data import assign_quintiles, compute_forward_returns

        forward = compute_forward_returns(forward, horizon=h)
        forward = assign_quintiles(forward)

    first_fwd = forward.groupby(id_col, observed=True)[time_col].transform("min")
    is_first = forward[time_col] == first_fwd

    realized = forward[is_first][[id_col, "return_fwd", "quintile"]].dropna(subset=["quintile"])
    realized = realized.rename(columns={"quintile": target_col, "return_fwd": "y_return"})

    merged = preds.merge(realized, on=id_col, how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["cutoff"] = cutoff
    return merged[
        [id_col, time_col, "cutoff", target_col, "y_return"] + [f"p{i}" for i in range(1, N_QUINTILES + 1)]
    ]


def _run_cv_for_model(
    df: pd.DataFrame,
    model_name: str,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
    model_fn=None,
) -> pd.DataFrame:
    """Run rolling-origin CV for a single model.

    Args:
        df: Long frame with daily returns and quintile assignments.
        model_name: Label for the forecast columns.
        h: Forecast horizon in trading days.
        n_windows: Number of CV windows.
        model_fn: Callable(df, cutoff, h) -> DataFrame with p1-p5 columns.

    Returns:
        Wide CV frame with columns unique_id, ds, cutoff, y, y_return, <model>_p1..p5
    """
    set_global_seed()
    cutoffs = make_cv_cutoffs(df, n_windows=n_windows, horizon=h)
    logger.info(f"cv[{model_name}]: h={h} n_windows={len(cutoffs)}")

    folds = []
    for cutoff in cutoffs:
        preds = model_fn(df, cutoff, h)
        if preds.empty:
            logger.warning(f"cv[{model_name}]: no predictions for cutoff {cutoff.date()}")
            continue
        aligned = _align_to_realized(df, preds, cutoff, h)
        if aligned.empty:
            logger.warning(f"cv[{model_name}]: no realized data for cutoff {cutoff.date()}")
            continue
        rename_map = {f"p{i}": f"{model_name}_p{i}" for i in range(1, N_QUINTILES + 1)}
        aligned = aligned.rename(columns=rename_map)
        folds.append(aligned)
        logger.info(f"cv[{model_name}]: cutoff {cutoff.date()} → {len(aligned)} assets")

    if not folds:
        raise ValueError(f"cv[{model_name}]: no folds produced any data.")

    result = pd.concat(folds, ignore_index=True)
    logger.info(f"cv[{model_name}]: total {len(result):,d} rows across {len(cutoffs)} folds")
    return result


def naive_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
) -> pd.DataFrame:
    """CV for the naive equal-probability benchmark."""
    return _run_cv_for_model(df, "naive", h=h, n_windows=n_windows, model_fn=predict_naive)


def historical_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
) -> pd.DataFrame:
    """CV for the historical-frequency model."""
    return _run_cv_for_model(df, "historical", h=h, n_windows=n_windows, model_fn=predict_historical)


def gaussian_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
) -> pd.DataFrame:
    """CV for the multivariate Gaussian model."""
    return _run_cv_for_model(df, "gaussian", h=h, n_windows=n_windows, model_fn=predict_gaussian)


def ensemble_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
) -> pd.DataFrame:
    """CV for the ensemble (gaussian + historical) model."""
    return _run_cv_for_model(df, "ensemble", h=h, n_windows=n_windows, model_fn=predict_ensemble)


def adaptive_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = SETTINGS.n_windows,
) -> pd.DataFrame:
    """CV for the adaptive gradient boosting model."""
    return _run_cv_for_model(df, "adaptive", h=h, n_windows=n_windows, model_fn=predict_adaptive)
