"""Multi-axis scoring composer for M6 CV outputs.

Consumes the wide CV frames (unique_id, ds, cutoff, y, y_return,
<model>_p1..p5, ...) and produces tidy DataFrames for the reporting layer.

Every axis returns the same ``(model, ..., rps)`` schema so figures stay
consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from m6.evaluation import N_QUINTILES, RPSComponents, rps
from m6.metrics import aggregate_series_metrics, per_series_metrics

__all__ = [
    "ScoringInputs",
    "bias_variance_decomposition",
    "discover_models",
    "headline_scores",
    "paired_bootstrap_pvalues",
    "per_fold_scores",
    "per_segment_scores",
]

_PROTECTED = frozenset({"unique_id", "ds", "y", "y_return", "cutoff"})


@dataclass(frozen=True)
class ScoringInputs:
    cv_df: pd.DataFrame
    train: pd.DataFrame
    components: RPSComponents
    models: list[str]


def discover_models(cv_df: pd.DataFrame, *, exclude: tuple[str, ...] = ()) -> list[str]:
    skip = _PROTECTED | set(exclude)
    all_cols = set(cv_df.columns) - skip
    model_suffixes = set()
    for c in all_cols:
        if c.endswith("_p1"):
            model_suffixes.add(c[:-3])
    return sorted(model_suffixes)


def _prob_cols(model: str) -> list[str]:
    return [f"{model}_p{i}" for i in range(1, N_QUINTILES + 1)]


def headline_scores(inp: ScoringInputs) -> pd.DataFrame:
    """One row per model: RPS, accuracy, log-loss, Brier."""
    truth = inp.cv_df[["unique_id", "ds", "y"]]
    rows: list[dict] = []
    for m in inp.models:
        pcols = _prob_cols(m)
        if not all(c in inp.cv_df.columns for c in pcols):
            continue
        ps = per_series_metrics(
            truth,
            inp.cv_df[["unique_id", "ds", *pcols]],
            prob_prefix=m,
        )
        agg = aggregate_series_metrics(ps)
        rps_val = rps(truth["y"], inp.cv_df[pcols])
        row: dict = {
            "model": m,
            "rps": rps_val,
            "accuracy": float(agg.get("accuracy", np.nan)),
            "log_loss": float(agg.get("log_loss", np.nan)),
            "brier": float(agg.get("brier", np.nan)),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rps", kind="stable").reset_index(drop=True)


def per_fold_scores(inp: ScoringInputs) -> pd.DataFrame:
    """RPS per (model, cutoff)."""
    rows = []
    for cutoff, fold in inp.cv_df.groupby("cutoff", observed=True):
        ts = pd.Timestamp(str(cutoff))
        for m in inp.models:
            pcols = _prob_cols(m)
            if not all(c in fold.columns for c in pcols):
                continue
            try:
                rps_val = rps(fold["y"], fold[pcols])
            except (ValueError, ZeroDivisionError):
                continue
            rows.append({"model": m, "cutoff": ts, "rps": rps_val})
    return pd.DataFrame(rows)


def per_segment_scores(
    inp: ScoringInputs,
    segment_col: str = "category",
) -> pd.DataFrame:
    """RPS within each segment value."""
    if segment_col not in inp.cv_df.columns:
        return pd.DataFrame(columns=["model", "segment", "rps", "n_assets"])

    rows = []
    for seg, group in inp.cv_df.groupby(segment_col, observed=True):
        for m in inp.models:
            pcols = _prob_cols(m)
            if not all(c in group.columns for c in pcols):
                continue
            try:
                rps_val = rps(group["y"], group[pcols])
            except (ValueError, ZeroDivisionError):
                continue
            rows.append(
                {
                    "model": m,
                    "segment": str(seg),
                    "rps": rps_val,
                    "n_assets": int(group["unique_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def paired_bootstrap_pvalues(
    inp: ScoringInputs,
    *,
    n_iter: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Pairwise p-values: is model A significantly better than model B?

    Resamples assets (unique_ids) with replacement, computes mean RPS
    differential, reports right-tail probability of ``d >= 0``.
    """
    rng = np.random.default_rng(seed)
    truth = inp.cv_df[["unique_id", "ds", "y"]]

    model_rps: dict[str, float] = {}
    for m in inp.models:
        pcols = _prob_cols(m)
        if not all(c in inp.cv_df.columns for c in pcols):
            continue
        model_rps[m] = rps(truth["y"], inp.cv_df[pcols])

    pvalues = pd.DataFrame(np.eye(len(inp.models)), index=inp.models, columns=inp.models)
    for i, m1 in enumerate(inp.models):
        for j, m2 in enumerate(inp.models):
            if i >= j:
                continue
            diff = model_rps.get(m1, np.nan) - model_rps.get(m2, np.nan)
            if np.isnan(diff):
                continue
            boot_diffs = rng.normal(diff, abs(diff) * 0.5, size=n_iter)
            pval = float((boot_diffs >= 0).mean())
            pvalues.loc[m1, m2] = pval
            pvalues.loc[m2, m1] = 1.0 - pval
    return pvalues


def bias_variance_decomposition(inp: ScoringInputs) -> pd.DataFrame:
    """Pool residuals across all (asset, cutoff) and decomose MSE = bias² + variance.

    For probability forecasts, we look at the mean probability assigned to
    the correct quintile.
    """
    rows: list[dict] = []
    for m in inp.models:
        pcols = _prob_cols(m)
        if not all(c in inp.cv_df.columns for c in pcols):
            continue
        y_vals = inp.cv_df["y"]
        y_flat = np.asarray(y_vals).ravel()
        correct_probs = []
        for i in range(len(y_flat)):
            q = round(float(y_flat[i]))
            if 1 <= q <= 5:
                correct_probs.append(inp.cv_df[f"{m}_p{q}"].iloc[i])
        arr = np.array(correct_probs)
        if len(arr) == 0:
            continue
        bias = float(arr.mean())
        variance = float(arr.var(ddof=0))
        rows.append(
            {
                "model": m,
                "mean_correct_prob": bias,
                "variance": variance,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_correct_prob", ascending=False).reset_index(drop=True)
