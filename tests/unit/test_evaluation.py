"""Unit tests for RPS computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from m6.evaluation import (
    N_QUINTILES,
    compute_components,
    make_submission,
    rps,
    rps_for_models,
)


def _perfect_forecast(quintiles: pd.Series) -> pd.DataFrame:
    probs = np.zeros((len(quintiles), N_QUINTILES))
    for i, q in enumerate(quintiles):
        idx = round(q) - 1
        if 0 <= idx < N_QUINTILES:
            probs[i, idx] = 1.0
        else:
            probs[i, :] = 0.2
    return pd.DataFrame(probs, columns=[f"p{i}" for i in range(1, N_QUINTILES + 1)])


def _naive_forecast(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        np.full((n, N_QUINTILES), 0.2),
        columns=[f"p{i}" for i in range(1, N_QUINTILES + 1)],
    )


def test_naive_rps_is_0_point_16(toy_long: pd.DataFrame) -> None:
    """Naive equal-probability has RPS ≈ 0.16 on large samples."""
    actual = toy_long["quintile"].dropna()
    if len(actual) == 0:
        pytest.skip("No quintile data in toy_long")
    fc = _naive_forecast(len(actual))
    rps_val = rps(actual, fc)
    assert rps_val == pytest.approx(0.16, abs=0.03)


def test_perfect_rps_is_zero(toy_long: pd.DataFrame) -> None:
    """Perfect forecast should have RPS ≈ 0 (exact for integer quintiles)."""
    actual = toy_long["quintile"].dropna()
    if len(actual) == 0:
        pytest.skip("No quintile data in toy_long")
    fc = _perfect_forecast(actual)
    rps_val = rps(actual, fc)
    exact_quintiles = actual.round() == actual
    if exact_quintiles.all():
        assert rps_val == pytest.approx(0.0, abs=1e-10)
    else:
        assert rps_val < 0.02


def test_perfect_beats_naive(toy_long: pd.DataFrame) -> None:
    """Perfect forecast must score better (lower) than naive."""
    actual = toy_long["quintile"].dropna()
    if len(actual) == 0:
        pytest.skip("No quintile data in toy_long")
    naive_rps = rps(actual, _naive_forecast(len(actual)))
    perfect_rps = rps(actual, _perfect_forecast(actual))
    assert perfect_rps < naive_rps


def test_rps_for_models_ranks_models(toy_cv: pd.DataFrame) -> None:
    """rps_for_models should find models from _{p1..p5} columns and score them."""
    truth = toy_cv[["unique_id", "ds", "y"]].copy()
    if toy_cv.empty:
        pytest.skip("Empty CV frame")
    scores = rps_for_models(truth, toy_cv)
    assert "naive" in scores.index
    assert "perfect" in scores.index
    assert scores["perfect"] < scores["naive"]


def test_compute_components_returns_tickers(toy_long: pd.DataFrame) -> None:
    comp = compute_components(toy_long)
    assert comp.n_assets > 0
    assert len(comp.tickers) == comp.n_assets


def test_make_submission_keeps_prob_cols(toy_long: pd.DataFrame) -> None:
    df = toy_long.head(10).copy()
    prob_cols = ["p1", "p2", "p3", "p4", "p5"]
    for i, col in enumerate(prob_cols, 1):
        df[col] = 0.2
    sub = make_submission(df, prob_cols=prob_cols)
    assert sub.columns.tolist() == ["unique_id", "ds", *prob_cols]
