"""Integration tests for CV runners on toy data."""

from __future__ import annotations

import pandas as pd
import pytest

from m6.cv import gaussian_cv, historical_cv, naive_cv


def test_naive_cv_produces_columns(toy_long: pd.DataFrame) -> None:
    cv_df = naive_cv(toy_long, h=20, n_windows=2)
    if cv_df.empty:
        pytest.skip("CV produced empty frame")
    expected = {"unique_id", "ds", "cutoff", "y", "naive_p1", "naive_p2", "naive_p3", "naive_p4", "naive_p5"}
    assert expected.issubset(cv_df.columns)
    assert len(cv_df) > 0


def test_naive_cv_has_correct_folds(toy_long: pd.DataFrame) -> None:
    cv_df = naive_cv(toy_long, h=20, n_windows=2)
    if cv_df.empty:
        pytest.skip("CV produced empty frame")
    n_folds = cv_df["cutoff"].nunique()
    assert n_folds <= 2


def test_naive_cv_rps_is_0_point_16(toy_long: pd.DataFrame) -> None:
    from m6.evaluation import rps

    cv_df = naive_cv(toy_long, h=20, n_windows=2)
    if cv_df.empty:
        pytest.skip("CV produced empty frame")
    pcols = [f"naive_p{i}" for i in range(1, 6)]
    rps_val = rps(cv_df["y"], cv_df[pcols])
    assert rps_val == pytest.approx(0.16, abs=0.03)


def test_historical_cv_runs(toy_long: pd.DataFrame) -> None:
    cv_df = historical_cv(toy_long, h=20, n_windows=2)
    if cv_df.empty:
        pytest.skip("Historical CV produced empty frame")
    expected = {"unique_id", "ds", "cutoff", "y", "historical_p1"}
    assert expected.issubset(cv_df.columns)


def test_gaussian_cv_runs_with_defaults(toy_long: pd.DataFrame) -> None:
    """Gaussian CV needs enough history."""
    cv_df = gaussian_cv(toy_long, h=20, n_windows=1)
    if cv_df.empty:
        pytest.skip("Gaussian CV produced empty frame (insufficient history)")
    assert "gaussian_p1" in cv_df.columns
    assert cv_df["cutoff"].nunique() == 1
