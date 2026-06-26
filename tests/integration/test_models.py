"""Integration tests for M6 models — fit + predict on toy data."""

from __future__ import annotations

import pandas as pd
import pytest

from m6.models.adaptive import predict_adaptive
from m6.models.gaussian import predict_gaussian
from m6.models.historical import predict_historical
from m6.models.naive import predict_naive


def test_naive_always_0_point_2(toy_returns: pd.DataFrame) -> None:
    cutoff = toy_returns["ds"].max()
    preds = predict_naive(toy_returns, cutoff)
    assert not preds.empty
    for _, row in preds.iterrows():
        probs = [row[f"p{i}"] for i in range(1, 6)]
        assert all(p == 0.2 for p in probs)
        assert sum(probs) == pytest.approx(1.0)


def test_historical_probs_sum_to_one(toy_long: pd.DataFrame) -> None:
    cutoff = toy_long["ds"].max()
    preds = predict_historical(toy_long, cutoff, target_col="quintile")
    if preds.empty:
        pytest.skip("Historical model returned empty predictions")
    for _, row in preds.iterrows():
        probs = [row[f"p{i}"] for i in range(1, 6)]
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)
        assert all(p >= 0 for p in probs)


def test_historical_has_all_assets(toy_long: pd.DataFrame) -> None:
    cutoff = toy_long["ds"].max()
    preds = predict_historical(toy_long, cutoff, target_col="quintile")
    if preds.empty:
        pytest.skip("Historical model returned empty predictions")
    expected_assets = set(toy_long["unique_id"].unique())
    pred_assets = set(preds["unique_id"])
    assert pred_assets == expected_assets


def test_gaussian_probs_sum_to_one(toy_returns: pd.DataFrame) -> None:
    cutoff = toy_returns["ds"].max()
    preds = predict_gaussian(toy_returns, cutoff, n_simulations=1000)
    if preds.empty:
        pytest.skip("Gaussian model returned empty predictions (insufficient data)")
    for _, row in preds.iterrows():
        probs = [row[f"p{i}"] for i in range(1, 6)]
        assert sum(probs) == pytest.approx(1.0, abs=0.02)
        assert all(p >= 0 for p in probs)


def test_gaussian_different_from_naive(toy_returns: pd.DataFrame) -> None:
    cutoff = toy_returns["ds"].max()
    gauss_preds = predict_gaussian(toy_returns, cutoff, n_simulations=5000)
    if gauss_preds.empty:
        pytest.skip("Gaussian model returned empty predictions")
    naive_preds = predict_naive(toy_returns, cutoff)
    merged = gauss_preds.merge(naive_preds, on="unique_id", suffixes=("_g", "_n"))
    if not merged.empty:
        first_row = merged.iloc[0]
        gauss_probs = [first_row[f"p{i}_g"] for i in range(1, 6)]
        assert not all(p == pytest.approx(0.2, abs=1e-6) for p in gauss_probs)


def test_adaptive_probs_sum_to_one(toy_long: pd.DataFrame) -> None:
    cutoff = toy_long["ds"].max()
    preds = predict_adaptive(toy_long, cutoff, target_col="y")
    if preds.empty:
        pytest.skip("Adaptive model returned empty predictions (insufficient data)")
    for _, row in preds.iterrows():
        probs = [row[f"p{i}"] for i in range(1, 6)]
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)
        assert all(p >= 0 for p in probs)


def test_adaptive_has_all_assets(toy_long: pd.DataFrame) -> None:
    cutoff = toy_long["ds"].max()
    preds = predict_adaptive(toy_long, cutoff, target_col="y")
    if preds.empty:
        pytest.skip("Adaptive model returned empty predictions")
    expected_assets = set(toy_long["unique_id"].unique())
    pred_assets = set(preds["unique_id"])
    assert pred_assets == expected_assets
