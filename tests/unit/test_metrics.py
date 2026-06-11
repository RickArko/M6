"""Unit tests for metrics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from m6.metrics import (
    aggregate_series_metrics,
    brier_score,
    log_loss,
    per_series_metrics,
    quintile_accuracy,
)


def test_perfect_quintile_accuracy() -> None:
    actual = pd.Series([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    predicted = pd.Series([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    acc = quintile_accuracy(actual, predicted)
    assert acc == 1.0


def test_naive_quintile_accuracy() -> None:
    rng = np.random.default_rng(42)
    actual = pd.Series(rng.integers(1, 6, size=1000))
    predicted = pd.Series(rng.integers(1, 6, size=1000))
    acc = quintile_accuracy(actual, predicted)
    assert 0.15 <= acc <= 0.25  # random guessing ~20%


def test_log_loss_perfect() -> None:
    actual = pd.Series([1, 2, 3, 4, 5])
    probs = pd.DataFrame(
        {
            "p1": [1.0, 0.0, 0.0, 0.0, 0.0],
            "p2": [0.0, 1.0, 0.0, 0.0, 0.0],
            "p3": [0.0, 0.0, 1.0, 0.0, 0.0],
            "p4": [0.0, 0.0, 0.0, 1.0, 0.0],
            "p5": [0.0, 0.0, 0.0, 0.0, 1.0],
        }
    )
    ll = log_loss(actual, probs)
    assert ll == pytest.approx(0.0, abs=1e-9)


def test_log_loss_naive() -> None:
    actual = pd.Series([1, 2, 3, 4, 5])
    probs = pd.DataFrame(
        np.full((5, 5), 0.2),
        columns=[f"p{i}" for i in range(1, 6)],
    )
    ll = log_loss(actual, probs)
    expected = -np.log(0.2)
    assert ll == pytest.approx(expected, abs=1e-9)


def test_brier_score_perfect() -> None:
    actual = pd.Series([1, 2, 3, 4, 5])
    probs = pd.DataFrame(
        {
            "p1": [1.0, 0.0, 0.0, 0.0, 0.0],
            "p2": [0.0, 1.0, 0.0, 0.0, 0.0],
            "p3": [0.0, 0.0, 1.0, 0.0, 0.0],
            "p4": [0.0, 0.0, 0.0, 1.0, 0.0],
            "p5": [0.0, 0.0, 0.0, 0.0, 1.0],
        }
    )
    bs = brier_score(actual, probs)
    assert bs == pytest.approx(0.0, abs=1e-9)


def test_brier_score_naive() -> None:
    actual = pd.Series([1, 2, 3, 4, 5])
    probs = pd.DataFrame(
        np.full((5, 5), 0.2),
        columns=[f"p{i}" for i in range(1, 6)],
    )
    bs = brier_score(actual, probs)
    expected = 0.16  # (0.8^2 + 4*0.2^2) / 5 = 0.16
    assert bs == pytest.approx(expected, abs=1e-9)


def test_aggregate_series_metrics(toy_cv: pd.DataFrame) -> None:
    if toy_cv.empty:
        pytest.skip("Empty CV frame")
    ps = per_series_metrics(
        toy_cv[["unique_id", "ds", "y"]],
        toy_cv[["unique_id", "ds"] + [c for c in toy_cv.columns if "naive_p" in c]],
        prob_prefix="naive",
    )
    agg = aggregate_series_metrics(ps)
    assert "rps" in agg.index
    assert "accuracy" in agg.index
