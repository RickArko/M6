"""Unit tests for the scoring module."""

from __future__ import annotations

import pandas as pd
import pytest

from m6.evaluation import RPSComponents
from m6.scoring import (
    ScoringInputs,
    bias_variance_decomposition,
    discover_models,
    headline_scores,
    paired_bootstrap_pvalues,
    per_fold_scores,
    per_segment_scores,
)


def _make_inputs(toy_cv: pd.DataFrame) -> ScoringInputs | None:
    if toy_cv.empty:
        return None
    models = discover_models(toy_cv)
    if not models:
        return None
    comp = RPSComponents(
        n_assets=int(toy_cv["unique_id"].nunique()),
        tickers=toy_cv["unique_id"].unique(),
    )
    return ScoringInputs(
        cv_df=toy_cv,
        train=toy_cv,
        components=comp,
        models=models,
    )


def test_discover_models_finds_both(toy_cv: pd.DataFrame) -> None:
    if toy_cv.empty:
        pytest.skip("Empty CV frame")
    models = discover_models(toy_cv)
    assert "naive" in models
    assert "perfect" in models


def test_headline_scores_ranks_perfect_first(toy_cv: pd.DataFrame) -> None:
    inp = _make_inputs(toy_cv)
    if inp is None:
        pytest.skip("No models found in CV frame")
    h = headline_scores(inp)
    assert not h.empty
    assert h.iloc[0]["model"] == "perfect"


def test_headline_scores_rps_range(toy_cv: pd.DataFrame) -> None:
    inp = _make_inputs(toy_cv)
    if inp is None:
        pytest.skip("No models found in CV frame")
    h = headline_scores(inp)
    perfect_rps = h.loc[h["model"] == "perfect", "rps"].iloc[0]
    naive_rps = h.loc[h["model"] == "naive", "rps"].iloc[0]
    assert perfect_rps < 0.02  # near-perfect on small sample
    assert naive_rps == pytest.approx(0.16, abs=0.03)


def test_per_fold_scores(toy_cv: pd.DataFrame) -> None:
    inp = _make_inputs(toy_cv)
    if inp is None:
        pytest.skip("No models found in CV frame")
    pf = per_fold_scores(inp)
    assert not pf.empty
    assert {"model", "cutoff", "rps"}.issubset(pf.columns)


def test_per_segment_scores(toy_cv: pd.DataFrame) -> None:
    inp = _make_inputs(toy_cv)
    if inp is None:
        pytest.skip("No models found in CV frame")
    seg = per_segment_scores(inp, "category")
    if not seg.empty:
        assert {"model", "segment", "rps", "n_assets"}.issubset(seg.columns)


def test_perfect_has_low_bias_variance(toy_cv: pd.DataFrame) -> None:
    inp = _make_inputs(toy_cv)
    if inp is None:
        pytest.skip("No models found in CV frame")
    bv = bias_variance_decomposition(inp)
    if not bv.empty:
        perfect = bv[bv["model"] == "perfect"]
        if not perfect.empty:
            assert perfect.iloc[0]["mean_correct_prob"] == pytest.approx(1.0, abs=1e-9)


def test_paired_bootstrap(toy_cv: pd.DataFrame) -> None:
    inp = _make_inputs(toy_cv)
    if inp is None:
        pytest.skip("No models found in CV frame")
    pv = paired_bootstrap_pvalues(inp, n_iter=100, seed=42)
    assert pv.shape == (2, 2)
    assert pv.loc["perfect", "naive"] < 0.05
    assert pv.loc["naive", "perfect"] > 0.95
