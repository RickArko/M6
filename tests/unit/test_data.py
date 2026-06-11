"""Unit tests for the data module."""

from __future__ import annotations

import pandas as pd

from m6.data import (
    M6_TICKERS,
    assign_quintiles,
    compute_forward_returns,
    make_cv_cutoffs,
)


def test_compute_forward_returns_adds_column(toy_returns: pd.DataFrame) -> None:
    df = compute_forward_returns(toy_returns, horizon=20)
    assert "return_fwd" in df.columns
    assert df["return_fwd"].notna().sum() > 0


def test_assign_quintiles_range(toy_returns: pd.DataFrame) -> None:
    df = compute_forward_returns(toy_returns, horizon=20)
    df = assign_quintiles(df)
    quintiles = df["quintile"].dropna()
    assert (quintiles >= 1).all()
    assert (quintiles <= 5).all()


def test_assign_quintiles_balanced(toy_returns: pd.DataFrame) -> None:
    df = compute_forward_returns(toy_returns, horizon=20)
    df = assign_quintiles(df)
    q = df["quintile"].dropna()
    if len(q) > 0:
        mean_q = q.mean()
        assert 2.5 <= mean_q <= 3.5  # should be roughly uniform


def test_make_cv_cutoffs_length(toy_returns: pd.DataFrame) -> None:
    cutoffs = make_cv_cutoffs(toy_returns, n_windows=3, horizon=20)
    assert len(cutoffs) == 3
    assert all(isinstance(c, pd.Timestamp) for c in cutoffs)


def test_make_cv_cutoffs_ordered(toy_returns: pd.DataFrame) -> None:
    cutoffs = make_cv_cutoffs(toy_returns, n_windows=3, horizon=20)
    assert cutoffs == sorted(cutoffs)


def test_tickers_list_length() -> None:
    assert len(M6_TICKERS) == 100


def test_tickers_list_has_stocks_and_etfs() -> None:
    uk_etfs = [t for t in M6_TICKERS if t.endswith(".L")]
    us_etfs = [
        t
        for t in M6_TICKERS
        if t
        in {
            "EWA",
            "EWC",
            "EWG",
            "EWH",
            "EWJ",
            "EWL",
            "EWQ",
            "EWT",
            "EWU",
            "EWY",
            "EWZ",
            "GSG",
            "HYG",
            "IAU",
            "ICLN",
            "IEF",
            "IEMG",
            "IGF",
            "INDA",
            "IVV",
            "IWM",
            "IXN",
            "LQD",
            "MCHI",
            "RE",
            "REET",
            "SHY",
            "SLV",
            "TLT",
            "VXX",
            "XLB",
            "XLC",
            "XLE",
            "XLF",
            "XLI",
            "XLK",
            "XLP",
            "XLU",
            "XLV",
            "XLY",
        }
    ]
    stocks = [t for t in M6_TICKERS if t not in uk_etfs and t not in us_etfs]
    assert len(stocks) == 50
    assert len(us_etfs) == 40
    assert len(uk_etfs) == 10
