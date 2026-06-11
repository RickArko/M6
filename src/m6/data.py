"""Download and shape M6 financial data into an evaluation-ready long frame.

Builds a long DataFrame with assets as ``unique_id``, trading dates as ``ds``,
and daily returns as ``y``. Then for each rolling forecast window (cutoff),
computes forward ``horizon``-day returns and quintile assignments.

The M6 asset universe: 50 US stocks + 50 international ETFs.

Schema convention (Nixtla-style):
    unique_id : asset ticker (str)
    ds        : trading date (datetime64[ns])
    y         : daily log return or price used for feature engineering
    plus forward-return and quintile columns for evaluation
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from m6.config import SETTINGS
from m6.logging import logger

M6_TICKERS = [
    "ABBV",
    "ACN",
    "AEP",
    "AIZ",
    "ALLE",
    "AMAT",
    "AMP",
    "AMZN",
    "AVB",
    "AVY",
    "AXP",
    "BDX",
    "BF-B",
    "BMY",
    "BR",
    "CARR",
    "CDW",
    "CE",
    "CHTR",
    "CNC",
    "CNP",
    "COP",
    "CTAS",
    "CZR",
    "DG",
    "DPZ",
    "O",
    "DXC",
    "FTV",
    "GOOG",
    "GPC",
    "HIG",
    "HST",
    "JPM",
    "KR",
    "META",
    "OGN",
    "PG",
    "PPL",
    "PRU",
    "PYPL",
    "ROL",
    "ROST",
    "UNH",
    "URI",
    "V",
    "VRSK",
    "IP",
    "XOM",
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
    "HIGH.L",
    "HYG",
    "IAU",
    "ICLN",
    "IEAA.L",
    "IEF",
    "IEFM.L",
    "IEMG",
    "IEUS",
    "IEVL.L",
    "IGF",
    "INDA",
    "IUMO.L",
    "IUVL.L",
    "IVV",
    "IWM",
    "IXN",
    "JPEA.L",
    "LQD",
    "MCHI",
    "MVEU.L",
    "EG",
    "REET",
    "SEGA.L",
    "SHY",
    "SLV",
    "SPMV.L",
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
]

ASSET_CATEGORIES = {t: "stock" for t in M6_TICKERS[:50]}
ASSET_CATEGORIES.update({t: "etf" for t in M6_TICKERS[50:]})
for t in [
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
    "EG",
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
]:
    ASSET_CATEGORIES[t] = "us_etf"
for t in ["HIGH.L", "IEAA.L", "IEFM.L", "IEVL.L", "IUMO.L", "IUVL.L", "JPEA.L", "MVEU.L", "SEGA.L", "SPMV.L"]:
    ASSET_CATEGORIES[t] = "uk_etf"


# --- Published top-3 M6 solutions for benchmarks --------------------
# These RPS scores let users judge whether their model is competitive.
PUBLISHED_RPS = {
    "Dan (1st)": 0.15645,
    "FinQBoost (2nd)": 0.15648,
    "SebastianR (3rd)": 0.15649,
    "Naive benchmark": 0.16,
}


def load_prices(
    tickers: list[str] | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Download or cache daily adjusted close prices from Yahoo Finance.

    Returns a long DataFrame with columns: unique_id, ds, price, volume.
    Caches to parquet if ``cache_dir`` is provided.
    """
    import yfinance as yf

    tickers = tickers or M6_TICKERS
    start = start or SETTINGS.start_date
    end = end or SETTINGS.end_date
    cache_path = (cache_dir or SETTINGS.raw_dir) / "prices.parquet"

    if cache_path and cache_path.exists():
        logger.info(f"load_prices: loading cached {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"load_prices: downloading {len(tickers)} tickers from Yahoo Finance …")
    t0 = time.time()

    n_batches = (len(tickers) + 19) // 20
    frames = []
    for i in range(n_batches):
        batch = tickers[i * 20 : (i + 1) * 20]
        data = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            actions=False,
        )
        if data.empty:
            logger.warning(f"Batch {i}: no data returned for {batch}")
            continue
        prices = data["Close"]
        prices.columns = [str(c) for c in prices.columns]

        long_batch = prices.reset_index().melt(id_vars="Date", var_name="unique_id", value_name="price")
        long_batch = long_batch.rename(columns={"Date": "ds"})
        long_batch = long_batch.dropna(subset=["price"])
        frames.append(long_batch)
        logger.info(f"  Batch {i + 1}/{n_batches}: {len(batch)} tickers")

    if not frames:
        raise RuntimeError("No price data downloaded — check tickers and date range.")

    df = pd.concat(frames, ignore_index=True)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    df["y"] = df.groupby("unique_id", observed=True)["price"].pct_change().astype(np.float32)
    df["y"] = df["y"].fillna(0.0)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info(f"load_prices: cached {len(df):,d} rows → {cache_path}")

    logger.info(f"load_prices: done in {time.time() - t0:.1f}s ({len(df):,d} rows)")
    return df


def compute_forward_returns(
    prices: pd.DataFrame,
    *,
    horizon: int | None = None,
) -> pd.DataFrame:
    """For each row, compute the forward ``horizon``-day total return.

    Adds columns:
        return_fwd : forward horizon-day log return
        quintile   : quintile rank (1 = worst, 5 = best) among assets at the same cutoff
    """
    h = horizon or SETTINGS.horizon
    df = prices.sort_values(["unique_id", "ds"]).copy()
    df["return_fwd"] = (
        df.groupby("unique_id", observed=True)["price"]
        .transform(lambda s: s.shift(-h) / s - 1)
        .astype(np.float32)
    )
    return df


def assign_quintiles(df: pd.DataFrame) -> pd.DataFrame:
    """Assign cross-sectional quintiles based on forward returns at each date.

    Quintiles: 1 = worst 20%, 5 = best 20%.
    Ties on boundaries get fractional quintiles (mean of tied ranks).
    """
    df = df.sort_values(["ds", "unique_id"]).copy()

    def _assign(group: pd.Series) -> pd.Series:
        r = group.rank(method="first", ascending=True)
        n = len(r)
        return pd.Series(np.ceil(r / n * 5).clip(1, 5), index=r.index)

    df["quintile"] = df.groupby("ds", observed=True)["return_fwd"].transform(_assign).astype(np.float32)
    return df


def build_long_frame(
    *,
    horizon: int | None = None,
    n_assets: int | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Build the complete Nixtla-style long frame with forward returns and quintiles.

    Columns: unique_id, ds, y, price, return_fwd, quintile, category
    """
    horizon = horizon or SETTINGS.horizon
    prices = load_prices(cache_dir=cache_dir)
    if n_assets is not None and n_assets > 0:
        kept = prices["unique_id"].drop_duplicates().sample(n=n_assets, random_state=SETTINGS.seed)
        prices = prices[prices["unique_id"].isin(kept)]
        logger.info(f"Subsampled to {n_assets:,d} assets.")

    df = compute_forward_returns(prices, horizon=horizon)
    df = assign_quintiles(df)
    df["category"] = df["unique_id"].map(ASSET_CATEGORIES).fillna("other")
    df = df.reset_index(drop=True)
    logger.info(
        f"build_long_frame: {len(df):,d} rows, {df['unique_id'].nunique():,d} assets, "
        f"{df['ds'].nunique():,d} trading days"
    )
    return df


def make_cv_cutoffs(
    df: pd.DataFrame,
    *,
    n_windows: int | None = None,
    horizon: int | None = None,
) -> list[pd.Timestamp]:
    """Generate non-overlapping cutoff dates for rolling-origin CV.

    Returns ``n_windows`` cutoff dates spaced ``horizon`` trading days apart,
    ending at the last complete training date.
    """
    h = horizon or SETTINGS.horizon
    nw = n_windows or SETTINGS.n_windows
    dates = sorted(df["ds"].unique())
    last_train = dates[-h - 1] if len(dates) > h else dates[0]
    idx = dates.index(last_train)
    step = max(h, 1)
    cutoffs = [dates[idx - i * step] for i in range(nw) if idx - i * step >= 0]
    return sorted(cutoffs)
