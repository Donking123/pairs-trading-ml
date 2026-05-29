"""
spx_data.py — Download and cache S&P 500 universe + factor ETFs via yfinance.

Universe: S&P 500 constituent list scraped from Wikipedia; filtered to top-N
by 63d rolling average dollar volume (ADV). Adjusted close + volume cached to
parquet so repeated runs skip the network call.

Factor ETF set (~17 ETFs as specified in EquityPairsTradingProposal.md):
  Market: SPY
  Sectors (11): XLB XLE XLF XLI XLK XLP XLU XLV XLY XLC XLRE
  Styles (2):   MTUM VTV
  Macro (4):    TLT GLD UUP USO
  Optional:     XBI (biotech — toggle via INCLUDE_XBI flag)

Data range: 2014-01-01 to present (proposal's full backtest horizon).
Parquet cache lives in QTS Lectures/data/ alongside this file.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SPX_CLOSE_CACHE = DATA_DIR / "spx_close.parquet"
SPX_VOLUME_CACHE = DATA_DIR / "spx_volume.parquet"
FACTOR_CLOSE_CACHE = DATA_DIR / "factor_close.parquet"

START_DATE = "2014-01-01"
UNIVERSE_N = 300
ADV_WINDOW = 63

FACTOR_ETFS = [
    "SPY",
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLC", "XLRE",
    "MTUM", "VTV",
    "TLT", "GLD", "UUP", "USO",
]
INCLUDE_XBI = False


# ── SPX ticker list ────────────────────────────────────────────────────

def get_spx_tickers() -> list[str]:
    """Scrape current S&P 500 tickers from Wikipedia with a real User-Agent.

    Falls back to yfinance's built-in S&P500 tickers list if scraping fails.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        # Try direct read_html first
        tables = pd.read_html(url, header=0)
        df = tables[0]
    except Exception:
        # Retry with requests + UA to avoid 403 blocks; if lxml not present, fall back to yfinance
        try:
            import requests
            from io import StringIO

            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"}
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            html = r.text
            tables = pd.read_html(StringIO(html), header=0)
            df = tables[0]
        except Exception:
            try:
                import yfinance as yf

                tickers = yf.tickers_sp500()
                return [t.replace(".", "-") for t in tickers]
            except Exception:
                raise RuntimeError("Unable to obtain S&P 500 tickers from Wikipedia or yfinance.")

    if "Symbol" not in df.columns:
        # Fallback: yfinance has a helper for SP500 tickers
        try:
            import yfinance as yf

            tickers = yf.tickers_sp500()
            return [t.replace(".", "-") for t in tickers]
        except Exception:
            raise RuntimeError("Unable to obtain S&P 500 tickers from Wikipedia or yfinance.")

    tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).str.strip().tolist()
    return tickers


# ── Download helpers ───────────────────────────────────────────────────

def _download_batch(
    tickers: list[str],
    start: str,
    fields: list[str] = ("Close", "Volume"),
    chunk: int = 100,
) -> dict[str, pd.DataFrame]:
    """Download yfinance data in chunks; return dict {field: wide DataFrame}."""
    results: dict[str, list[pd.DataFrame]] = {f: [] for f in fields}

    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                batch,
                start=start,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        if raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            for f in fields:
                if f in raw.columns.get_level_values(0):
                    df = raw[f]
                    df.columns = [str(c) for c in df.columns]
                    results[f].append(df)
        else:
            # Single ticker returned flat columns
            tkr = batch[0]
            for f in fields:
                if f in raw.columns:
                    df = raw[[f]].rename(columns={f: tkr})
                    results[f].append(df)

    out: dict[str, pd.DataFrame] = {}
    for f in fields:
        if results[f]:
            combined = pd.concat(results[f], axis=1)
            combined = combined.loc[:, ~combined.columns.duplicated()]
            combined.index = pd.to_datetime(combined.index).tz_localize(None)
            out[f] = combined.sort_index()
        else:
            out[f] = pd.DataFrame()
    return out


# ── Universe download ──────────────────────────────────────────────────

def download_spx_universe(
    n: int = UNIVERSE_N,
    start: str = START_DATE,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download SPX universe: returns (close_df, volume_df) filtered to top-n by ADV.

    Cached to parquet; only re-downloads when refresh=True or cache missing.
    Returns:
        close_df:  (dates × tickers) adjusted close prices
        volume_df: (dates × tickers) daily volume
    """
    if not refresh and SPX_CLOSE_CACHE.exists() and SPX_VOLUME_CACHE.exists():
        close_df = pd.read_parquet(SPX_CLOSE_CACHE)
        volume_df = pd.read_parquet(SPX_VOLUME_CACHE)
        return close_df, volume_df

    print("Fetching S&P 500 tickers from Wikipedia...")
    all_tickers = get_spx_tickers()
    print(f"  Found {len(all_tickers)} tickers. Downloading prices...")

    data = _download_batch(all_tickers, start=start, fields=["Close", "Volume"])
    close_all = data["Close"].dropna(how="all", axis=1)
    volume_all = data["Volume"].dropna(how="all", axis=1)

    # Filter to top-n by trailing ADV (dollar volume)
    common = close_all.columns.intersection(volume_all.columns)
    close_all = close_all[common]
    volume_all = volume_all[common]

    dollar_vol = close_all * volume_all
    adv = dollar_vol.tail(ADV_WINDOW).mean()
    top_tickers = adv.nlargest(n).index.tolist()

    close_df = close_all[top_tickers]
    volume_df = volume_all[top_tickers]

    close_df.to_parquet(SPX_CLOSE_CACHE)
    volume_df.to_parquet(SPX_VOLUME_CACHE)
    print(f"  Cached {len(top_tickers)} tickers to {DATA_DIR}/")

    return close_df, volume_df


# ── Factor ETF download ────────────────────────────────────────────────

def download_factor_etfs(
    start: str = START_DATE,
    refresh: bool = False,
    include_xbi: bool = INCLUDE_XBI,
) -> pd.DataFrame:
    """Download factor ETF adjusted closes. Returns (dates × ETFs) DataFrame.

    Cached to parquet.
    """
    if not refresh and FACTOR_CLOSE_CACHE.exists():
        return pd.read_parquet(FACTOR_CLOSE_CACHE)

    tickers = FACTOR_ETFS + (["XBI"] if include_xbi else [])
    print(f"Downloading {len(tickers)} factor ETFs...")

    data = _download_batch(tickers, start=start, fields=["Close"])
    factor_df = data["Close"].dropna(how="all", axis=1).sort_index()

    missing = set(tickers) - set(factor_df.columns)
    if missing:
        warnings.warn(f"Factor ETFs not downloaded: {missing}")

    factor_df.to_parquet(FACTOR_CLOSE_CACHE)
    print(f"  Cached factor ETFs to {FACTOR_CLOSE_CACHE}")
    return factor_df


# ── Convenience loader ─────────────────────────────────────────────────

def load_all(
    n: int = UNIVERSE_N,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load everything needed for feature matrix construction.

    Returns:
        stock_close:   (dates × tickers) adjusted close
        stock_volume:  (dates × tickers) volume
        stock_returns: (dates × tickers) simple daily returns
        factor_close:  (dates × ETFs) factor ETF adjusted close
        factor_returns:(dates × ETFs) factor ETF simple daily returns
    """
    stock_close, stock_volume = download_spx_universe(n=n, refresh=refresh)
    factor_close = download_factor_etfs(refresh=refresh)

    stock_returns = stock_close.pct_change()
    factor_returns = factor_close.pct_change()

    # Align to common date range
    common_idx = stock_returns.index.intersection(factor_returns.index)
    stock_returns = stock_returns.loc[common_idx]
    factor_returns = factor_returns.loc[common_idx]
    stock_close = stock_close.loc[common_idx]
    stock_volume = stock_volume.loc[common_idx]
    factor_close = factor_close.loc[common_idx]

    return stock_close, stock_volume, stock_returns, factor_close, factor_returns


if __name__ == "__main__":
    close, volume, rets, fclose, frets = load_all()
    print(f"Stock universe : {close.shape[1]} tickers, {close.shape[0]} days")
    print(f"Date range     : {close.index[0].date()} → {close.index[-1].date()}")
    print(f"Factor ETFs    : {list(frets.columns)}")
    print(f"Factor days    : {frets.shape[0]}")
