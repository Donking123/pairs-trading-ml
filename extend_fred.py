"""
Extend data/fred.parquet with BAMLH0A0HYM2 (HY OAS) back to 2000-01-01.

FRED's BAMLH0A0HYM2 series goes back to 1996; the local fred.parquet was
only pulled from 2014. Extending it unlocks regime_hmm.py to cover 2003-2025,
matching the full IS/OOS backtest horizon.

Run from repo root:
  python extend_fred.py
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
FRED_PATH = DATA_DIR / "fred.parquet"

SERIES_ID = "BAMLH0A0HYM2"
FRED_CSV_URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES_ID}"


def fetch_hy_oas(start: str = "2000-01-01", end: str = "2025-12-31") -> pd.Series:
    print(f"Fetching {SERIES_ID} from FRED public endpoint...")
    resp = requests.get(FRED_CSV_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    s = df.iloc[:, 0].replace(".", float("nan")).astype(float)
    s = s.loc[start:end].dropna()
    print(f"  Downloaded {len(s)} rows: {s.index.min().date()} to {s.index.max().date()}")
    return s


def main() -> None:
    old = pd.read_parquet(FRED_PATH)
    print(f"Existing fred.parquet: {old.index.min().date()} to {old.index.max().date()}, columns: {old.columns.tolist()}")

    hy_ext = fetch_hy_oas(start="2000-01-01")

    # Merge: extended HY OAS fills the pre-2014 gap; 2014+ values stay as-is
    merged = old.copy()
    merged[SERIES_ID] = hy_ext.combine_first(merged[SERIES_ID])

    # Backfill the other series (DGS2, DGS10, etc.) with NaN for pre-2014 — they
    # weren't available then and we don't want to invent values
    merged = merged.sort_index()

    merged.to_parquet(FRED_PATH)
    print(f"\nSaved updated fred.parquet: {merged.index.min().date()} to {merged.index.max().date()}")
    print(f"  {SERIES_ID}: {merged[SERIES_ID].dropna().index.min().date()} to {merged[SERIES_ID].dropna().index.max().date()} ({merged[SERIES_ID].notna().sum()} days)")


if __name__ == "__main__":
    main()
