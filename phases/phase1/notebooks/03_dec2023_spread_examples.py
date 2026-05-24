"""
Phase 1b on real Dec-2023 pairs — fit γ, build spread, compute z-score
for a handful of showcase pairs from the Dec-2023 clustering output.

Pairs chosen (from notebook 01 output):
  GOOG  / GOOGL   — dual share class (γ should be ≈ 1.0)
  XOM   / CVX     — oil majors
  MA    / V       — payment networks
  MCO   / SPGI    — rating agencies
  GS    / MS      — investment banks
  CDNS  / SNPS    — EDA software

For each pair we print:
  * γ, α from OLS on the 756-day formation window
  * implied position ratio (γ × price_B / price_A)
  * formation-window spread mean/std
  * how many days in the formation window had |z| ≥ 2 (a rough sense of how
    often the spread historically diverged)

Run:  python -m notebooks.03_dec2023_spread_examples
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config import ZSCORE_WINDOW_MONTHS
from src.panel import (
    formation_window_panel,
    load_crsp_daily,
    load_sp500_constituents,
    ticker_lookup,
)
from src.spread import fit_hedge_ratio, rolling_zscore, spread_series


AS_OF = "2023-12-29"
ZSCORE_WINDOW = ZSCORE_WINDOW_MONTHS * 21  # ~126 trading days

# tickers to showcase, in order
SHOWCASE = [
    ("GOOG", "GOOGL"),
    ("XOM",  "CVX"),
    ("MA",   "V"),
    ("MCO",  "SPGI"),
    ("GS",   "MS"),
    ("CDNS", "SNPS"),
]


def find_permno(ticker: str, ticker_map: pd.Series) -> int | None:
    matches = ticker_map.index[ticker_map == ticker].tolist()
    return matches[0] if matches else None


def main() -> None:
    print("=" * 90)
    print(f"Phase 1b — spread.py on real Dec-2023 pairs")
    print("=" * 90)

    crsp = load_crsp_daily()
    cons = load_sp500_constituents()
    panel = formation_window_panel(AS_OF, crsp=crsp, constituents=cons)
    ticker_map = ticker_lookup(panel.columns.tolist(), crsp=crsp, as_of=pd.Timestamp(AS_OF))

    print(f"\nFormation window : {panel.index.min().date()} → {panel.index.max().date()}  ({panel.shape[0]} days)")
    print(f"Universe          : {panel.shape[1]} stocks")
    print(f"z-score lookback  : {ZSCORE_WINDOW} trading days (~6 months)")
    print()
    print(
        f"{'pair':<14} | {'γ':>8} | {'α':>9} | {'resid σ':>8} | "
        f"{'dollar-hedge ratio':>20} | {'|z|≥2 days':>10}"
    )
    print(
        f"{'-' * 14}-+-{'-' * 8}-+-{'-' * 9}-+-{'-' * 8}-+-{'-' * 20}-+-{'-' * 10}"
    )

    for ta, tb in SHOWCASE:
        pa = find_permno(ta, ticker_map)
        pb = find_permno(tb, ticker_map)
        if pa is None or pb is None:
            print(f"{ta + '/' + tb:<14} | (missing from universe)")
            continue

        prices_a = panel[pa]
        prices_b = panel[pb]
        fit = fit_hedge_ratio(prices_a, prices_b)
        spread = spread_series(prices_a, prices_b, fit)
        z = rolling_zscore(spread, window=ZSCORE_WINDOW)

        # dollar-hedge ratio at the last day = γ * P_B / P_A
        last_a = float(prices_a.iloc[-1])
        last_b = float(prices_b.iloc[-1])
        dollar_ratio = fit.gamma * last_b / last_a

        n_breaches = int((z.dropna().abs() >= 2.0).sum())

        print(
            f"{ta + '/' + tb:<14} | "
            f"{fit.gamma:>8.4f} | "
            f"{fit.alpha:>9.3f} | "
            f"{fit.residual_std:>8.3f} | "
            f"{dollar_ratio:>20.3f} | "
            f"{n_breaches:>10}"
        )

    # show one full spread series in detail for GOOG/GOOGL (the cleanest case)
    print("\n" + "=" * 90)
    print("Detail — GOOG / GOOGL spread (last 10 days of formation window)")
    print("=" * 90)
    pa = find_permno("GOOG", ticker_map)
    pb = find_permno("GOOGL", ticker_map)
    prices_a = panel[pa]
    prices_b = panel[pb]
    fit = fit_hedge_ratio(prices_a, prices_b)
    spread = spread_series(prices_a, prices_b, fit)
    z = rolling_zscore(spread, window=ZSCORE_WINDOW)

    last10 = pd.DataFrame({
        "GOOG": prices_a.iloc[-10:].round(2),
        "GOOGL": prices_b.iloc[-10:].round(2),
        f"spread = GOOG - {fit.gamma:.3f}·GOOGL": spread.iloc[-10:].round(3),
        f"z (window={ZSCORE_WINDOW}d)": z.iloc[-10:].round(3),
    })
    print(last10.to_string())
    print(f"\nformation γ={fit.gamma:.4f}, α={fit.alpha:.3f}, resid σ={fit.residual_std:.3f}")
    print(f"latest z={z.iloc[-1]:.3f}  (|z|≥2 would trigger entry)")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
