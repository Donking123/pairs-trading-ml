"""
In-sample backtest: 2003-01 to 2020-12 (216 months).
Runs the full 2x2 grid: {SSD, PC} x {core, filtered}.

Run from pairs-trading-ml/:
  python submission/run_insample.py

Each cell ~1-2 hours, total ~4-8 hours. Results cached — re-run skips finished cells.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import Trade, load_delisting, run_backtest
from src.panel import load_crsp_daily, load_market_returns, load_sp500_constituents
from src.performance import compute_metrics, format_metrics

IS_START = "2003-01-01"
IS_END = "2020-12-31"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CELLS = {
    "ssd_core":        dict(metric="ssd"),
    "ssd_filtered":    dict(metric="ssd", cointegration_filter=True),
    "pc_core":         dict(metric="pc"),
    "pc_filtered":     dict(metric="pc", cointegration_filter=True),
    "factor_core":     dict(metric="factor"),
    "factor_filtered": dict(metric="factor", cointegration_filter=True),
}


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([{
        "permno_a": t.permno_a, "permno_b": t.permno_b, "direction": t.direction,
        "entry_date": t.entry_date, "exit_date": t.exit_date,
        "entry_z": t.entry_z, "exit_z": t.exit_z,
        "round_trip_return": t.round_trip_return, "exit_reason": t.exit_reason,
    } for t in trades])


def main() -> None:
    print(f"In-sample period: {IS_START} to {IS_END}")
    print("Loading data...")
    data = dict(
        crsp=load_crsp_daily(),
        constituents=load_sp500_constituents(),
        delisting_df=load_delisting(),
        market_returns=load_market_returns(),
    )

    for name, kwargs in CELLS.items():
        out_monthly = RESULTS_DIR / f"{name}_monthly.parquet"
        out_trades = RESULTS_DIR / f"{name}_trades.parquet"

        if out_monthly.exists():
            print(f"\n=== {name}: SKIP (cached) ===")
            monthly = pd.read_parquet(out_monthly)
        else:
            print(f"\n{'=' * 70}")
            print(f"=== {name}  ({IS_START} → {IS_END})")
            print(f"{'=' * 70}")
            t0 = time.time()
            monthly, trades = run_backtest(
                start=IS_START, end=IS_END, verbose=True, **data, **kwargs,
            )
            elapsed = (time.time() - t0) / 60
            monthly.to_parquet(out_monthly)
            trades_to_df(trades).to_parquet(out_trades)
            print(f"\n--- {name} done in {elapsed:.1f} min ---")

        m = compute_metrics(monthly["monthly_return"])
        print(format_metrics(m))

    print("\n✓ All in-sample cells complete. Results in submission/results/")
    print("Next: python submission/run_oos.py")


if __name__ == "__main__":
    main()
