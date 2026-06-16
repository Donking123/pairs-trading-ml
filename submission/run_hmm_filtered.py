"""
Run PC + stationarity filter + HMM regime overlay — IS and OOS.

This is the correct full-stack configuration:
  - metric="pc"                  : PC distance clustering (idiosyncratic residuals)
  - cointegration_filter=True    : half-life [5,60] bounds + ADF secondary screen
  - use_regime_scale=True        : HMM 3-state overlay (calm 1.0x, stressed 0.5x, crisis 0.0x)

Prerequisite: data/hmm_regimes.parquet must exist (run regime_hmm.py first).

Run from pairs-trading-ml/:
  python submission/run_hmm_filtered.py

IS (~1-2h), OOS (~20-40 min). Results cached — re-run skips finished cells.

Outputs:
  submission/results/pc_filtered_hmm_is_monthly.parquet
  submission/results/pc_filtered_hmm_oos_monthly.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import Trade, load_delisting, load_regime_scale, run_backtest
from src.panel import load_crsp_daily, load_market_returns, load_sp500_constituents
from src.performance import compute_metrics, format_metrics

IS_START  = "2003-01-01"
IS_END    = "2020-12-31"
OOS_START = "2021-01-01"
OOS_END   = "2024-12-31"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CELLS = {
    "pc_filtered_hmm_is":  dict(start=IS_START,  end=IS_END),
    "pc_filtered_hmm_oos": dict(start=OOS_START, end=OOS_END),
}

COMMON = dict(
    metric="pc",
    cointegration_filter=True,
    use_regime_scale=True,
)


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([{
        "permno_a": t.permno_a, "permno_b": t.permno_b, "direction": t.direction,
        "entry_date": t.entry_date, "exit_date": t.exit_date,
        "entry_z": t.entry_z, "exit_z": t.exit_z,
        "round_trip_return": t.round_trip_return, "exit_reason": t.exit_reason,
    } for t in trades])


def main() -> None:
    print("PC + stationarity filter + HMM regime overlay")
    print(f"  IS:  {IS_START} → {IS_END}")
    print(f"  OOS: {OOS_START} → {OOS_END}")

    print("\nLoading data...")
    data = dict(
        crsp=load_crsp_daily(),
        constituents=load_sp500_constituents(),
        delisting_df=load_delisting(),
        market_returns=load_market_returns(),
    )

    print("Loading HMM regime scale...")
    regime_scale = load_regime_scale()
    print(f"  {len(regime_scale)} days: {regime_scale.index[0].date()} → {regime_scale.index[-1].date()}")

    for name, period_kw in CELLS.items():
        out_monthly = RESULTS_DIR / f"{name}_monthly.parquet"
        out_trades  = RESULTS_DIR / f"{name}_trades.parquet"

        if out_monthly.exists():
            print(f"\n=== {name}: SKIP (cached) ===")
            monthly = pd.read_parquet(out_monthly)
        else:
            print(f"\n{'=' * 70}")
            print(f"=== {name}  ({period_kw['start']} → {period_kw['end']})")
            print(f"{'=' * 70}")
            t0 = time.time()
            monthly, trades = run_backtest(
                verbose=True,
                regime_scale=regime_scale,
                **data,
                **COMMON,
                **period_kw,
            )
            elapsed = (time.time() - t0) / 60
            monthly.to_parquet(out_monthly)
            trades_to_df(trades).to_parquet(out_trades)
            print(f"\n--- {name} done in {elapsed:.1f} min ---")

        m = compute_metrics(monthly["monthly_return"])
        print(format_metrics(m))

    # Compare against baseline variants
    print("\n" + "=" * 70)
    print("Summary vs baseline variants")
    print("=" * 70)
    comparisons = [
        ("pc_core",              "PC core (no filter)"),
        ("pc_filtered",          "PC + stationarity filter"),
        ("pc_filtered_hmm_is",   "PC + stationarity filter + HMM  [IS]"),
        ("pc_filtered_hmm_oos",  "PC + stationarity filter + HMM  [OOS]"),
    ]
    for fname, label in comparisons:
        period = "IS" if "oos" not in fname else "OOS"
        path = RESULTS_DIR / f"{fname}_monthly.parquet"
        if not path.exists():
            print(f"  {label}: not found")
            continue
        m = compute_metrics(pd.read_parquet(path)["monthly_return"])
        print(f"  {label}")
        print(f"    Sharpe {m.sharpe:.3f} | Ann.ret {m.ann_ret:.2%} | Ann.vol {m.ann_vol:.2%} | MaxDD {m.max_dd:.2%}")


if __name__ == "__main__":
    main()
