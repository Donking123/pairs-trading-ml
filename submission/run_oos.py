"""
Out-of-sample test: 2021-01 to 2023-12 (36 months).
Runs PC and factor-beta on data AFTER the in-sample cutoff.

Prerequisite: run_insample.py must have finished (for comparison numbers).

Run from pairs-trading-ml/:
  python submission/run_oos.py

Each cell ~20-40 min (shorter window). Results cached.
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
from src.performance import PerformanceMetrics, compute_metrics, format_metrics

OOS_START = "2021-01-01"
OOS_END = "2025-12-31"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CELLS = {
    "oos_pc_core":         dict(metric="pc"),
    "oos_pc_filtered":     dict(metric="pc", cointegration_filter=True),
    "oos_ssd_filtered":    dict(metric="ssd", cointegration_filter=True),
    "oos_factor_core":     dict(metric="factor"),
    "oos_factor_filtered": dict(metric="factor", cointegration_filter=True),
}


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([{
        "permno_a": t.permno_a, "permno_b": t.permno_b, "direction": t.direction,
        "entry_date": t.entry_date, "exit_date": t.exit_date,
        "entry_z": t.entry_z, "exit_z": t.exit_z,
        "round_trip_return": t.round_trip_return, "exit_reason": t.exit_reason,
    } for t in trades])


def load_is_sharpe(name: str) -> float | None:
    path = RESULTS_DIR / f"{name}_monthly.parquet"
    if path.exists():
        m = compute_metrics(pd.read_parquet(path)["monthly_return"])
        return round(m.sharpe, 3)
    return None


def main() -> None:
    print(f"Out-of-sample period: {OOS_START} to {OOS_END}")
    data_dir = ROOT / "data_through_2025"
    if not (data_dir / "crsp_daily.parquet").exists():
        raise SystemExit(
            f"No data in {data_dir}. Pull it first:\n"
            f"  python src/wrds_pull.py --start 2000-01-01 --end 2025-12-31 "
            f"--data-dir data_through_2025")

    print(f"Loading extended data from {data_dir} ...")
    data = dict(
        crsp=load_crsp_daily(data_dir),
        constituents=load_sp500_constituents(data_dir),
        delisting_df=load_delisting(data_dir),
        market_returns=load_market_returns(data_dir),
    )

    oos_results: dict[str, PerformanceMetrics] = {}

    for name, kwargs in CELLS.items():
        out_monthly = RESULTS_DIR / f"{name}_monthly.parquet"
        out_trades = RESULTS_DIR / f"{name}_trades.parquet"

        if out_monthly.exists():
            print(f"\n=== {name}: loading cached result ===")
            monthly = pd.read_parquet(out_monthly)
        else:
            print(f"\n{'=' * 70}")
            print(f"=== {name}  ({OOS_START} → {OOS_END})")
            print(f"{'=' * 70}")
            t0 = time.time()
            monthly, trades = run_backtest(
                start=OOS_START, end=OOS_END, verbose=True, **data, **kwargs,
            )
            elapsed = (time.time() - t0) / 60
            monthly.to_parquet(out_monthly)
            trades_to_df(trades).to_parquet(out_trades)
            print(f"\n--- {name} done in {elapsed:.1f} min ---")

        m = compute_metrics(monthly["monthly_return"])
        oos_results[name] = m
        print(format_metrics(m))

    # ── comparison table ──
    print(f"\n{'=' * 70}")
    print("IN-SAMPLE (2003–2020) vs OUT-OF-SAMPLE (2021–2025)")
    print(f"{'=' * 70}")

    pc_is = load_is_sharpe("pc_core")
    factor_is = load_is_sharpe("pc_core")  # placeholder if factor IS wasn't run

    print(f"{'Metric':<20} {'IS Sharpe':>12} {'OOS Sharpe':>12} {'OOS Ann.Ret':>12} {'OOS MDD':>12} {'OOS Months':>12}")
    print("-" * 80)

    for name, m in oos_results.items():
        is_name = name.replace("oos_", "")
        is_sr = load_is_sharpe(is_name)
        is_str = f"{is_sr:.3f}" if is_sr else "N/A"
        print(f"{name:<20} {is_str:>12} {m.sharpe:>12.3f} {m.ann_return:>12.2%} {m.max_drawdown:>12.2%} {m.n_months:>12}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
