"""
Carry-over backtest: trades roll over across months instead of force-closing.

Uses carry_over=True, max_carry_months=3 (default from config).
Runs filtered strategies on IS (2003-2020) and OOS (2021-2025).

Run from pairs-trading-ml/:
  python submission/run_carryover.py
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

IS_START = "2003-01-01"
IS_END = "2020-12-31"
OOS_START = "2021-01-01"
OOS_END = "2025-12-31"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CELLS = {
    # IS carry-over filtered
    "carry_ssd_filtered":    dict(metric="ssd", cointegration_filter=True, start=IS_START, end=IS_END, data_key="is"),
    "carry_pc_filtered":     dict(metric="pc",  cointegration_filter=True, start=IS_START, end=IS_END, data_key="is"),
    "carry_factor_filtered": dict(metric="factor", cointegration_filter=True, start=IS_START, end=IS_END, data_key="is"),
    # OOS carry-over filtered
    "carry_oos_ssd_filtered":    dict(metric="ssd", cointegration_filter=True, start=OOS_START, end=OOS_END, data_key="oos"),
    "carry_oos_pc_filtered":     dict(metric="pc",  cointegration_filter=True, start=OOS_START, end=OOS_END, data_key="oos"),
    "carry_oos_factor_filtered": dict(metric="factor", cointegration_filter=True, start=OOS_START, end=OOS_END, data_key="oos"),
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
    print("=== Carry-over backtest (no force-close at month-end) ===")
    print(f"IS: {IS_START} to {IS_END} | OOS: {OOS_START} to {OOS_END}")
    print(f"carry_over=True, max_carry_months=3\n")

    print("Loading IS data...")
    is_data = dict(
        crsp=load_crsp_daily(),
        constituents=load_sp500_constituents(),
        delisting_df=load_delisting(),
        market_returns=load_market_returns(),
    )

    oos_data_dir = ROOT / "data_through_2025"
    print(f"Loading OOS data from {oos_data_dir} ...")
    oos_data = dict(
        crsp=load_crsp_daily(oos_data_dir),
        constituents=load_sp500_constituents(oos_data_dir),
        delisting_df=load_delisting(oos_data_dir),
        market_returns=load_market_returns(oos_data_dir),
    )

    datasets = {"is": is_data, "oos": oos_data}
    results: dict[str, PerformanceMetrics] = {}

    for name, cell in CELLS.items():
        out_monthly = RESULTS_DIR / f"{name}_monthly.parquet"
        out_trades = RESULTS_DIR / f"{name}_trades.parquet"

        if out_monthly.exists():
            print(f"\n=== {name}: loading cached result ===")
            monthly = pd.read_parquet(out_monthly)
        else:
            start = cell.pop("start")
            end = cell.pop("end")
            data_key = cell.pop("data_key")
            data = datasets[data_key]

            print(f"\n{'=' * 70}")
            print(f"=== {name}  ({start} → {end})  carry_over=True")
            print(f"{'=' * 70}")
            t0 = time.time()
            monthly, trades = run_backtest(
                start=start, end=end, verbose=True,
                carry_over=True,
                **data, **cell,
            )
            elapsed = (time.time() - t0) / 60
            monthly.to_parquet(out_monthly)
            trades_to_df(trades).to_parquet(out_trades)
            print(f"\n--- {name} done in {elapsed:.1f} min ---")

        m = compute_metrics(monthly["monthly_return"])
        results[name] = m
        print(format_metrics(m))

    # ── comparison table: carry-over vs no-carry ──
    print(f"\n{'=' * 80}")
    print("CARRY-OVER vs NO-CARRY (filtered strategies)")
    print(f"{'=' * 80}")
    print(f"{'Strategy':<28} {'No-carry Sharpe':>15} {'Carry Sharpe':>13} {'Delta':>8}")
    print("-" * 70)

    comparisons = {
        "carry_ssd_filtered":        "ssd_filtered",
        "carry_pc_filtered":         "pc_filtered",
        "carry_factor_filtered":     "factor_filtered",
        "carry_oos_ssd_filtered":    "oos_ssd_filtered",
        "carry_oos_pc_filtered":     "oos_pc_filtered",
        "carry_oos_factor_filtered": "oos_factor_filtered",
    }

    for carry_name, base_name in comparisons.items():
        if carry_name not in results:
            continue
        carry_sr = results[carry_name].sharpe
        base_sr = load_is_sharpe(base_name)
        base_str = f"{base_sr:.3f}" if base_sr is not None else "N/A"
        delta = f"{carry_sr - base_sr:+.3f}" if base_sr is not None else "N/A"
        period = "OOS" if "oos" in carry_name else "IS"
        label = carry_name.replace("carry_", "").replace("oos_", "")
        print(f"{label} ({period}){'':<12} {base_str:>15} {carry_sr:>13.3f} {delta:>8}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
