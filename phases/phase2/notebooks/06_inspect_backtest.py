"""
Inspect the Phase 2 backtest grid — anomalies, exit reasons, outliers.

Same diagnostic spirit as phase1/notebooks/07_inspect_backtest.py but covers
all 4 cells of the Phase 2 grid side-by-side.

Reads:
  phases/phase2/results/{ssd_core, ssd_filtered, pc_core, pc_filtered}_{monthly, trades}.parquet
  phases/phase2/results/phase2_grid_log.txt   (optional)

Reports per cell:
  * Continuity check, NaN/inf returns, extreme months
  * Total trades, exit reason breakdown (the lever check)
  * Outlier trades (|rt| > 50%)

Usage: python phases/phase2/notebooks/06_inspect_backtest.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# walk up to project root
_p = Path(__file__).resolve()
while _p != _p.parent:
    if (_p / "src" / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
    _p = _p.parent
del _p

from src.config import PHASE2_DIR


CELLS = ["ssd_core", "ssd_filtered", "pc_core", "pc_filtered"]


def inspect_log() -> None:
    log_path = PHASE2_DIR / "results" / "phase2_grid_log.txt"
    print("── Log file inspection ──")
    if not log_path.exists():
        print(f"  ⚠ log not found at {log_path}")
        return
    text = log_path.read_text()
    print(f"  log file   : {log_path}")
    print(f"  size       : {len(text):,} bytes")
    skipped = re.findall(r"^\s*(\d{4}-\d{2}-\d{2}):\s*skipped\s*\((.*?)\)", text, flags=re.MULTILINE)
    errors = re.findall(r"(Error|Traceback|ERROR|FAIL)", text)
    warnings_ = re.findall(r"(Warning|UserWarning|FutureWarning)", text)
    print(f"  skipped months: {len(skipped)}")
    print(f"  errors        : {len(errors)}")
    print(f"  warnings      : {len(warnings_)}")


def inspect_cell(cell_name: str) -> None:
    monthly_path = PHASE2_DIR / "results" / f"{cell_name}_monthly.parquet"
    trades_path  = PHASE2_DIR / "results" / f"{cell_name}_trades.parquet"
    print()
    print("=" * 78)
    print(f"CELL: {cell_name}")
    print("=" * 78)
    if not monthly_path.exists():
        print(f"  ⚠ monthly not found at {monthly_path}")
        return
    monthly = pd.read_parquet(monthly_path)
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()

    print(f"  months              : {len(monthly):,}")
    print(f"  trades              : {len(trades):,}")
    rets = monthly["monthly_return"]
    nan_count = rets.isna().sum()
    print(f"  NaN returns         : {nan_count}")
    extreme = rets[rets.abs() > 0.10]
    print(f"  |return| > 10% months: {len(extreme)}")
    for d, r in extreme.head(5).items():
        print(f"    {d.date()}: {r:+.2%}")
    zero_trade_months = monthly[monthly["n_trades"] == 0] if "n_trades" in monthly.columns else pd.DataFrame()
    print(f"  zero-trade months   : {len(zero_trade_months)}")

    if not trades.empty:
        # exit reason breakdown — the load-bearing diagnostic for Phase 2
        print()
        print("  Exit reasons (the Phase 2 lever check):")
        reasons = trades["exit_reason"].value_counts()
        for r, n in reasons.items():
            sub = trades.loc[trades["exit_reason"] == r, "round_trip_return"]
            print(f"    {r:<12}  n={n:>6,}  ({n / len(trades):>5.1%})  total={sub.sum():+.2f}  mean={sub.mean() * 10000:+.0f}bps")

        # outliers
        rt = trades["round_trip_return"]
        outliers = trades.loc[rt.abs() > 0.5]
        print(f"\n  |rt| > 50% outliers : {len(outliers)}")
        for _, row in outliers.head(5).iterrows():
            print(f"    ({row['permno_a']},{row['permno_b']})  dir={row['direction']:+d}  "
                  f"{row['entry_date']:%Y-%m-%d} → {row['exit_date']:%Y-%m-%d}  "
                  f"[{row['exit_reason']:>11}]  rt={row['round_trip_return']:+.2%}")


def main() -> None:
    print("=" * 78)
    print("Phase 2 backtest grid inspection")
    print("=" * 78)
    inspect_log()
    for cell in CELLS:
        inspect_cell(cell)
    print()


if __name__ == "__main__":
    main()
