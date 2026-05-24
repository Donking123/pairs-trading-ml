"""
Run the full 2003-2023 SSD backtest and save outputs to disk.

Output files (in results/):
  ssd_core_monthly.parquet     One row per month: return + diagnostics
  ssd_core_trades.parquet      Every round-trip trade across the sample
  ssd_core_log.txt             Console log of the run

Expected wall-clock: ~1-2 hours on a laptop. Safe to run in background:

  cd pairs-trading-ml
  nohup python notebooks/05_run_full_backtest.py > results/ssd_core_log.txt 2>&1 &

Then `tail -f results/ssd_core_log.txt` to watch progress.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.backtest import run_backtest
from src.config import RESULTS_DIR


def main() -> None:
    start_time = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("Full SSD backtest — paper-faithful core variant")
    print("  variant       : core (no costs, no stop-loss)")
    print("  metric        : SSD distance")
    print("  algo          : OPTICS (xi=0.10, locked)")
    print("  cointegration : OFF (Phase 2 filter not yet wired)")
    print("  window        : 2003-01 → 2023-12  (252 monthly returns)")
    print("=" * 80)
    print()

    monthly, trades = run_backtest(
        start="2003-01-01",
        end="2023-12-31",
        verbose=True,
    )

    elapsed = time.time() - start_time
    print(f"\nWall-clock: {elapsed/60:.1f} min")

    # save
    out_monthly = RESULTS_DIR / "ssd_core_monthly.parquet"
    out_trades = RESULTS_DIR / "ssd_core_trades.parquet"
    monthly.to_parquet(out_monthly)
    pd.DataFrame([{
        "permno_a": t.permno_a,
        "permno_b": t.permno_b,
        "direction": t.direction,
        "entry_date": t.entry_date,
        "exit_date": t.exit_date,
        "entry_z": t.entry_z,
        "exit_z": t.exit_z,
        "round_trip_return": t.round_trip_return,
        "exit_reason": t.exit_reason,
    } for t in trades]).to_parquet(out_trades)

    print(f"\nSaved:")
    print(f"  {out_monthly}    ({len(monthly)} months)")
    print(f"  {out_trades}     ({len(trades):,} trades)")
    print()

    # quick top-line stats
    rets = monthly["monthly_return"]
    total = (1 + rets).prod() - 1
    ann_ret = (1 + total) ** (12 / len(rets)) - 1
    ann_vol = rets.std() * (12 ** 0.5)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    cum = (1 + rets).cumprod()
    drawdown = (cum / cum.cummax() - 1).min()

    print(f"Quick stats (no costs, no stop, gross):")
    print(f"  total return    : {total:+.1%}")
    print(f"  annualised ret  : {ann_ret:+.2%}")
    print(f"  annualised vol  : {ann_vol:.2%}")
    print(f"  Sharpe (rf=0)   : {sharpe:.2f}    (paper target: 0.88)")
    print(f"  max drawdown    : {drawdown:.1%}")
    print(f"  hit rate        : {(rets > 0).mean():.1%}")
    print(f"  total trades    : {len(trades):,}")


if __name__ == "__main__":
    main()
