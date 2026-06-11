"""
Run the Phase 2.5 factor-beta backtest grid (2003-2023):
  factor / factor × {no filter, with cointegration filter}  =  2 backtests.

Phase 2.5 is the QF621 group's contribution: cluster stocks by their RIDGE-estimated
factor-exposure (beta) vector over 18 factors (6 FF style + 12 FF12 industry, all
built from our own CRSP/FF data — see phases/phase2_5/decisions.md), then trade the
within-cluster pairs with the same engine as Phases 1-2.

Only 2 NEW cells are run here. The head-to-head baselines (ssd_core 0.589,
pc_core 1.028) already live in phases/phase2/results/ and are NOT re-run — Phase 2
artifacts stay frozen.

Output files (in phases/phase2_5/results/):
  factor_core_monthly.parquet
  factor_core_trades.parquet
  factor_filtered_monthly.parquet
  factor_filtered_trades.parquet
  phase2_5_grid_log.txt              ← console log of the whole run

Expected wall-clock (laptop): ~1-2h per cell (cointegration filter adds time).

Safe to run in background:
  cd pairs-trading-ml
  nohup python phases/phase2_5/notebooks/01_run_factor_backtest.py \\
    > phases/phase2_5/results/phase2_5_grid_log.txt 2>&1 &
  tail -f phases/phase2_5/results/phase2_5_grid_log.txt
"""
from __future__ import annotations

import sys
import time
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

from src.backtest import Trade, load_delisting, run_backtest
from src.config import PHASE2_5_DIR
from src.panel import load_crsp_daily, load_market_returns, load_sp500_constituents


RESULTS_DIR = PHASE2_5_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([{
        "permno_a": t.permno_a,
        "permno_b": t.permno_b,
        "direction": t.direction,
        "entry_date": t.entry_date,
        "exit_date": t.exit_date,
        "entry_z": t.entry_z,
        "exit_z": t.exit_z,
        "round_trip_return": t.round_trip_return,
        "exit_reason": t.exit_reason,
    } for t in trades])


def run_one_cell(
    cell_name: str,
    metric: str,
    cointegration_filter: bool,
    crsp: pd.DataFrame,
    cons: pd.DataFrame,
    dlst: pd.DataFrame,
    mkt: pd.Series,
) -> None:
    print()
    print("=" * 80)
    print(f"CELL: {cell_name}")
    print(f"  metric              : {metric}")
    print(f"  cointegration filter: {cointegration_filter}")
    print("=" * 80)
    t0 = time.time()
    monthly, trades = run_backtest(
        start="2003-01-01",
        end="2023-12-31",
        crsp=crsp, constituents=cons, delisting_df=dlst,
        market_returns=mkt,
        metric=metric, cointegration_filter=cointegration_filter,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n  wall-clock: {elapsed / 60:.1f} min")

    out_monthly = RESULTS_DIR / f"{cell_name}_monthly.parquet"
    out_trades = RESULTS_DIR / f"{cell_name}_trades.parquet"
    monthly.to_parquet(out_monthly)
    trades_to_df(trades).to_parquet(out_trades)
    print(f"  saved monthly: {out_monthly}")
    print(f"  saved trades : {out_trades}  ({len(trades):,} trades)")

    rets = monthly["monthly_return"]
    total = (1 + rets).prod() - 1
    ann_ret = (1 + total) ** (12 / len(rets)) - 1
    ann_vol = rets.std() * (12 ** 0.5)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    print(f"  total ret: {total:+.1%}  ann ret: {ann_ret:+.2%}  "
          f"ann vol: {ann_vol:.2%}  Sharpe: {sharpe:.3f}")


CELLS = [
    # cell_name, metric, cointegration_filter
    ("factor_core",     "factor", False),
    ("factor_filtered", "factor", True),
]


def main() -> None:
    start_time = time.time()

    print("=" * 80)
    print("Phase 2.5 — Factor-beta backtest grid")
    print("  variant      : core (no costs, no stop-loss; same engine as Phase 1/2)")
    print("  window       : 2003-01 → 2023-12  (251 monthly returns)")
    print(f"  grid         : {[c[0] for c in CELLS]}")
    print("=" * 80)
    print()

    print("[Loading data — once for all cells]")
    crsp = load_crsp_daily()
    cons = load_sp500_constituents()
    dlst = load_delisting()
    mkt = load_market_returns()
    print(f"  crsp_daily  : {crsp.shape[0]:>10,} rows")
    print(f"  constituents: {cons.shape[0]:>10,} intervals")
    print(f"  delisting   : {dlst.shape[0]:>10,} events")
    print(f"  market_ret  : {mkt.shape[0]:>10,} days")

    for cell_name, metric, cf in CELLS:
        run_one_cell(cell_name, metric, cf, crsp, cons, dlst, mkt)

    total_elapsed = time.time() - start_time
    print()
    print("=" * 80)
    print(f"GRID COMPLETE — total wall-clock {total_elapsed / 60:.1f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()
