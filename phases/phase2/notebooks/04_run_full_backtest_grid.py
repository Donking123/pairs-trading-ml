"""
Run the full 2003-2023 Phase 2 backtest grid:
  ssd / pc × {no filter, with cointegration filter}  =  4 backtests.

Each backtest uses the same core conventions as Phase 1 (equal-dollar position
sizing, equal-weight allocation across open pairs, t+1 close-to-close
execution, Option-B delisting, no stop in core). Only the metric and the
optional cointegration filter differ across cells.

Output files (in phases/phase2/results/):
  ssd_core_monthly.parquet           ← reproduces Phase 1's 0.589 Sharpe
  ssd_core_trades.parquet
  ssd_filtered_monthly.parquet
  ssd_filtered_trades.parquet
  pc_core_monthly.parquet
  pc_core_trades.parquet
  pc_filtered_monthly.parquet
  pc_filtered_trades.parquet
  phase2_grid_log.txt                ← console log of the whole run

Expected wall-clock (laptop): ~6-8 hours for all 4 cells. Each cell is ~1-2h.

Safe to run in background:
  cd pairs-trading-ml
  nohup python phases/phase2/notebooks/04_run_full_backtest_grid.py \\
    > phases/phase2/results/phase2_grid_log.txt 2>&1 &
  tail -f phases/phase2/results/phase2_grid_log.txt

Or run one cell at a time by editing the CELLS list at the bottom.
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

from src.backtest import (
    Trade,
    load_delisting,
    run_backtest,
)
from src.config import PHASE2_DIR
from src.panel import load_crsp_daily, load_market_returns, load_sp500_constituents


RESULTS_DIR = PHASE2_DIR / "results"
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

    # save outputs
    out_monthly = RESULTS_DIR / f"{cell_name}_monthly.parquet"
    out_trades = RESULTS_DIR / f"{cell_name}_trades.parquet"
    monthly.to_parquet(out_monthly)
    trades_to_df(trades).to_parquet(out_trades)
    print(f"  saved monthly: {out_monthly}")
    print(f"  saved trades : {out_trades}  ({len(trades):,} trades)")

    # quick stats
    rets = monthly["monthly_return"]
    total = (1 + rets).prod() - 1
    ann_ret = (1 + total) ** (12 / len(rets)) - 1
    ann_vol = rets.std() * (12 ** 0.5)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    print(f"  total ret: {total:+.1%}  ann ret: {ann_ret:+.2%}  ann vol: {ann_vol:.2%}  Sharpe: {sharpe:.3f}")


# ────────────────────────────────────────────────────────────────────────────────
# 4-way grid
# ────────────────────────────────────────────────────────────────────────────────


CELLS = [
    # cell_name, metric, cointegration_filter
    ("ssd_core",     "ssd", False),   # should reproduce Phase 1's 0.589 Sharpe exactly
    ("ssd_filtered", "ssd", True),
    ("pc_core",      "pc",  False),
    ("pc_filtered",  "pc",  True),
]


def main() -> None:
    start_time = time.time()

    print("=" * 80)
    print("Phase 2 — Full 4-cell backtest grid")
    print("  variant      : core (no costs, no stop-loss; same as Phase 1)")
    print("  window       : 2003-01 → 2023-12  (251 monthly returns)")
    print(f"  grid         : {[c[0] for c in CELLS]}")
    print("=" * 80)
    print()

    # Load once, reuse for all cells
    print("[Loading data — once for all 4 cells]")
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
    print()
    print("Next step: python phases/phase2/notebooks/05_evaluate_cp2.py")


if __name__ == "__main__":
    main()
