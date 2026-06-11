"""
Phase 3 — robustness backtest grid.

Re-runs the two HEADLINE metrics (PC, factor-beta) under each robustness perturbation,
to see whether Sharpe 1.028 / 1.013 survive. Baselines (pc_core, factor_core) already
exist in phase2/ and phase2_5/ — NOT re-run here.

New cells = {pc, factor} × {hdbscan, hierarchical, rlm, zweight} = 8 backtests:

  *_hdbscan       clusterer="hdbscan"        (3a — density clustering, no xi)
  *_hierarchical  clusterer="hierarchical"   (3a — avg-linkage, quantile cut)
  *_rlm           hedge_method="rlm"         (3b — robust hedge ratio)
  *_zweight       allocation="zweight"       (3c — |entry-z|-weighted sizing)

Each cell ~1-3h (hdbscan/hierarchical are denser → slower). Edit CELLS to run a subset.

Output → phases/phase3/results/{cell}_{monthly,trades}.parquet + phase3_grid_log.txt

Background run:
  cd pairs-trading-ml
  nohup python phases/phase3/notebooks/01_run_robustness_grid.py \\
    > phases/phase3/results/phase3_grid_log.txt 2>&1 &
  tail -f phases/phase3/results/phase3_grid_log.txt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

_p = Path(__file__).resolve()
while _p != _p.parent:
    if (_p / "src" / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
    _p = _p.parent
del _p

from src.backtest import Trade, load_delisting, run_backtest
from src.config import PHASE3_DIR
from src.panel import load_crsp_daily, load_market_returns, load_sp500_constituents

RESULTS_DIR = PHASE3_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([{
        "permno_a": t.permno_a, "permno_b": t.permno_b, "direction": t.direction,
        "entry_date": t.entry_date, "exit_date": t.exit_date,
        "entry_z": t.entry_z, "exit_z": t.exit_z,
        "round_trip_return": t.round_trip_return, "exit_reason": t.exit_reason,
    } for t in trades])


# cell_name, metric, kwargs-for-run_backtest
CELLS = [
    ("pc_hdbscan",         "pc",     dict(clusterer="hdbscan")),
    ("pc_hierarchical",    "pc",     dict(clusterer="hierarchical")),
    ("pc_rlm",             "pc",     dict(hedge_method="rlm")),
    ("pc_zweight",         "pc",     dict(allocation="zweight")),
    ("factor_hdbscan",     "factor", dict(clusterer="hdbscan")),
    ("factor_hierarchical","factor", dict(clusterer="hierarchical")),
    ("factor_rlm",         "factor", dict(hedge_method="rlm")),
    ("factor_zweight",     "factor", dict(allocation="zweight")),
]


def run_one_cell(cell_name, metric, kwargs, crsp, cons, dlst, mkt):
    print("\n" + "=" * 80)
    print(f"CELL: {cell_name}   metric={metric}  {kwargs}")
    print("=" * 80)
    t0 = time.time()
    monthly, trades = run_backtest(
        start="2003-01-01", end="2023-12-31",
        crsp=crsp, constituents=cons, delisting_df=dlst, market_returns=mkt,
        metric=metric, verbose=True, **kwargs,
    )
    monthly.to_parquet(RESULTS_DIR / f"{cell_name}_monthly.parquet")
    trades_to_df(trades).to_parquet(RESULTS_DIR / f"{cell_name}_trades.parquet")
    rets = monthly["monthly_return"]
    total = (1 + rets).prod() - 1
    ann_ret = (1 + total) ** (12 / len(rets)) - 1
    ann_vol = rets.std() * (12 ** 0.5)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    print(f"  wall-clock {(time.time()-t0)/60:.1f} min | {len(trades):,} trades | "
          f"ann {ann_ret:+.2%} vol {ann_vol:.2%} Sharpe {sharpe:.3f}")


def main():
    print("=" * 80)
    print("Phase 3 — robustness grid")
    print(f"  cells: {[c[0] for c in CELLS]}")
    print("=" * 80)
    crsp = load_crsp_daily(); cons = load_sp500_constituents()
    dlst = load_delisting(); mkt = load_market_returns()
    t0 = time.time()
    for cell_name, metric, kwargs in CELLS:
        run_one_cell(cell_name, metric, kwargs, crsp, cons, dlst, mkt)
    print(f"\nGRID COMPLETE — total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
