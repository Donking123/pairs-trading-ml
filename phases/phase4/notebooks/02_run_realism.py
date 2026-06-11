"""
Phase 4a — realism variant backtest.

Re-runs the two headline strategies (PC core, factor-beta core) with ALL frictions on:
  * transaction costs = actual CRSP bid/ask half-spread at entry & exit,
  * borrow = 35 bps annual on the short leg,
  * 3.5σ stop-loss.
Then prints the Sharpe vs the frictionless baselines (pc_core 1.028, factor_core 1.013) to
show whether the edge survives real costs.

Output → phases/phase4/results/{pc,factor}_realism_{monthly,trades}.parquet
         + realism_summary printed (compare with phase2/phase2_5 baselines).

Run (each ~1-3h; transaction costs add a bit):
  cd pairs-trading-ml
  nohup python phases/phase4/notebooks/02_run_realism.py \\
    > phases/phase4/results/realism_run.log 2>&1 &
  tail -f phases/phase4/results/realism_run.log
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
from src.config import PHASE2_DIR, PHASE2_5_DIR, PHASE4_DIR
from src.costs import RealismConfig
from src.panel import load_crsp_daily, load_market_returns, load_sp500_constituents
from src.performance import compute_metrics

RESULTS_DIR = PHASE4_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Realism preset (proposal): bid/ask costs + 35bps borrow + 3.5σ stop.
REALISM = RealismConfig(transaction_costs=True, borrow_bps_annual=35.0,
                        default_spread_bps=10.0)
STOP_SIGMA = 3.5

CELLS = [
    ("pc_realism",     "pc",     PHASE2_DIR / "results" / "pc_core_monthly.parquet"),
    ("factor_realism", "factor", PHASE2_5_DIR / "results" / "factor_core_monthly.parquet"),
]


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([{
        "permno_a": t.permno_a, "permno_b": t.permno_b, "direction": t.direction,
        "entry_date": t.entry_date, "exit_date": t.exit_date,
        "entry_z": t.entry_z, "exit_z": t.exit_z,
        "round_trip_return": t.round_trip_return, "exit_reason": t.exit_reason,
    } for t in trades])


def main() -> None:
    crsp = load_crsp_daily(); cons = load_sp500_constituents()
    dlst = load_delisting(); mkt = load_market_returns()

    rows = []
    for cell, metric, base_path in CELLS:
        print("\n" + "=" * 78)
        print(f"CELL: {cell}  (metric={metric}, costs+borrow+3.5σ stop)")
        print("=" * 78)
        t0 = time.time()
        monthly, trades = run_backtest(
            start="2003-01-01", end="2023-12-31", crsp=crsp, constituents=cons,
            delisting_df=dlst, market_returns=mkt, metric=metric,
            stop_sigma=STOP_SIGMA, realism=REALISM, verbose=True)
        monthly.to_parquet(RESULTS_DIR / f"{cell}_monthly.parquet")
        trades_to_df(trades).to_parquet(RESULTS_DIR / f"{cell}_trades.parquet")

        m = compute_metrics(monthly["monthly_return"].astype(float))
        base_sharpe = float("nan")
        if base_path.exists():
            base_sharpe = compute_metrics(
                pd.read_parquet(base_path)["monthly_return"].astype(float)).sharpe
        print(f"  wall-clock {(time.time()-t0)/60:.1f} min | {len(trades):,} trades")
        print(f"  realism Sharpe {m.sharpe:.3f}  vs frictionless {base_sharpe:.3f}  "
              f"(Δ {m.sharpe-base_sharpe:+.3f}) | ann {m.ann_return:+.2%} vol {m.ann_vol:.2%}")
        rows.append({"cell": cell, "frictionless_sharpe": round(base_sharpe, 3),
                     "realism_sharpe": round(m.sharpe, 3),
                     "delta": round(m.sharpe - base_sharpe, 3),
                     "ann_ret": m.ann_return, "ann_vol": m.ann_vol,
                     "mdd": m.max_drawdown, "n_trades": len(trades)})

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "realism_summary.csv", index=False)
    print("\n" + "=" * 78)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
