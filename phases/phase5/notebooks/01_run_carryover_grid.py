"""
Phase 5 — position carry-over grid.

Tests two improvements over the force-close-every-month engine:
  * CARRY-OVER : hold a still-open position into next month (capped at MAX_CARRY_MONTHS,
                 tied to the 60-trading-day half-life ceiling) instead of force-closing it.
                 88.4% of trades exited via force-close at a mean loss — the biggest drag.
  * PASSIVE execution : limit orders (spread_cost_multiplier=0.5), the Phase-4 cost-opt
                 winner (+~0.20 net Sharpe).

Cells (each run over all of --metrics):
  E  carry      + frictionless     → isolates the SIGNAL effect vs the stored F baseline
  B  no-carry   + passive (no stop)→ realistic operating point WITHOUT carry
  D  carry      + passive (no stop)→ headline: realistic operating point WITH carry
PC-only sensitivity (run when 'pc' in --metrics and --sens):
  Dstop carry + passive + 3.5σ stop  → does the stop still hurt under carry?
  Dmkt  carry + marketable (no stop) → cost sensitivity vs passive

The frictionless no-carry baseline (cell F) is NOT re-run — it already exists as the
phase1/2/2.5 *core* parquets and is read by 02_evaluate_phase5.py.

FAIL-FAST: run the single most informative cell first and stop if carry doesn't help —
    python phases/phase5/notebooks/01_run_carryover_grid.py --metrics pc --cells E
If PC-E frictionless does not beat the stored PC core (1.028), the hypothesis is wrong and
the rest of the grid is moot. Otherwise run the whole thing (background; each cell ~1-3h,
cost cells slower):
    nohup python phases/phase5/notebooks/01_run_carryover_grid.py \\
        > phases/phase5/results/grid.log 2>&1 &
    tail -f phases/phase5/results/grid.log
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

_p = Path(__file__).resolve()
while _p != _p.parent:
    if (_p / "src" / "config.py").exists():
        sys.path.insert(0, str(_p))
        ROOT = _p
        break
    _p = _p.parent
del _p

from src.backtest import Trade, load_delisting, run_backtest
from src.config import PHASE5_DIR
from src.costs import REALISM_FULL, REALISM_PASSIVE
from src.panel import load_crsp_daily, load_market_returns, load_sp500_constituents
from src.performance import compute_metrics

RESULTS_DIR = PHASE5_DIR / "results"

# stored frictionless no-carry core Sharpe — the reference each carry cell is judged against
INSAMPLE_F = {"pc": 1.028, "factor": 1.013, "ssd": 0.589}

# cell -> run_backtest kwargs (metric injected per loop). Omitting `realism` = frictionless;
# stop_sigma defaults to None (no stop) — the Phase-4 best operating point.
CELLS = {
    "E": dict(carry_over=True),                                         # carry, frictionless
    "B": dict(carry_over=False, realism=REALISM_PASSIVE),              # no-carry, passive
    "D": dict(carry_over=True,  realism=REALISM_PASSIVE),              # carry, passive (headline)
}
PC_SENS = {
    "Dstop": dict(carry_over=True, realism=REALISM_PASSIVE, stop_sigma=3.5),  # +3.5σ stop
    "Dmkt":  dict(carry_over=True, realism=REALISM_FULL),                     # marketable
}


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([{
        "permno_a": t.permno_a, "permno_b": t.permno_b, "direction": t.direction,
        "entry_date": t.entry_date, "exit_date": t.exit_date,
        "entry_z": t.entry_z, "exit_z": t.exit_z,
        "round_trip_return": t.round_trip_return, "exit_reason": t.exit_reason,
    } for t in trades])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="pc,factor,ssd",
                    help="comma list of metrics to run (pc,factor,ssd)")
    ap.add_argument("--cells", default="E,B,D",
                    help="comma list of cells to run (E,B,D[,Dstop,Dmkt])")
    ap.add_argument("--sens", action="store_true",
                    help="also run the PC-only sensitivity cells (Dstop, Dmkt)")
    ap.add_argument("--start", default="2003-01-01")
    ap.add_argument("--end", default="2023-12-31")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    print("Loading data ...")
    crsp = load_crsp_daily(); cons = load_sp500_constituents()
    dlst = load_delisting(); mkt = load_market_returns()

    # build the run list. CELLS run for every metric; PC_SENS cells are PC-only and can be
    # selected either explicitly (--cells Dstop,Dmkt) or all at once (--sens).
    runs: list[tuple[str, str, dict]] = []
    seen: set[tuple[str, str]] = set()

    def _add(cell: str, metric: str, kw: dict) -> None:
        if (cell, metric) not in seen:
            seen.add((cell, metric))
            runs.append((cell, metric, kw))

    for metric in metrics:
        for cell in cells:
            if cell in CELLS:
                _add(cell, metric, CELLS[cell])
            elif cell in PC_SENS:
                if metric == "pc":
                    _add(cell, "pc", PC_SENS[cell])
                else:
                    print(f"  (skip {cell} for {metric}: sensitivity cells are PC-only)")
            else:
                print(f"  (skip unknown cell {cell!r})")
    if args.sens and "pc" in metrics:
        for cell, kw in PC_SENS.items():
            _add(cell, "pc", kw)

    if not runs:
        print("  no runs selected — check --metrics / --cells "
              f"(valid: {list(CELLS)} + PC-only {list(PC_SENS)})")
        return

    rows = []
    for cell, metric, kwargs in runs:
        tag = f"{cell}_{metric}"
        print("\n" + "=" * 78)
        print(f"CELL {tag}: {kwargs}")
        print("=" * 78)
        t0 = time.time()
        monthly, trades = run_backtest(
            start=args.start, end=args.end, crsp=crsp, constituents=cons,
            delisting_df=dlst, market_returns=mkt, metric=metric, verbose=True, **kwargs)
        monthly.to_parquet(RESULTS_DIR / f"{tag}_monthly.parquet")
        trades_to_df(trades).to_parquet(RESULTS_DIR / f"{tag}_trades.parquet")

        m = compute_metrics(monthly["monthly_return"].astype(float))
        fc = sum(t.exit_reason == "force_close" for t in trades)
        fc_frac = fc / len(trades) if trades else float("nan")
        ref = INSAMPLE_F.get(metric, float("nan"))
        print(f"  wall {(time.time()-t0)/60:.1f} min | {len(trades):,} trades | "
              f"force-close {fc_frac:.1%}")
        print(f"  Sharpe {m.sharpe:.3f}   (frictionless no-carry baseline F = {ref})   "
              f"ann {m.ann_return:+.2%}  vol {m.ann_vol:.2%}  MDD {m.max_drawdown:.1%}")
        rows.append({"cell": cell, "metric": metric, "sharpe": round(m.sharpe, 3),
                     "baseline_F": ref, "ann_ret": round(m.ann_return, 4),
                     "ann_vol": round(m.ann_vol, 4), "mdd": round(m.max_drawdown, 4),
                     "n_trades": len(trades), "force_close_frac": round(fc_frac, 3)})

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "carryover_grid_summary.csv", index=False)
    print("\n" + "=" * 78)
    print(summary.to_string(index=False))
    print(f"\nsaved → {RESULTS_DIR/'carryover_grid_summary.csv'}")


if __name__ == "__main__":
    main()
