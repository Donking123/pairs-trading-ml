"""
Phase 4a — net-of-cost Sharpe optimisation sweep.

Starting from the realism baseline (PC 0.572, factor 0.578), test PRINCIPLED cost-reduction
levers — trade less but better, pay less per trade — and rank them by net Sharpe. All use
the actual cost model (CRSP bid/ask + 35bps borrow); levers are turned via existing knobs:

  baseline   : core + 3.5σ stop                       (reference, = phase4a result)
  coint      : + Engle-Granger cointegration filter   (cut turnover ~3-4×)
  entry2.5   : entry |z| > 2.5                         (only stronger dislocations)
  entry3.0   : entry |z| > 3.0
  nostop     : drop the 3.5σ stop                      (stop adds cost-churn?)
  passive    : spread_cost_multiplier = 0.5            (limit orders vs marketable)
  combo      : coint + entry2.5 + passive              (stack the winners)

NOTE: this is exploratory — net Sharpe optimised on the full sample is in-sample. Validate
any winner out-of-sample (the lookahead/holdout machinery is in place) before claiming it.

Output → phases/phase4/results/cost_optimization_summary.csv

Background run:
  nohup python phases/phase4/notebooks/03_cost_optimization.py \\
    > phases/phase4/results/cost_opt.log 2>&1 &
  tail -f phases/phase4/results/cost_opt.log
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

from src.backtest import load_delisting, run_backtest
from src.config import PHASE4_DIR
from src.costs import RealismConfig
from src.panel import load_crsp_daily, load_market_returns, load_sp500_constituents
from src.performance import compute_metrics

RESULTS_DIR = PHASE4_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

FULL = RealismConfig(transaction_costs=True, borrow_bps_annual=35.0, default_spread_bps=10.0)
PASSIVE = RealismConfig(transaction_costs=True, borrow_bps_annual=35.0,
                        default_spread_bps=10.0, spread_cost_multiplier=0.5)

# name -> run_backtest kwargs (metric injected per loop)
LEVERS = {
    "baseline":  dict(stop_sigma=3.5, realism=FULL),
    "coint":     dict(stop_sigma=3.5, realism=FULL, cointegration_filter=True),
    "entry2.5":  dict(stop_sigma=3.5, realism=FULL, entry_sigma=2.5),
    "entry3.0":  dict(stop_sigma=3.5, realism=FULL, entry_sigma=3.0),
    "nostop":    dict(stop_sigma=None, realism=FULL),
    "passive":   dict(stop_sigma=3.5, realism=PASSIVE),
    "combo":     dict(stop_sigma=3.5, realism=PASSIVE, cointegration_filter=True,
                      entry_sigma=2.5),
}
METRICS = ["pc", "factor"]


def main() -> None:
    crsp = load_crsp_daily(); cons = load_sp500_constituents()
    dlst = load_delisting(); mkt = load_market_returns()

    rows = []
    for metric in METRICS:
        for name, kwargs in LEVERS.items():
            t0 = time.time()
            monthly, trades = run_backtest(
                start="2003-01-01", end="2023-12-31", crsp=crsp, constituents=cons,
                delisting_df=dlst, market_returns=mkt, metric=metric, verbose=False, **kwargs)
            m = compute_metrics(monthly["monthly_return"].astype(float))
            rows.append({"metric": metric, "lever": name, "net_sharpe": round(m.sharpe, 3),
                         "ann_ret": round(m.ann_return, 4), "ann_vol": round(m.ann_vol, 4),
                         "mdd": round(m.max_drawdown, 4), "n_trades": len(trades)})
            print(f"  {metric:6s} {name:9s}: net Sharpe {m.sharpe:.3f}  "
                  f"({len(trades):,} trades, {(time.time()-t0)/60:.1f} min)")

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "cost_optimization_summary.csv", index=False)
    print("\n" + "=" * 70)
    for metric in METRICS:
        sub = summary[summary.metric == metric].sort_values("net_sharpe", ascending=False)
        print(f"\n### {metric} — ranked by net Sharpe")
        print(sub[["lever", "net_sharpe", "ann_ret", "ann_vol", "n_trades"]].to_string(index=False))


if __name__ == "__main__":
    main()
