"""
Phase 4b — lookahead-bias test on the headline strategies.

For each metric, run the backtest over the FULL window (2003→2023) and again over each
truncated window (2003→cut), then verify every overlapping daily target-position vector
is identical. Any mismatch = the future leaked into the past = lookahead bias.

Multiple cut dates are used on purpose — subtle leaks (delistings, dividends, universe
selection) only surface when a cut straddles the offending event (prof's advice).

Output → phases/phase4/results/lookahead_{metric}.txt + lookahead_summary.csv

Run (each full run ~1-2h; truncated runs are shorter):
  cd pairs-trading-ml
  nohup python phases/phase4/notebooks/01_lookahead_test.py \\
    > phases/phase4/results/lookahead_run.log 2>&1 &
  tail -f phases/phase4/results/lookahead_run.log
"""
from __future__ import annotations

import sys
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
from src.config import PHASE4_DIR
from src.lookahead import compare_position_panels, reconstruct_daily_positions
from src.panel import load_crsp_daily, load_market_returns, load_sp500_constituents

RESULTS_DIR = PHASE4_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

FULL_START = "2003-01-01"
FULL_END = "2023-12-31"
CUT_DATES = ["2017-12-31", "2013-12-31", "2009-12-31"]  # straddle GFC + later years
METRICS = ["pc", "factor"]


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([{
        "permno_a": t.permno_a, "permno_b": t.permno_b, "direction": t.direction,
        "entry_date": t.entry_date, "exit_date": t.exit_date,
    } for t in trades])


def main() -> None:
    crsp = load_crsp_daily(); cons = load_sp500_constituents()
    dlst = load_delisting(); mkt = load_market_returns()
    cal = pd.DatetimeIndex(crsp["date"].drop_duplicates().sort_values())
    cal = cal[(cal >= pd.Timestamp(FULL_START)) & (cal <= pd.Timestamp(FULL_END))]

    summary_rows = []
    for metric in METRICS:
        print("\n" + "=" * 78)
        print(f"METRIC: {metric}  —  full {FULL_START} → {FULL_END}")
        print("=" * 78)
        full_monthly, full_trades = run_backtest(
            start=FULL_START, end=FULL_END, crsp=crsp, constituents=cons,
            delisting_df=dlst, market_returns=mkt, metric=metric, verbose=False)
        full_pos = reconstruct_daily_positions(trades_to_df(full_trades), cal)
        print(f"  full run: {len(full_trades):,} trades, "
              f"{full_pos.shape[1]} pairs over {full_pos.shape[0]} days")

        lines = [f"Lookahead test — metric={metric}, full {FULL_START}..{FULL_END}"]
        for cut in CUT_DATES:
            _, short_trades = run_backtest(
                start=FULL_START, end=cut, crsp=crsp, constituents=cons,
                delisting_df=dlst, market_returns=mkt, metric=metric, verbose=False)
            short_pos = reconstruct_daily_positions(trades_to_df(short_trades), cal)
            res = compare_position_panels(full_pos, short_pos, cut_date=cut)
            print("  " + res.summary())
            lines.append(res.summary())
            if not res.passed:
                lines.append(res.mismatches.head(50).to_string())
            summary_rows.append({
                "metric": metric, "cut_date": cut, "passed": res.passed,
                "overlap_days": res.n_overlap_days, "pairs": res.n_pairs_compared,
                "mismatched_cells": res.n_mismatched_cells,
                "mismatched_days": res.n_mismatched_days,
            })
        (RESULTS_DIR / f"lookahead_{metric}.txt").write_text("\n".join(lines))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "lookahead_summary.csv", index=False)
    print("\n" + "=" * 78)
    print(summary.to_string(index=False))
    verdict = "ALL PASS ✅ — no lookahead bias detected" if summary["passed"].all() \
        else "FAIL ❌ — lookahead bias detected (see per-metric .txt)"
    print(verdict)


if __name__ == "__main__":
    main()
