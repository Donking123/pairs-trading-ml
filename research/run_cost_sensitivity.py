#!/usr/bin/env python3
"""
run_cost_sensitivity.py
=======================

Sweeps borrow rate × slippage assumptions on an existing OOS trades CSV and
reports how median RUCE-net changes — and the breakeven cost level at which it
hits zero. Consumes trades_oos.csv without re-running the backtest.

The trades file must have columns: ruce, roll_cost_pct, duration_days, was_aborted.

NOTE on baseline: run_backtest bakes roll cost into the effective fill prices,
so the ``ruce`` column already has roll subtracted (but NOT borrow/slippage).
To sweep costs cleanly from a single gross baseline we first ADD roll back out
(``ruce_gross = ruce + roll_cost_pct``) and then re-apply roll + borrow + slippage
per cell. This avoids double-counting roll, which would otherwise be subtracted
once inside ``ruce`` and again here.

Usage
-----
::

    python research/run_cost_sensitivity.py \
        --trades  research/output/walkforward/trades_oos.csv \
        --out-dir research/output/walkforward
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("cost_sensitivity")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cost sensitivity sweep: borrow rate × slippage on OOS trades.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trades", type=Path, required=True,
                   help="trades_oos.csv from run_walkforward.py")
    p.add_argument("--borrow-rates", type=str, default="0.0,0.005,0.01,0.02",
                   help="comma-separated annualised borrow rates to sweep")
    p.add_argument("--slippage-bps", type=str, default="0,5,10,20",
                   help="comma-separated one-way slippage in basis points per leg")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    trades = pd.read_csv(args.trades)
    closed = trades[~trades["was_aborted"].astype(bool)].copy()
    log.info("loaded %d closed trades from %s", len(closed), args.trades)

    borrow_rates  = [float(x) for x in args.borrow_rates.split(",")]
    slippage_bps  = [float(x) for x in args.slippage_bps.split(",")]

    if "ruce" not in closed.columns:
        log.error("trades file missing 'ruce' column")
        return 2

    duration = closed["duration_days"].to_numpy(dtype=float)

    # roll_cost_pct already estimated per pair (Roll (1984) bid-ask per pair).
    roll_cost = closed["roll_cost_pct"].to_numpy(dtype=float) if "roll_cost_pct" in closed.columns \
                else np.zeros(len(closed))

    # Recover a truly gross baseline: run_backtest already subtracts roll via the
    # effective fill prices that produced `ruce`, so add it back out before the
    # sweep re-applies it. Without this, roll would be double-counted.
    ruce_gross = closed["ruce"].to_numpy(dtype=float) + roll_cost

    results = []
    for borrow in borrow_rates:
        for slip_bps in slippage_bps:
            slip_frac = slip_bps / 10_000.0
            # slippage: 2 legs × 2 turns (entry + exit) × slip per leg
            total_slip = 4.0 * slip_frac
            borrow_cost = borrow * (duration / 365.0)
            ruce_net = ruce_gross - roll_cost - borrow_cost - total_slip
            median_net = float(np.median(ruce_net))
            pct_positive = float((ruce_net > 0).mean())
            results.append({
                "borrow_rate_pct": round(borrow * 100, 2),
                "slippage_bps_per_leg": slip_bps,
                "total_extra_cost_bps": round((borrow * duration.mean() / 365.0 + total_slip) * 10_000, 1),
                "median_ruce_net": round(median_net, 6),
                "pct_profitable": round(pct_positive, 4),
                "breakeven": median_net <= 0,
            })
            log.info("  borrow=%.1f%% slip=%dbps -> median RUCE-net=%.4f  pct_profit=%.1f%%  %s",
                     borrow * 100, slip_bps, median_net, pct_positive * 100,
                     "BELOW ZERO" if median_net <= 0 else "positive")

    df_out = pd.DataFrame(results)
    out_dir = args.out_dir or Path(args.trades).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_dir / "cost_sensitivity.csv", index=False)
    (out_dir / "cost_sensitivity.json").write_text(json.dumps(results, indent=2))

    # Print pivot table: rows=borrow, cols=slippage
    pivot = df_out.pivot_table(
        index="borrow_rate_pct",
        columns="slippage_bps_per_leg",
        values="median_ruce_net",
    )
    log.info("\nMedian RUCE-net by borrow rate (%%) × slippage (bps per leg):\n%s", pivot.to_string())

    breakeven = df_out[df_out["breakeven"]]
    if breakeven.empty:
        log.info("Strategy remains positive across the entire cost grid.")
    else:
        log.info("Strategy goes negative at: borrow=%.1f%%, slip=%dbps",
                 breakeven.iloc[0]["borrow_rate_pct"],
                 breakeven.iloc[0]["slippage_bps_per_leg"])

    log.info("wrote cost_sensitivity.csv + .json to %s", out_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
