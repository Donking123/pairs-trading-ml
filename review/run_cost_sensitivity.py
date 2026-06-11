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

    # No flags needed: auto-discovers the latest trades file.
    python review/run_cost_sensitivity.py

    # Or point at a specific run:
    python review/run_cost_sensitivity.py \
        --trades  datastream/data/walkforward_output/walkforward_XXXX/trades_oos.csv \
        --out-dir datastream/data/walkforward_output/walkforward_XXXX
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("cost_sensitivity")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASTREAM = _REPO_ROOT / "datastream"
# Output roots that may contain walkforward_* run folders with a trades file.
_OUTPUT_ROOTS = [
    _DATASTREAM / "data" / "walkforward_output",
    _REPO_ROOT / "review" / "output",
    _REPO_ROOT / "research" / "output",
]
# Trades filenames in order of preference.
_TRADES_NAMES = ["trades_oos.csv", "trades.csv", "trades.parquet"]


def _find_latest_trades() -> Optional[Path]:
    """Auto-discover the most recently modified trades file under the known
    output roots so ``--trades`` can be omitted. Returns None if none found."""
    candidates: list[Path] = []
    for root in _OUTPUT_ROOTS:
        if not root.exists():
            continue
        for name in _TRADES_NAMES:
            candidates.extend(root.rglob(name))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# -----------------------------------------------------------------------------
# Market-impact model (square-root impact, Almgren et al. 2005)
# -----------------------------------------------------------------------------
def market_impact_sweep(
    closed: pd.DataFrame,
    notional_usd: float,
    adv_levels: list[float],
    borrow_rates: list[float],
    eta: float = 0.1,
) -> list[dict]:
    """Square-root market-impact model: cost per leg = eta * sqrt(notional/ADV).
    Four applications per round-trip (entry + exit of both legs). Sweeps ADV
    levels to find the break-even liquidity threshold."""
    duration = closed["duration_days"].to_numpy(dtype=float)
    roll_cost = (closed["roll_cost_pct"].to_numpy(dtype=float)
                 if "roll_cost_pct" in closed.columns else np.zeros(len(closed)))
    ruce_gross = closed["ruce"].to_numpy(dtype=float) + roll_cost
    results = []
    for adv in adv_levels:
        impact_per_leg = eta * float(np.sqrt(notional_usd / adv))
        total_impact = 4.0 * impact_per_leg
        for borrow in borrow_rates:
            borrow_cost = borrow * (duration / 365.0)
            ruce_net = ruce_gross - roll_cost - borrow_cost - total_impact
            median_net = float(np.median(ruce_net))
            results.append({
                "adv_usd": int(adv),
                "borrow_rate_pct": round(borrow * 100, 2),
                "impact_per_leg_pct": round(impact_per_leg * 100, 4),
                "total_impact_pct": round(total_impact * 100, 4),
                "median_ruce_net": round(median_net, 6),
                "pct_profitable": round(float((ruce_net > 0).mean()), 4),
                "breakeven": median_net <= 0,
            })
    return results


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cost sensitivity sweep: borrow rate × slippage on OOS trades.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trades", type=Path, default=None,
                   help="trades_oos.csv from run_walkforward.py "
                        "(auto-discovers latest under datastream/data/walkforward_output if omitted)")
    p.add_argument("--borrow-rates", type=str, default="0.0,0.005,0.01,0.02",
                   help="comma-separated annualised borrow rates to sweep")
    p.add_argument("--slippage-bps", type=str, default="0,5,10,20",
                   help="comma-separated one-way slippage in basis points per leg")
    p.add_argument("--notional-usd", type=float, default=100_000,
                   help="USD notional per ADR leg for market-impact model")
    p.add_argument("--adv-levels-usd", type=str,
                   default="500000,1000000,5000000,10000000",
                   help="comma-separated assumed ADV in USD for market-impact sweep")
    p.add_argument("--impact-eta", type=float, default=0.1,
                   help="market-impact coefficient eta in sqrt(notional/ADV) model")
    p.add_argument("--out-dir", type=Path,
                   default=_REPO_ROOT / "review" / "output"
                   / f"cost_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # --- resolve trades path (auto-discover latest if not supplied) -----------
    if args.trades is None:
        args.trades = _find_latest_trades()
        if args.trades is None:
            log.error("no trades file found under %s — pass --trades explicitly",
                      ", ".join(str(r) for r in _OUTPUT_ROOTS))
            return 2
        log.info("auto-discovered trades file: %s", args.trades)
    if not args.trades.exists():
        log.error("trades file not found: %s", args.trades)
        return 2

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
    out_dir = args.out_dir
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

    # --- market impact model --------------------------------------------------
    adv_levels = [float(x) for x in args.adv_levels_usd.split(",")]
    log.info("running market-impact sweep (eta=%.2f, notional=$%s) ...",
             args.impact_eta, f"{int(args.notional_usd):,}")
    impact_results = market_impact_sweep(
        closed, args.notional_usd, adv_levels, borrow_rates, args.impact_eta
    )
    df_impact = pd.DataFrame(impact_results)
    df_impact.to_csv(out_dir / "market_impact.csv", index=False)
    (out_dir / "market_impact.json").write_text(json.dumps(impact_results, indent=2))
    pivot_impact = df_impact.pivot_table(
        index="adv_usd", columns="borrow_rate_pct", values="median_ruce_net"
    )
    log.info("\nMedian RUCE-net by ADV (USD) × borrow rate (%%):\n%s", pivot_impact.to_string())
    breakeven_i = df_impact[df_impact["breakeven"]]
    if breakeven_i.empty:
        log.info("Strategy remains positive across the entire ADV/borrow impact grid.")
    else:
        first = breakeven_i.iloc[0]
        log.info("Strategy breaks even at: ADV=$%s, borrow=%.1f%%, impact/leg=%.2f%%",
                 f"{int(first['adv_usd']):,}", first["borrow_rate_pct"],
                 first["impact_per_leg_pct"])
    log.info("wrote market_impact.csv + .json to %s", out_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
