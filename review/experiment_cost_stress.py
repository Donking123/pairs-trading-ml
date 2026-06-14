#!/usr/bin/env python3
"""
experiment_cost_stress.py
=========================

Diagnostic experiment: how robust is the walk-forward portfolio Sharpe to
**realistic transaction costs**?

Motivation
----------
The backtest already bakes in a Roll-estimated bid/ask spread (~2% round-trip
median) plus a flat 1%/yr borrow, but it assumes you fill **exactly** at the
close/open with no slippage or market impact, and that every small Asian ADR is
freely shortable. Those are the assumptions most likely to be flattering the
~5.5 Sharpe. This script charges an **incremental** cost on top of what is
already in ``roce_net`` / ``ruce_net`` and sweeps it.

Cost model
----------
``c`` = incremental cost in **bps per side, per leg** (slippage + impact + wider
real spread than the Roll estimate). A round-trip trade has 4 fills:

    short ADR (enter) + cover ADR (exit) + buy local (enter) + sell local (exit)

so the per-trade deduction is ``c * (2*w_adr + 2)`` where ``w_adr`` is 2 for the
``ruce`` metrics (ADR leg held at 2x under Reg-T) and 1 for ``roce``. An optional
annualised ``--extra-borrow-bps`` is added, prorated by each trade's duration.

Two cost regimes are reported:

    UNIFORM           : every trade pays the same ``c``.
    LIQUIDITY-SCALED  : ``c`` is scaled by the trade's Roll spread relative to the
                        median (``roll_cost_pct / median``), so illiquid pairs —
                        the ones where real costs bite — pay proportionally more.

Each trade's total return is reduced deterministically, so the ``adj``
reconciliation in ``daily_returns_mtm`` spreads the cost smoothly across the
holding days (costs add no volatility); annualised return falls and the Sharpe
follows. Marks use the synchronous (tradeable) local timing now built into
``run_portfolio.daily_returns_mtm``.

Changes no production code. Run::

    python review/experiment_cost_stress.py
    python review/experiment_cost_stress.py --metric ruce_net --costs-bps 0,25,50,100
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s")
log = logging.getLogger("cost_expt")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASTREAM = _REPO_ROOT / "datastream"


def _lm(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_pf = _lm("review_portfolio", _REPO_ROOT / "review" / "run_portfolio.py")
_wfp = _lm("ds_wf_portfolio", _REPO_ROOT / "review" / "run_walkforward_portfolio.py")


def _stressed_trades(trades: pd.DataFrame, metric: str, c_bps: float,
                     fills_factor: float, extra_borrow_bps: float,
                     liquidity_scaled: bool) -> pd.DataFrame:
    """Return a copy of ``trades`` with ``metric`` reduced by the incremental cost.

    ``c_bps`` is per side per leg; ``fills_factor`` converts it to a per-trade
    round-trip deduction. ``liquidity_scaled`` multiplies the cost by each trade's
    Roll spread relative to the median (illiquid pairs pay more)."""
    out = trades.copy()
    c = c_bps / 1e4
    scale = 1.0
    if liquidity_scaled and "roll_cost_pct" in out.columns:
        med = out["roll_cost_pct"].replace(0, np.nan).median()
        if med and med > 0:
            scale = (out["roll_cost_pct"] / med).clip(lower=0.1, upper=10.0).fillna(1.0)
    cost = c * fills_factor * scale
    borrow = (extra_borrow_bps / 1e4) * out["duration_days"].astype(float) / 365.0
    out[metric] = out[metric] - cost - borrow
    return out


def _sweep(trades, panel, metric, fills_factor, c_grid, extra_borrow_bps,
           liquidity_scaled, weight, max_concurrent, rf):
    rows = []
    for c in c_grid:
        st = _pf.performance_stats(
            _pf.daily_returns_mtm(
                _stressed_trades(trades, metric, c, fills_factor,
                                 extra_borrow_bps, liquidity_scaled),
                panel, metric, weight, max_concurrent),
            rf=rf)
        rows.append((c, st))
    return rows


def _breakeven(trades, panel, metric, fills_factor, extra_borrow_bps,
               liquidity_scaled, weight, max_concurrent, rf, sharpe_target):
    """Fine grid scan for the per-side cost where ann_return hits 0 and where
    Sharpe drops to ``sharpe_target``."""
    be_ret = be_sharpe = None
    for c in np.arange(0.0, 400.0, 5.0):
        st = _pf.performance_stats(
            _pf.daily_returns_mtm(
                _stressed_trades(trades, metric, c, fills_factor,
                                 extra_borrow_bps, liquidity_scaled),
                panel, metric, weight, max_concurrent),
            rf=rf)
        if be_sharpe is None and st["sharpe"] <= sharpe_target:
            be_sharpe = c
        if be_ret is None and st["annualised_return"] <= 0:
            be_ret = c
        if be_ret is not None and be_sharpe is not None:
            break
    return be_sharpe, be_ret


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wf-dir", type=Path, default=None,
                   help="walk-forward run dir (default: latest)")
    p.add_argument("--metric", default="roce_net",
                   choices=["roce", "ruce", "roce_net", "ruce_net"])
    p.add_argument("--costs-bps", default="0,10,25,50,75,100",
                   help="comma-separated incremental costs (bps per side per leg)")
    p.add_argument("--extra-borrow-bps", type=float, default=0.0,
                   help="annualised extra borrow (bps), prorated by trade duration")
    p.add_argument("--sharpe-target", type=float, default=2.5,
                   help="report the per-side cost that pushes Sharpe down to this")
    p.add_argument("--weight", default="equal", choices=["equal", "capital"])
    p.add_argument("--max-concurrent", type=int, default=20)
    p.add_argument("--rf", type=float, default=0.0)
    p.add_argument("--adr-reference", type=Path,
                   default=_DATASTREAM / "data/parquet/adr/adr_reference.parquet")
    p.add_argument("--adr-prices", type=Path,
                   default=_DATASTREAM / "data/parquet/adr/adr_prices.parquet")
    p.add_argument("--global-prices", type=Path,
                   default=_DATASTREAM / "data/parquet/global/global_prices.parquet")
    p.add_argument("--fx-rates", type=Path,
                   default=_DATASTREAM / "data/parquet/fx/fx_rates.parquet")
    p.add_argument("--pairs", type=Path,
                   default=_REPO_ROOT / "config/pairs/asian_adr_pairs.json")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    wf_dir = args.wf_dir or _wfp.find_latest_wf_dir()
    if wf_dir is None:
        log.error("no walk-forward run found; pass --wf-dir")
        return 2
    log.info("walk-forward run dir: %s", wf_dir)
    trades = _pf.load_trades(wf_dir / "trades_oos.csv")
    log.info("loaded %d trades", len(trades))

    panel = _wfp.build_panel_from_trades(
        trades, args.adr_reference, args.adr_prices,
        args.global_prices, args.fx_rates, registry=args.pairs)

    adr_w = 2.0 if args.metric.startswith("ruce") else 1.0
    fills_factor = 2.0 * adr_w + 2.0          # 4 for roce, 6 for ruce
    c_grid = [float(x) for x in args.costs_bps.split(",")]

    log.info("=" * 86)
    log.info("COST-STRESS EXPERIMENT  (metric=%s, %d fills/trade -> deduction = %.0f x cost/side"
             "%s)", args.metric, int(fills_factor), fills_factor,
             f", +{args.extra_borrow_bps:.0f}bps/yr borrow" if args.extra_borrow_bps else "")
    log.info("=" * 86)
    hdr = (f"{'cost/side(bps)':>15}{'ann_ret':>10}{'ann_vol':>10}{'Sharpe':>9}"
           f"{'maxDD':>9}{'hit':>7}     {'ann_ret':>10}{'Sharpe':>9}")
    log.info("%-61s%s", "                                       UNIFORM",
             "LIQUIDITY-SCALED")
    log.info(hdr)
    log.info("-" * 86)
    for c in c_grid:
        u = _pf.performance_stats(_pf.daily_returns_mtm(
            _stressed_trades(trades, args.metric, c, fills_factor,
                             args.extra_borrow_bps, False),
            panel, args.metric, args.weight, args.max_concurrent), rf=args.rf)
        s = _pf.performance_stats(_pf.daily_returns_mtm(
            _stressed_trades(trades, args.metric, c, fills_factor,
                             args.extra_borrow_bps, True),
            panel, args.metric, args.weight, args.max_concurrent), rf=args.rf)
        log.info("%15.0f%10.3f%10.3f%9.2f%9.3f%7.2f     %10.3f%9.2f",
                 c, u["annualised_return"], u["annualised_vol"], u["sharpe"],
                 u["max_drawdown"], u["hit_rate"], s["annualised_return"], s["sharpe"])
    log.info("=" * 86)

    for label, scaled in [("UNIFORM", False), ("LIQUIDITY-SCALED", True)]:
        be_sh, be_ret = _breakeven(
            trades, panel, args.metric, fills_factor, args.extra_borrow_bps,
            scaled, args.weight, args.max_concurrent, args.rf, args.sharpe_target)
        log.info("[%s] cost/side to push Sharpe to %.1f: %s bps   |  to zero annual return: %s bps",
                 label, args.sharpe_target,
                 f"{be_sh:.0f}" if be_sh is not None else ">395",
                 f"{be_ret:.0f}" if be_ret is not None else ">395")
    log.info("Note: this is ON TOP of the ~%s already modelled (Roll spread + 1%%/yr borrow). "
             "Real costs are heterogeneous and concentrate in illiquid names — the "
             "liquidity-scaled column is the more realistic read.",
             "2%-round-trip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
