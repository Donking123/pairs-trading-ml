#!/usr/bin/env python3
"""
run_param_grid.py
=================

Parameter-robustness study for the Hong & Susmel (2013) Asian ADR pairs
strategy. Sweeps the (k0, kc, T, H) grid the paper tests and reports the
out-of-sample median RUCE-net (and other stats) per grid cell, so the result
can be shown not to depend on a single lucky parameter pick.

    k0 (entry multiplier)   default grid: 1.65, 2.0, 2.33
    kc (exit  multiplier)   default grid: 0.0, 0.5, 1.0
    T  (estimation window)  default grid: 30, 60, 90, 120
    H  (max holding days)   default grid: 30, 90, 120

By default the sweep is **out-of-sample**: pairs are re-selected on the train
window for each (T, H) combination (selection depends on T/H but not on the
entry/exit thresholds), then traded on the test window. Pass ``--in-sample`` to
sweep over the whole history using a fixed pre-selected pair registry instead
(faster, but optimistic — use only as a sanity check against the paper).

Output is a tidy ``grid_results.csv`` (one row per cell) plus pivoted heatmap
CSVs (median RUCE-net as a function of each parameter pair).

Usage
-----
Out-of-sample (recommended)::

    python research/run_param_grid.py \
        --train-start 2015-01-01 --split 2021-12-31 --test-end 2024-12-31 \
        --out-dir research/output/grid_$(date +%Y%m%d_%H%M%S)

In-sample against a fixed registry::

    python research/run_param_grid.py --in-sample \
        --pairs datastream/config/pairs/asian_adr_pairs.json \
        --out-dir research/output/grid_insample
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("paramgrid")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASTREAM = _REPO_ROOT / "datastream"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_screen = _load_module("ds_screening", _DATASTREAM / "run_asian_adr_screening.py")
_bt = _load_module("ds_backtest", _DATASTREAM / "run_backtest.py")
_wf = _load_module("research_walkforward", _DATASTREAM / "run_walkforward.py")


# -----------------------------------------------------------------------------
# Per-cell evaluation
# -----------------------------------------------------------------------------
def _cell_stats(trades: list, k0: float, kc: float, T: int, H: int) -> dict:
    closed = [t for t in trades if not t.was_aborted]
    row = {"k0": k0, "kc": kc, "T": T, "H": H,
           "n_trades": len(trades), "n_closed": len(closed),
           "n_abort": len(trades) - len(closed)}
    if closed:
        ruce_net = np.array([t.ruce_net for t in closed], dtype=float)
        roce_net = np.array([t.roce_net for t in closed], dtype=float)
        dur = np.array([t.duration_days for t in closed], dtype=float)
        row.update({
            "median_ruce_net": round(float(np.median(ruce_net)), 6),
            "mean_ruce_net": round(float(np.mean(ruce_net)), 6),
            "median_roce_net": round(float(np.median(roce_net)), 6),
            "median_duration": round(float(np.median(dur)), 2),
            "pct_profitable": round(float((ruce_net > 0).mean()), 4),
        })
    else:
        row.update({"median_ruce_net": None, "mean_ruce_net": None,
                    "median_roce_net": None, "median_duration": None,
                    "pct_profitable": None})
    return row


def run_grid_oos(args, cfg_base: dict) -> list[dict]:
    """Out-of-sample sweep: re-select per T (screening depends on T, not H),
    then sweep (H, k0, kc) from the cached pairs+panel."""
    adr_prices    = _screen._load_or_die(args.adr_prices, "ADR prices")
    adr_reference = _screen._load_or_die(args.adr_reference, "ADR reference")
    global_prices = _screen._load_or_die(args.global_prices, "Global prices")
    fx_rates      = _screen._load_or_die(args.fx_rates, "FX rates")
    fx_rates = fx_rates.copy()
    fx_rates["date"] = pd.to_datetime(fx_rates["date"]).dt.normalize()

    train_adr = adr_prices[adr_prices["marketdate"] >= pd.Timestamp(args.train_start)]
    train_glb = global_prices[global_prices["marketdate"] >= pd.Timestamp(args.train_start)]
    test_start = args.split + timedelta(days=1)
    results: list[dict] = []

    # Screen once per T — H is a backtest parameter, not a screening parameter.
    # Cache (approved_pairs, panel) keyed by T.
    screen_cache: dict[int, tuple[list, dict]] = {}

    for T in args.T_grid:
        cfg = dict(cfg_base, estimation_days=T, holding_days=args.H_grid[0])
        log.info("screening for T=%d (as_of=%s) …", T, args.split)
        approved, _ = _screen.run_pipeline(
            train_adr, adr_reference, train_glb, fx_rates, args.split, cfg)
        log.info("T=%d: %d pairs approved", T, len(approved))
        if not approved:
            screen_cache[T] = ([], {})
            continue
        # Build Pair objects with placeholder H; we'll overwrite per cell.
        base_pairs = [
            _bt.Pair(
                pair_id=p.pair_id, adr_ticker=p.adr_ticker,
                underlying_ticker=p.underlying_ticker,
                underlying_exchange=p.underlying_exchange,
                underlying_currency=p.underlying_currency,
                adr_ratio=float(p.adr_ratio), estimation_days=T, holding_days=args.H_grid[0],
                roll_spread_adr=float(p.roll_spread_adr),
                roll_spread_local=float(p.roll_spread_local),
            )
            for p in approved
        ]
        panel = _wf._build_panel(adr_prices, global_prices, fx_rates, base_pairs)
        screen_cache[T] = (base_pairs, panel)

    n_th = len(args.T_grid) * len(args.H_grid)
    n_kk = len(args.k0_grid) * len(args.kc_grid)
    total_cells = n_th * n_kk
    cell_num = 0

    for th_idx, (T, H) in enumerate(itertools.product(args.T_grid, args.H_grid), 1):
        base_pairs, panel = screen_cache[T]
        log.info("── (T=%d H=%d) block %d/%d, %d pairs, sweeping %d (k0,kc) cells ──",
                 T, H, th_idx, n_th, len(base_pairs), n_kk)
        if not base_pairs:
            for k0, kc in itertools.product(args.k0_grid, args.kc_grid):
                results.append(_cell_stats([], k0, kc, T, H))
                cell_num += 1
            continue

        for k0, kc in itertools.product(args.k0_grid, args.kc_grid):
            # Update mutable fields per cell (Pair is a dataclass, fields are writable)
            for pair in base_pairs:
                pair.holding_days = H
                pair.k0 = k0
                pair.kc = kc
            bt_cfg = {"T": T, "H": H, "k0": k0, "kc": kc}
            trades: list = []
            for pair in base_pairs:
                df = panel.get(pair.pair_id)
                if df is None:
                    continue
                trades.extend(_bt.backtest_pair(pair, df, bt_cfg,
                                                start=test_start, end=args.test_end))
            stats = _cell_stats(trades, k0, kc, T, H)
            results.append(stats)
            cell_num += 1
            log.info("  [%d/%d] k0=%.2f kc=%.1f T=%d H=%d -> median RUCE-net=%s (%d trades)",
                     cell_num, total_cells, k0, kc, T, H, stats["median_ruce_net"], stats["n_trades"])
    return results


def run_grid_insample(args, _cfg_base: dict) -> list[dict]:
    """In-sample sweep over a fixed pre-selected registry and the full history."""
    pairs = _bt.load_pairs(args.pairs)
    if not pairs:
        log.error("no pairs in %s", args.pairs)
        return []
    panel = _bt.load_panel(args.adr_prices, args.global_prices, args.fx_rates, pairs)
    results: list[dict] = []
    for k0, kc, T, H in itertools.product(args.k0_grid, args.kc_grid,
                                          args.T_grid, args.H_grid):
        for pair in pairs:
            pair.k0, pair.kc, pair.estimation_days, pair.holding_days = k0, kc, T, H
        bt_cfg = {"T": T, "H": H, "k0": k0, "kc": kc}
        trades: list = []
        for pair in pairs:
            df = panel.get(pair.pair_id)
            if df is None:
                continue
            trades.extend(_bt.backtest_pair(pair, df, bt_cfg))
        stats = _cell_stats(trades, k0, kc, T, H)
        results.append(stats)
        log.info("cell k0=%.2f kc=%.1f T=%d H=%d -> median RUCE-net=%s (%d trades)",
                 k0, kc, T, H, stats["median_ruce_net"], stats["n_trades"])
    return results


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def write_outputs(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "grid_results.csv", index=False)
    (out_dir / "grid_results.json").write_text(json.dumps(results, indent=2, default=str))

    # heatmaps of median RUCE-net for each parameter pairing
    if not df.empty and df["median_ruce_net"].notna().any():
        for a, b in [("k0", "kc"), ("T", "H"), ("k0", "T")]:
            piv = df.pivot_table(index=a, columns=b, values="median_ruce_net",
                                 aggfunc="median")
            piv.to_csv(out_dir / f"heatmap_{a}_{b}.csv")
    log.info("wrote grid_results.csv (%d cells) + heatmaps to %s", len(df), out_dir)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",")]


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parameter-robustness grid sweep for the Asian ADR pairs strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adr-prices", type=Path,
                   default=_DATASTREAM / "data/parquet/adr/adr_prices.parquet")
    p.add_argument("--adr-reference", type=Path,
                   default=_DATASTREAM / "data/parquet/adr/adr_reference.parquet")
    p.add_argument("--global-prices", type=Path,
                   default=_DATASTREAM / "data/parquet/global/global_prices.parquet")
    p.add_argument("--fx-rates", type=Path,
                   default=_DATASTREAM / "data/parquet/fx/fx_rates.parquet")
    p.add_argument("--pairs", type=Path,
                   default=_REPO_ROOT / "config/pairs/asian_adr_pairs.json",
                   help="pre-selected registry (in-sample mode only)")
    p.add_argument("--out-dir", type=Path,
                   default=_REPO_ROOT / "review" / "output"
                   / f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    p.add_argument("--in-sample", action="store_true",
                   help="sweep over full history with fixed registry (optimistic)")
    p.add_argument("--train-start", type=_parse_date, default=None)
    p.add_argument("--split", type=_parse_date, default=None)
    p.add_argument("--test-end", type=_parse_date, default=None)

    p.add_argument("--k0-grid", dest="k0_grid", type=_floats, default=[1.65, 2.0, 2.33])
    p.add_argument("--kc-grid", dest="kc_grid", type=_floats, default=[0.0, 0.5, 1.0])
    p.add_argument("--T-grid", dest="T_grid", type=_ints, default=[30, 60, 90, 120])
    p.add_argument("--H-grid", dest="H_grid", type=_ints, default=[30, 90, 120])

    p.add_argument("--cointegration-alpha", type=float, default=0.05)
    p.add_argument("--min-history-days", type=int, default=504)
    p.add_argument("--min-non-zero-return-pct", type=float, default=0.50)
    p.add_argument("--max-zero-return-pct-adr", type=float, default=0.50)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg_base = {
        "k0": args.k0_grid[0], "kc": args.kc_grid[0],
        "cointegration_alpha": args.cointegration_alpha,
        "min_history_days": args.min_history_days,
        "min_non_zero_return_pct": args.min_non_zero_return_pct,
        "max_zero_return_pct_adr": args.max_zero_return_pct_adr,
    }
    n_cells = (len(args.k0_grid) * len(args.kc_grid)
               * len(args.T_grid) * len(args.H_grid))
    log.info("grid: %d cells (k0=%s kc=%s T=%s H=%s)", n_cells,
             args.k0_grid, args.kc_grid, args.T_grid, args.H_grid)

    if args.in_sample:
        results = run_grid_insample(args, cfg_base)
    else:
        if not (args.train_start and args.split and args.test_end):
            log.error("out-of-sample mode needs --train-start --split --test-end "
                      "(or pass --in-sample)")
            return 2
        results = run_grid_oos(args, cfg_base)

    write_outputs(results, args.out_dir)

    # print the best cells by median RUCE-net
    valid = [r for r in results if r.get("median_ruce_net") is not None]
    valid.sort(key=lambda r: r["median_ruce_net"], reverse=True)
    log.info("=" * 72)
    log.info("TOP 5 CELLS BY MEDIAN RUCE-NET (%s)",
             "in-sample" if args.in_sample else "out-of-sample")
    log.info("=" * 72)
    for r in valid[:5]:
        log.info("  k0=%.2f kc=%.1f T=%3d H=%3d | median RUCE-net=%.4f | %d trades",
                 r["k0"], r["kc"], r["T"], r["H"], r["median_ruce_net"], r["n_trades"])
    log.info("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
