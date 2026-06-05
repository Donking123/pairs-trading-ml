#!/usr/bin/env python3
"""
run_benchmark.py
================

Significance test for the Asian ADR pairs strategy: is the realised edge real,
or could random entries on the same pairs / dates have done as well?

Two complementary checks:

1. **Random-entry null.** For each pair, replace the spread-threshold entry rule
   with random entries that match the *strategy's own* trade frequency and
   holding behaviour, drawn from the same price panel. Repeat ``--n-sims`` times
   to build a null distribution of median RUCE-net. The strategy's empirical
   median is located in that distribution to give an empirical p-value.

2. **Bootstrap CI on the strategy itself.** Resample the strategy's realised
   per-trade RUCE-net with replacement to get a confidence interval on the
   median, and a bootstrap p-value for median > 0.

This consumes an existing trades file (from ``run_backtest.py`` or
``run_walkforward.py``) for the bootstrap, and the price panel + registry for
the random-entry null.

Usage
-----
::

    python research/run_benchmark.py \
        --trades  research/output/walkforward_XXXX/trades_oos.csv \
        --pairs   datastream/config/pairs/asian_adr_pairs.json \
        --n-sims  1000 \
        --out-dir research/output/walkforward_XXXX
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("benchmark")

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


_bt = _load_module("ds_backtest", _DATASTREAM / "run_backtest.py")


# -----------------------------------------------------------------------------
# Bootstrap on the realised strategy returns
# -----------------------------------------------------------------------------
def bootstrap_median(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    if len(values) == 0:
        return {"median": None, "ci_low": None, "ci_high": None, "p_value_gt_0": None}
    boot = np.empty(n_boot)
    n = len(values)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        boot[b] = np.median(sample)
    return {
        "median": round(float(np.median(values)), 6),
        "ci_low": round(float(np.quantile(boot, 0.025)), 6),
        "ci_high": round(float(np.quantile(boot, 0.975)), 6),
        # one-sided bootstrap p-value for H0: median <= 0
        "p_value_gt_0": round(float((boot <= 0).mean()), 4),
    }


# -----------------------------------------------------------------------------
# Random-entry null
# -----------------------------------------------------------------------------
def random_entry_trades(
    pair,
    df: pd.DataFrame,
    n_entries: int,
    holding_days: int,
    rng: np.random.Generator,
) -> list[float]:
    """
    Generate ``n_entries`` random round-trips on this pair's panel and return
    their RUCE-net values. Each entry: pick a random bar, SHORT ADR at its
    close, buy local next bar (no overnight-abort logic — we want a neutral
    null), then close after ``holding_days`` calendar days or at the panel end.
    Mirrors the return arithmetic in run_backtest.backtest_pair.
    """
    if len(df) < 5 or n_entries <= 0:
        return []
    adr_close = df["adr_close"].to_numpy()
    local_usd = df["local_close_usd"].to_numpy()
    dates = [pd.Timestamp(d).date() for d in df.index]
    roll_cost = float(pair.roll_spread_adr + pair.roll_spread_local)
    n = len(df)
    out: list[float] = []

    for _ in range(n_entries):
        i = int(rng.integers(0, n - 2))
        j = i + 1  # local leg next bar
        adr_open = adr_close[i]
        loc_open = local_usd[j]
        if adr_open <= 0 or loc_open <= 0:
            continue
        # close: first bar at least holding_days after entry, else last bar
        target = dates[i]
        close_idx = n - 1
        for k in range(j + 1, n):
            if (dates[k] - target).days >= holding_days:
                close_idx = k
                break
        adr_cover = adr_close[close_idx]
        loc_close = local_usd[close_idx]
        if adr_cover <= 0 or loc_close <= 0:
            continue
        local_ret = (loc_close - loc_open) / loc_open
        adr_ret = (adr_open - adr_cover) / adr_open
        ruce = local_ret + 2.0 * adr_ret
        out.append(ruce - roll_cost)
    return out


def random_entry_null(
    pairs: list,
    panel: dict,
    entries_per_pair: dict,
    holding_days: int,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an array of length n_sims: median RUCE-net of a random-entry book."""
    sims = np.full(n_sims, np.nan)
    for s in range(n_sims):
        all_r: list[float] = []
        for pair in pairs:
            df = panel.get(pair.pair_id)
            if df is None:
                continue
            k = entries_per_pair.get(pair.pair_id, 0)
            all_r.extend(random_entry_trades(pair, df, k, holding_days, rng))
        if all_r:
            sims[s] = float(np.median(all_r))
        if (s + 1) % max(1, n_sims // 10) == 0:
            log.info("  random-entry sim %d/%d", s + 1, n_sims)
    return sims[~np.isnan(sims)]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Random-entry null + bootstrap significance test for the strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trades", type=Path, required=True,
                   help="strategy trades_oos.csv / trades.csv / trades.parquet")
    p.add_argument("--pairs", type=Path,
                   default=_DATASTREAM / "config/pairs/asian_adr_pairs.json")
    p.add_argument("--adr-prices", type=Path,
                   default=_DATASTREAM / "data/parquet/adr/adr_prices.parquet")
    p.add_argument("--global-prices", type=Path,
                   default=_DATASTREAM / "data/parquet/global/global_prices.parquet")
    p.add_argument("--fx-rates", type=Path,
                   default=_DATASTREAM / "data/parquet/fx/fx_rates.parquet")
    p.add_argument("--metric", default="ruce_net",
                   choices=["roce", "ruce", "roce_net", "ruce_net"])
    p.add_argument("--holding-days", type=int, default=90)
    p.add_argument("--n-sims", type=int, default=1000, help="random-entry simulations")
    p.add_argument("--n-boot", type=int, default=10000, help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    # --- load strategy trades -------------------------------------------------
    if args.trades.suffix == ".parquet":
        trades = pd.read_parquet(args.trades)
    else:
        trades = pd.read_csv(args.trades)
    closed = trades[~trades.get("was_aborted", False).astype(bool)]
    strat_vals = closed[args.metric].to_numpy(dtype=float)
    strat_vals = strat_vals[~np.isnan(strat_vals)]
    strat_median = float(np.median(strat_vals)) if len(strat_vals) else float("nan")
    log.info("strategy: %d closed trades, median %s = %.4f",
             len(strat_vals), args.metric, strat_median)

    # --- bootstrap CI on the strategy ----------------------------------------
    boot = bootstrap_median(strat_vals, args.n_boot, rng)
    log.info("bootstrap median %s: %.4f  95%% CI [%.4f, %.4f]  p(median<=0)=%s",
             args.metric, boot["median"], boot["ci_low"], boot["ci_high"],
             boot["p_value_gt_0"])

    # --- random-entry null ----------------------------------------------------
    pairs = _bt.load_pairs(args.pairs)
    for p in pairs:
        p.holding_days = args.holding_days
    panel = _bt.load_panel(args.adr_prices, args.global_prices, args.fx_rates, pairs)
    # match the strategy's trade frequency per pair
    entries_per_pair = (
        closed.groupby("pair_id").size().to_dict() if "pair_id" in closed else {}
    )
    log.info("running %d random-entry simulations matched to strategy frequency",
             args.n_sims)
    null = random_entry_null(pairs, panel, entries_per_pair,
                             args.holding_days, args.n_sims, rng)

    if len(null):
        # empirical one-sided p-value: P(null median >= strategy median)
        p_emp = float((null >= strat_median).mean())
        null_summary = {
            "n_valid_sims": int(len(null)),
            "null_median_mean": round(float(np.mean(null)), 6),
            "null_median_p95": round(float(np.quantile(null, 0.95)), 6),
            "strategy_median": round(strat_median, 6),
            "empirical_p_value": round(p_emp, 4),
        }
    else:
        null_summary = {"n_valid_sims": 0}

    report = {
        "metric": args.metric,
        "n_strategy_trades": int(len(strat_vals)),
        "bootstrap": boot,
        "random_entry_null": null_summary,
    }

    out_dir = args.out_dir or args.trades.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "significance.json").write_text(json.dumps(report, indent=2, default=str))

    log.info("=" * 64)
    log.info("SIGNIFICANCE SUMMARY (%s)", args.metric)
    log.info("=" * 64)
    log.info("  strategy median            %.4f", strat_median)
    log.info("  bootstrap 95%% CI           [%.4f, %.4f]", boot["ci_low"], boot["ci_high"])
    log.info("  bootstrap p(median<=0)     %s", boot["p_value_gt_0"])
    if null_summary.get("n_valid_sims"):
        log.info("  random-entry null mean     %.4f", null_summary["null_median_mean"])
        log.info("  empirical p-value          %s", null_summary["empirical_p_value"])
    log.info("=" * 64)
    log.info("wrote significance.json to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
