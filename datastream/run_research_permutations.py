#!/usr/bin/env python3
"""
run_research_permutations.py
============================

Rolling train/test research runner for the Hong & Susmel (2013) Asian ADR
pairs strategy.  Sweeps multiple train/test windows and parameter combinations,
saves every permutation's result, then ranks by out-of-sample performance.

Strategy
--------
For each (train_window, test_window, T, k0, kc, H) combination:
  1. Use pre-screened pairs from the registry; filter to those with enough
     pre-test warm-up history (>= T + 30 observations before test_start).
  2. Backtest those pairs on the TEST window only.  Rolling stats warm up on
     all pre-test data so no look-ahead bias is introduced.
  3. Record ROCE/RUCE/duration distributions + metadata per run.

Note on pair selection
----------------------
Pairs are sourced from the pre-built registry (config/pairs/asian_adr_pairs.json),
which was screened on the full 2011-2026 history.  Pair SELECTION is therefore
in-sample, but parameter EVALUATION is properly out-of-sample for every test
window.  For fully OOS pair selection use datastream/run_walkforward.py instead
(slower: re-screens on each train window).

Usage
-----
::

    # default (5-year train, 1/2/4-year test, standard grid):
    .venv\\Scripts\\python.exe datastream/run_research_permutations.py

    # custom windows:
    .venv\\Scripts\\python.exe datastream/run_research_permutations.py \\
        --train-years 3 5 7 --test-years 1 2 4

    # full parameter grid (~3x more combinations, ~3x slower):
    .venv\\Scripts\\python.exe datastream/run_research_permutations.py --full-grid
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("research")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASTREAM = _REPO_ROOT / "datastream"

# Default data paths (actual location in repo)
_DEFAULT_ADR     = _REPO_ROOT / "data/parquet/adr_prices.parquet"
_DEFAULT_GLOBAL  = _REPO_ROOT / "data/parquet/global_prices.parquet"
_DEFAULT_FX      = _REPO_ROOT / "data/parquet/fx_rates.parquet"
_DEFAULT_PAIRS   = _REPO_ROOT / "config/pairs/asian_adr_pairs.json"
_DEFAULT_OUT     = _REPO_ROOT / "data/backtest"


# ---------------------------------------------------------------------------
# Import backtest logic from the standalone datastream script
# ---------------------------------------------------------------------------
def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_bt = _load_module("ds_backtest", _DATASTREAM / "run_backtest.py")


# ---------------------------------------------------------------------------
# Rolling window generation
# ---------------------------------------------------------------------------
DATA_START = date(2011, 5, 4)
DATA_END   = date(2026, 4, 30)


def generate_rolling_windows(
    train_years_list: list[int],
    test_years_list:  list[int],
    step_years:       int  = 1,
    data_start:       date = DATA_START,
    data_end:         date = DATA_END,
    min_test_obs:     int  = 60,   # minimum calendar days in test window
) -> list[dict]:
    """
    Return all valid (train_start, train_end, test_start, test_end) windows.

    For each (train_years, test_years) pair the earliest possible train_end
    year is data_start.year + train_years; windows are rolled forward by
    step_years until insufficient test data remains.
    """
    windows: list[dict] = []
    for train_years in train_years_list:
        for test_years in test_years_list:
            first_train_end_year = data_start.year + train_years
            year = first_train_end_year

            while True:
                train_start = date(year - train_years, 1, 1)
                if train_start < data_start:
                    train_start = data_start

                train_end  = date(year,              12, 31)
                test_start = date(year + 1,           1,  1)
                test_end   = date(year + test_years, 12, 31)

                if test_start >= data_end:
                    break

                # Cap test_end at data_end
                if test_end > data_end:
                    test_end = data_end

                # Require meaningful test window
                if (test_end - test_start).days < min_test_obs:
                    break

                windows.append({
                    "train_start": train_start,
                    "train_end":   train_end,
                    "test_start":  test_start,
                    "test_end":    test_end,
                    "train_years": train_years,
                    "test_years":  test_years,
                })
                year += step_years

    return windows


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
def build_param_grid(full: bool = False) -> list[dict]:
    """
    ``full=False``  → standard grid (paper's core range; fast).
    ``full=True``   → extended grid covering the paper's full (k0,kc,T,H) space.
    """
    if full:
        T_vals  = [30, 60, 90, 120]
        k0_vals = [1.65, 2.0, 2.5, 3.0]
        kc_vals = [0.0, 0.5, 1.0]
        H_vals  = [30, 60, 90, 120]
    else:
        T_vals  = [30, 60, 90]
        k0_vals = [1.65, 2.0, 2.5]
        kc_vals = [0.0, 0.5]
        H_vals  = [30, 90]

    return [
        {"T": T, "k0": k0, "kc": kc, "H": H}
        for T, k0, kc, H in itertools.product(T_vals, k0_vals, kc_vals, H_vals)
    ]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
PCTILE_LABELS = ["mean", "std", "max", "p90", "p75", "median", "p25", "p10", "min"]


def _dist_stats(values: np.ndarray, prefix: str) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {f"{prefix}_{k}": float("nan") for k in PCTILE_LABELS}
    return {
        f"{prefix}_mean":   float(np.mean(arr)),
        f"{prefix}_std":    float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        f"{prefix}_max":    float(np.max(arr)),
        f"{prefix}_p90":    float(np.quantile(arr, 0.90)),
        f"{prefix}_p75":    float(np.quantile(arr, 0.75)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p25":    float(np.quantile(arr, 0.25)),
        f"{prefix}_p10":    float(np.quantile(arr, 0.10)),
        f"{prefix}_min":    float(np.min(arr)),
    }


def _run_stats(trades: list, eligible_pairs: list, meta: dict) -> dict:
    closed  = [t for t in trades if not t.was_aborted]
    n_total = len(trades)
    n_closed = len(closed)
    n_abort  = sum(1 for t in trades if t.was_aborted)
    n_force  = sum(1 for t in closed
                   if t.close_reason.value == "force_close")

    pair_ids = [t.pair_id for t in trades]
    n_active = len(set(pair_ids))
    max_pair_pct = (max(Counter(pair_ids).values()) / n_total
                    if n_total else float("nan"))

    test_start = meta["test_start"]
    test_end   = meta["test_end"]
    actual_yrs = max((test_end - test_start).days / 365.25, 0.25)
    trades_per_year = n_closed / actual_yrs

    roce     = np.array([t.roce      for t in closed])
    ruce     = np.array([t.ruce      for t in closed])
    roce_net = np.array([t.roce_net  for t in closed])
    ruce_net = np.array([t.ruce_net  for t in closed])
    dur      = np.array([t.duration_days for t in closed])

    win_rate = float(np.mean(ruce_net > 0)) if len(ruce_net) > 0 else float("nan")

    rec: dict = {
        "run_id":         meta["run_id"],
        "train_start":    meta["train_start"].isoformat(),
        "train_end":      meta["train_end"].isoformat(),
        "test_start":     meta["test_start"].isoformat(),
        "test_end":       meta["test_end"].isoformat(),
        "train_years":    meta["train_years"],
        "test_years":     meta["test_years"],
        "estimation_days": meta["T"],
        "k0":             meta["k0"],
        "kc":             meta["kc"],
        "holding_days":   meta["H"],
        "n_eligible_pairs": len(eligible_pairs),
        "n_active_pairs": n_active,
        "n_trades":       n_total,
        "n_closed":       n_closed,
        "n_force_closes": n_force,
        "n_overnight_aborts": n_abort,
        "win_rate":       round(win_rate, 4) if np.isfinite(win_rate) else None,
        "trades_per_year": round(trades_per_year, 2),
        "max_pair_pct":   round(max_pair_pct, 4) if np.isfinite(max_pair_pct) else None,
    }
    rec.update(_dist_stats(roce,     "roce"))
    rec.update(_dist_stats(ruce,     "ruce"))
    rec.update(_dist_stats(roce_net, "roce_net"))
    rec.update(_dist_stats(ruce_net, "ruce_net"))
    rec.update(_dist_stats(dur,      "duration"))
    return rec


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def compute_score(rec: dict) -> float:
    """
    Composite OOS score.  Higher is better.
    Returns -inf for runs that fail minimum validity filters.

    score = median_ruce_net
          + 0.5 * mean_ruce_net
          - 0.25 * std_ruce_net
          + trade_count_bonus    (capped at 0.01 for 100+ closed trades)
          - concentration_penalty (0.02 if any single pair > 50% of trades)
    """
    n_closed    = rec.get("n_closed", 0)
    n_active    = rec.get("n_active_pairs", 0)
    med         = rec.get("ruce_net_median", float("nan"))
    mn          = rec.get("ruce_net_mean",   float("nan"))
    sd          = rec.get("ruce_net_std",    float("nan"))
    max_pct     = rec.get("max_pair_pct")

    if n_closed < 10:               return float("-inf")
    if n_active < 3:                return float("-inf")
    if not np.isfinite(med):        return float("-inf")
    if med <= 0:                    return float("-inf")

    bonus = min(n_closed, 100) / 100 * 0.01
    pen   = 0.02 if (max_pct is not None and max_pct > 0.50) else 0.0
    score = (
        med
        + 0.5  * (mn if np.isfinite(mn) else 0.0)
        - 0.25 * (sd if np.isfinite(sd) else 0.0)
        + bonus
        - pen
    )
    return float(score)


# ---------------------------------------------------------------------------
# Main research loop
# ---------------------------------------------------------------------------
def run_research(
    adr_prices_path:    Path,
    global_prices_path: Path,
    fx_rates_path:      Path,
    pairs:              list,
    windows:            list[dict],
    param_grid:         list[dict],
    max_overnight_gap:  int = 4,
) -> tuple[list[dict], list[dict]]:
    """
    Build price panel (once), then run all (window, params) permutations.
    Returns (run_records, all_trade_rows).
    """
    log.info("Loading price panel for %d pairs (one-time, ~2 min) ...", len(pairs))
    panel = _bt.load_panel(adr_prices_path, global_prices_path, fx_rates_path, pairs)
    log.info("Panel built for %d pairs.", len(panel))

    total_runs  = len(windows) * len(param_grid)
    log.info("Starting %d research runs ...", total_runs)

    run_records:    list[dict] = []
    all_trade_rows: list[dict] = []
    run_idx = 0

    for window in windows:
        test_start = window["test_start"]
        test_end   = window["test_end"]
        train_end  = window["train_end"]

        for params in param_grid:
            run_idx += 1
            run_id = (
                f"tr{window['train_years']}y"
                f"_te{window['test_years']}y"
                f"_{test_start.year}"
                f"_T{params['T']}"
                f"_k{params['k0']:.2f}"
                f"_kc{params['kc']:.1f}"
                f"_H{params['H']}"
            )

            meta = {
                "run_id":     run_id,
                "train_start": window["train_start"],
                "train_end":   train_end,
                "test_start":  test_start,
                "test_end":    test_end,
                "train_years": window["train_years"],
                "test_years":  window["test_years"],
                "T":           params["T"],
                "k0":          params["k0"],
                "kc":          params["kc"],
                "H":           params["H"],
            }

            # Apply parameter overrides to pair objects
            for p in pairs:
                p.estimation_days = params["T"]
                p.holding_days    = params["H"]
                p.k0              = params["k0"]
                p.kc              = params["kc"]

            cfg = {"T": params["T"], "H": params["H"],
                   "k0": params["k0"], "kc": params["kc"]}

            run_trades:     list = []
            eligible_pairs: list = []

            for pair in pairs:
                df = panel.get(pair.pair_id)
                if df is None:
                    continue
                # Restrict to <= test_end (no future price data)
                df_run = df[df.index <= pd.Timestamp(test_end)]
                # Need enough pre-test data for rolling warm-up
                pre = df_run[df_run.index < pd.Timestamp(test_start)]
                if len(pre) < params["T"] + 10:
                    continue
                eligible_pairs.append(pair)
                pair_trades = _bt.backtest_pair(
                    pair, df_run, cfg,
                    start=test_start,
                    end=test_end,
                    max_overnight_gap_days=max_overnight_gap,
                )
                run_trades.extend(pair_trades)

            rec = _run_stats(run_trades, eligible_pairs, meta)
            rec["score"] = compute_score(rec)
            run_records.append(rec)

            # Collect trade rows with run metadata
            for t in run_trades:
                row = asdict(t)
                row["run_id"]      = run_id
                row["test_start"]  = test_start.isoformat()
                row["test_end"]    = test_end.isoformat()
                row["train_years"] = window["train_years"]
                row["test_years"]  = window["test_years"]
                row["T"]           = params["T"]
                row["k0"]          = params["k0"]
                row["kc"]          = params["kc"]
                row["H"]           = params["H"]
                # Serialise enums
                for k, v in row.items():
                    if hasattr(v, "value"):
                        row[k] = v.value
                all_trade_rows.append(row)

            if run_idx % 50 == 0 or run_idx == total_runs:
                log.info(
                    "  [%d/%d] %s  eligible=%d  closed=%d"
                    "  RUCE_net_med=%s",
                    run_idx, total_runs, run_id,
                    len(eligible_pairs),
                    rec.get("n_closed", 0),
                    f"{rec.get('ruce_net_median', float('nan')):.4f}",
                )

    return run_records, all_trade_rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _table7(run_records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in run_records:
        rows.append({
            "run_id":        r["run_id"],
            "train_yrs":     r["train_years"],
            "test_yrs":      r["test_years"],
            "test_start":    r["test_start"],
            "test_end":      r["test_end"],
            "T":             r["estimation_days"],
            "k0":            r["k0"],
            "kc":            r["kc"],
            "H":             r["holding_days"],
            "n_pairs":       r["n_active_pairs"],
            "n_trades":      r["n_trades"],
            "n_closed":      r["n_closed"],
            "n_abort":       r["n_overnight_aborts"],
            "n_force":       r["n_force_closes"],
            "ROCE_med":      r.get("roce_median"),
            "RUCE_med":      r.get("ruce_median"),
            "ROCE_net_med":  r.get("roce_net_median"),
            "RUCE_net_med":  r.get("ruce_net_median"),
            "RUCE_net_mean": r.get("ruce_net_mean"),
            "RUCE_net_std":  r.get("ruce_net_std"),
            "RUCE_net_p90":  r.get("ruce_net_p90"),
            "RUCE_net_p75":  r.get("ruce_net_p75"),
            "RUCE_net_p25":  r.get("ruce_net_p25"),
            "RUCE_net_p10":  r.get("ruce_net_p10"),
            "dur_med":       r.get("duration_median"),
            "dur_mean":      r.get("duration_mean"),
            "win_rate":      r.get("win_rate"),
            "trades_per_yr": r.get("trades_per_year"),
            "max_pair_pct":  r.get("max_pair_pct"),
            "score":         r.get("score"),
        })
    return pd.DataFrame(rows)


def _best_perm(run_records: list[dict], top_n: int = 10) -> pd.DataFrame:
    ranked = sorted(
        run_records,
        key=lambda r: r.get("score", float("-inf")),
        reverse=True,
    )
    rows = []
    for rank, r in enumerate(ranked[:top_n], 1):
        sc = r.get("score", float("-inf"))
        rows.append({
            "rank":           rank,
            "run_id":         r["run_id"],
            "train_start":    r["train_start"],
            "train_end":      r["train_end"],
            "test_start":     r["test_start"],
            "test_end":       r["test_end"],
            "train_yrs":      r["train_years"],
            "test_yrs":       r["test_years"],
            "T":              r["estimation_days"],
            "k0":             r["k0"],
            "kc":             r["kc"],
            "H":              r["holding_days"],
            "n_active_pairs": r["n_active_pairs"],
            "n_trades":       r["n_trades"],
            "n_closed":       r["n_closed"],
            "ROCE_net_med":   r.get("roce_net_median"),
            "RUCE_net_med":   r.get("ruce_net_median"),
            "RUCE_net_mean":  r.get("ruce_net_mean"),
            "win_rate":       r.get("win_rate"),
            "dur_med":        r.get("duration_median"),
            "max_pair_pct":   r.get("max_pair_pct"),
            "score":          round(sc, 6) if np.isfinite(sc) else sc,
            "comments":       (
                "PASS" if np.isfinite(sc)
                else (
                    f"FAIL n_closed={r.get('n_closed',0)} "
                    f"n_pairs={r.get('n_active_pairs',0)} "
                    f"ruce_med={r.get('ruce_net_median')}"
                )
            ),
        })
    return pd.DataFrame(rows)


def _pair_diag(all_trade_rows: list[dict]) -> pd.DataFrame:
    if not all_trade_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_trade_rows)
    closed = df[~df["was_aborted"]]
    if closed.empty:
        return pd.DataFrame()
    grp = closed.groupby(["run_id", "pair_id"])
    return grp["ruce_net"].agg(
        n_trades="count",
        mean_ruce="mean",
        median_ruce="median",
        win_rate=lambda x: (x > 0).mean(),
    ).reset_index()


def _safe_json(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    raise TypeError(f"not JSON serialisable: {type(obj)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rolling train/test research runner — Hong & Susmel ADR strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adr-prices",    type=Path, default=_DEFAULT_ADR)
    p.add_argument("--global-prices", type=Path, default=_DEFAULT_GLOBAL)
    p.add_argument("--fx-rates",      type=Path, default=_DEFAULT_FX)
    p.add_argument("--pairs",         type=Path, default=_DEFAULT_PAIRS)
    p.add_argument("--out-dir",       type=Path, default=_DEFAULT_OUT)
    p.add_argument(
        "--train-years", type=int, nargs="+", default=[5],
        help="Training window lengths in years",
    )
    p.add_argument(
        "--test-years",  type=int, nargs="+", default=[1, 2, 4],
        help="Test window lengths in years",
    )
    p.add_argument(
        "--full-grid",   action="store_true",
        help="Use full (T, k0, kc, H) grid — slower but more comprehensive",
    )
    p.add_argument("--max-overnight-gap", type=int, default=4)
    p.add_argument("--top-n",            type=int, default=10)
    return p.parse_args(argv)


def _load_pairs(path: Path) -> list:
    raw = json.loads(path.read_text())
    pairs = []
    for r in raw:
        if not r.get("is_active", True):
            continue
        try:
            ratio = float(r["adr_ratio"])
        except (KeyError, TypeError, ValueError):
            continue
        if ratio <= 0:
            continue
        pairs.append(_bt.Pair(
            pair_id=r["pair_id"],
            adr_ticker=r["adr_ticker"],
            underlying_ticker=r["underlying_ticker"],
            underlying_exchange=r["underlying_exchange"],
            underlying_currency=r["underlying_currency"],
            adr_ratio=ratio,
            estimation_days=int(r.get("estimation_days", 60)),
            holding_days=int(r.get("holding_days", 90)),
            k0=float(r.get("k0", 2.0)),
            kc=float(r.get("kc", 0.0)),
            roll_spread_adr=float(r.get("roll_spread_adr", 0.0)),
            roll_spread_local=float(r.get("roll_spread_local", 0.0)),
        ))
    return pairs


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    for pth, label in [
        (args.adr_prices,    "adr_prices"),
        (args.global_prices, "global_prices"),
        (args.fx_rates,      "fx_rates"),
        (args.pairs,         "pairs registry"),
    ]:
        if not pth.exists():
            log.error("%s not found: %s", label, pth)
            return 2

    pairs = _load_pairs(args.pairs)
    log.info("Loaded %d active pairs", len(pairs))

    windows = generate_rolling_windows(
        train_years_list=args.train_years,
        test_years_list=args.test_years,
    )
    log.info(
        "Generated %d rolling windows  (train_years=%s, test_years=%s)",
        len(windows), args.train_years, args.test_years,
    )

    param_grid = build_param_grid(full=args.full_grid)
    log.info(
        "Parameter grid: %d combinations  (%s mode)",
        len(param_grid), "FULL" if args.full_grid else "STANDARD",
    )

    total_runs = len(windows) * len(param_grid)
    log.info("Total research runs: %d × %d = %d", len(windows), len(param_grid), total_runs)

    t0 = datetime.now()
    run_records, all_trade_rows = run_research(
        adr_prices_path=args.adr_prices,
        global_prices_path=args.global_prices,
        fx_rates_path=args.fx_rates,
        pairs=pairs,
        windows=windows,
        param_grid=param_grid,
        max_overnight_gap=args.max_overnight_gap,
    )
    elapsed = (datetime.now() - t0).total_seconds()
    log.info("Completed in %.1f s (%.1f min)", elapsed, elapsed / 60)

    # Score
    for rec in run_records:
        if "score" not in rec:
            rec["score"] = compute_score(rec)

    n_valid  = sum(1 for r in run_records if np.isfinite(r.get("score", float("-inf"))))
    n_failed = len(run_records) - n_valid

    # Save outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs_df = pd.DataFrame(run_records)
    runs_df.to_csv(args.out_dir / "research_runs.csv", index=False)
    with (args.out_dir / "research_runs.json").open("w") as fh:
        json.dump(run_records, fh, indent=2, default=_safe_json)

    t7 = _table7(run_records)
    t7.to_csv(args.out_dir / "table_7_style_summary.csv",  index=False)
    t7.to_csv(args.out_dir / "research_summary_table.csv", index=False)

    best_df = _best_perm(run_records, top_n=args.top_n)
    best_df.to_csv(args.out_dir / "best_permutations.csv", index=False)

    if all_trade_rows:
        pd.DataFrame(all_trade_rows).to_csv(
            args.out_dir / "all_trades_by_run.csv", index=False)

    pair_diag = _pair_diag(all_trade_rows)
    if not pair_diag.empty:
        pair_diag.to_csv(args.out_dir / "pair_diagnostics_by_run.csv", index=False)

    # Print summary
    log.info("=" * 72)
    log.info("RESEARCH PERMUTATIONS SUMMARY")
    log.info("=" * 72)
    log.info("  Total runs:    %d", total_runs)
    log.info("  Valid runs:    %d  (passed min filters)", n_valid)
    log.info("  Failed runs:   %d  (zero trades / too few pairs / median <= 0)", n_failed)
    log.info("  Elapsed:       %.1f s", elapsed)
    log.info("  Output dir:    %s", args.out_dir)

    if not best_df.empty and n_valid > 0:
        log.info("")
        log.info("TOP %d PERMUTATIONS", args.top_n)
        log.info("%-3s  %-35s  %-3s  %-4s  %-4s  %-3s  %6s  %6s  %8s  %7s  %7s",
                 "Rk", "run_id", "T", "k0", "kc", "H",
                 "pairs", "trades", "RUCE_n_md", "win%", "score")
        log.info("-" * 100)
        for _, row in best_df.iterrows():
            sc = row["score"]
            log.info(
                "#%-2d  %-35s  %-3d  %-4.2f  %-4.1f  %-3d  %6d  %6d  %8.4f  %6.1f%%  %7.4f",
                row["rank"],
                str(row["run_id"])[:35],
                int(row["T"]), row["k0"], row["kc"], int(row["H"]),
                int(row["n_active_pairs"]),
                int(row["n_closed"]),
                row["RUCE_net_med"] if row["RUCE_net_med"] is not None else float("nan"),
                (row["win_rate"] * 100) if row["win_rate"] is not None else float("nan"),
                sc if np.isfinite(sc) else float("nan"),
            )
    else:
        log.warning("No valid permutations — check data coverage, min_test_obs, and filters.")

    log.info("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
