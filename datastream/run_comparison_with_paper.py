#!/usr/bin/env python3
"""
run_comparison_with_paper.py
============================

Orchestrates the full pipeline: optionally fetch WRDS data, run the backtest
over the Parquet cache, then print a paper benchmark validation table.

Steps
-----
1. If ``--fetch`` is passed (or any required Parquet file is missing), invoke
   the three WRDS fetch scripts and the pair screener in sequence.
2. Auto-detect the backtest date range from the ADR price Parquet.
3. Run the full backtest (all pairs treated as active over the full window).
4. Compare headline metrics against Hong & Susmel (2013) paper benchmarks
   (Table 7-B, k0=2, kc=0, T=60, H=90).

Usage
-----
::

    # Use existing Parquet/JSON without re-fetching from WRDS
    python datastream/run_comparison_with_paper.py

    # Force a fresh WRDS pull before backtesting
    python datastream/run_comparison_with_paper.py --fetch

    # Specific trading window
    python datastream/run_comparison_with_paper.py --start 2015-01-01 --end 2023-12-31

    # Strategy parameter grid (paper defaults shown)
    python datastream/run_comparison_with_paper.py --k0 2.0 --kc 0.0 --T 60 --H 90

Output
------
Artefacts written to ``datastream/data/backtest/run_<timestamp>/``:

    summary.json       -- headline metrics + paper benchmark validation flags
    distribution.json  -- Table 7-B distributions (ROCE, RUCE, duration, ...)
    trades.parquet     -- per-trade records
    trades.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("run_comparison_with_paper")

# ---- load the inline backtest module ----------------------------------------
# run_backtest.py lives in the same directory and contains all strategy logic.
# Use importlib so it works regardless of sys.path / package installation state.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_bt_path = HERE / "run_backtest.py"
_spec = importlib.util.spec_from_file_location("ds_backtest", _bt_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load backtest module from {_bt_path}")
_bt = importlib.util.module_from_spec(_spec)
sys.modules["ds_backtest"] = _bt
_spec.loader.exec_module(_bt)  # type: ignore[union-attr]

# ---- canonical file paths ---------------------------------------------------
_PARQUET = HERE / "data" / "parquet"
_ADR_PRICES = _PARQUET / "adr" / "adr_prices.parquet"
_ADR_REF = _PARQUET / "adr" / "adr_reference.parquet"
_GLOBAL_PRICES = _PARQUET / "global" / "global_prices.parquet"
_FX_RATES = _PARQUET / "fx" / "fx_rates.parquet"
_PAIRS_JSON = ROOT / "config" / "pairs" / "asian_adr_pairs.json"
_BACKTEST_BASE = HERE / "data" / "backtest"

# ---- paper benchmark targets (Hong & Susmel 2013, Table 7-B) ----------------
# k0=2, kc=0, T=60, H=90 — values from architecture spec §2.4
PAPER_BENCHMARKS = {
    "median_roce": {
        "target": 0.028,
        "tolerance": 0.015,
        "description": "Median ROCE per closed trade (~2.8%)",
    },
    "median_ruce": {
        "target": 0.053,
        "tolerance": 0.025,
        "description": "Median RUCE per closed trade (~5.3%)",
    },
    "median_duration": {
        "target": 3.0,
        "tolerance": 5.0,
        "description": "Median holding period in days (~3 days)",
    },
    "median_ruce_net": {
        "target": 0.027,
        "tolerance": 0.025,
        "description": "Median net RUCE after roll costs (~2.7%)",
    },
}


# ---- helpers ----------------------------------------------------------------

def _read_date_range(parquet_path: Path, date_col: str) -> tuple[date, date]:
    try:
        df = pd.read_parquet(parquet_path, columns=[date_col])
    except Exception:
        df = pd.read_parquet(parquet_path, columns=[date_col], engine="fastparquet")
    col = pd.to_datetime(df[date_col]).dt.date
    return col.min(), col.max()


def _run_script(script: Path, *extra_args: str) -> None:
    cmd = [sys.executable, str(script)] + list(extra_args)
    log.info("running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log.error("%s exited with code %d", script.name, result.returncode)
        raise SystemExit(result.returncode)


def _fetch_wrds(
    fetch_start: date,
    fetch_end: date,
    adr_prices: Path,
    global_prices: Path,
    fx_rates: Path,
    pairs_json: Path,
) -> None:
    _run_script(
        HERE / "fetch_datastream_adr_data.py",
        "--start", str(fetch_start),
        "--end",   str(fetch_end),
        "--out",   str(adr_prices.parent),
    )
    _run_script(
        HERE / "fetch_datastream_global_data.py",
        "--start",         str(fetch_start),
        "--end",           str(fetch_end),
        "--adr-reference", str(_ADR_REF),
        "--out",           str(global_prices.parent),
    )
    _run_script(
        HERE / "fetch_fx_history.py",
        "--start",         str(fetch_start),
        "--end",           str(fetch_end),
        "--adr-reference", str(_ADR_REF),
        "--out",           str(fx_rates.parent),
    )
    _run_script(
        HERE / "run_asian_adr_screening.py",
        "--adr-prices",    str(adr_prices),
        "--adr-reference", str(_ADR_REF),
        "--global-prices", str(global_prices),
        "--fx-rates",      str(fx_rates),
        "--out",           str(pairs_json),
        "--as-of",         str(fetch_end),
    )


def _validate_against_paper(summary: dict) -> dict:
    """
    Compare summary metrics against paper benchmarks.
    Returns a dict keyed by metric name, each value:
        {"value": float|None, "target": float, "tolerance": float,
         "within_tolerance": bool}
    """
    validation = {}
    for metric, spec in PAPER_BENCHMARKS.items():
        actual = summary.get(metric)
        if actual is None:
            within = False
        else:
            within = abs(float(actual) - spec["target"]) <= spec["tolerance"]
        validation[metric] = {
            "value": actual,
            "target": spec["target"],
            "tolerance": spec["tolerance"],
            "within_tolerance": within,
            "description": spec["description"],
        }
    return validation


# ---- CLI --------------------------------------------------------------------

def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run full backtest and compare against Hong & Susmel (2013) paper benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--fetch", action="store_true",
        help="Re-run the WRDS fetch + screening pipeline before backtesting",
    )
    p.add_argument(
        "--fetch-start", type=_parse_date, default=None,
        help="Data start for --fetch (default: 15 years before --fetch-end)",
    )
    p.add_argument(
        "--fetch-end", type=_parse_date, default=date.today(),
        help="Data end for --fetch (default: today)",
    )
    p.add_argument(
        "--start", type=_parse_date, default=None,
        help="Backtest start (default: data min-date)",
    )
    p.add_argument(
        "--end", type=_parse_date, default=None,
        help="Backtest end (default: data max-date)",
    )
    p.add_argument("--k0", type=float, default=2.0,
                   help="Entry spread multiplier (paper default: 2.0)")
    p.add_argument("--kc", type=float, default=0.0,
                   help="Exit spread multiplier (paper default: 0.0)")
    p.add_argument("--T", dest="T", type=int, default=60,
                   help="Rolling estimation window in days (paper default: 60)")
    p.add_argument("--H", dest="H", type=int, default=90,
                   help="Max holding period in days (paper default: 90)")
    p.add_argument("--max-overnight-gap", type=int, default=4, dest="max_overnight_gap",
                   help="Skip two-bar fills where next joint bar is more than this many days away")
    p.add_argument("--adr-prices",    type=Path, default=_ADR_PRICES)
    p.add_argument("--global-prices", type=Path, default=_GLOBAL_PRICES)
    p.add_argument("--fx-rates",      type=Path, default=_FX_RATES)
    p.add_argument("--pairs",         type=Path, default=_PAIRS_JSON)
    p.add_argument("--out-dir",       type=Path, default=None,
                   help="Output directory (default: datastream/data/backtest/run_<ts>)")
    return p.parse_args(argv)


# ---- main -------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # -- step 1: optionally fetch WRDS data -----------------------------------
    data_missing = any(not p.exists() for p in (
        args.adr_prices, args.global_prices, args.fx_rates, args.pairs
    ))
    if args.fetch or data_missing:
        if data_missing and not args.fetch:
            log.info("one or more data files are missing — running fetch pipeline")
        fetch_end = args.fetch_end
        fetch_start = args.fetch_start or (fetch_end - timedelta(days=365 * 15))
        log.info("fetching WRDS data  %s → %s", fetch_start, fetch_end)
        _fetch_wrds(
            fetch_start, fetch_end,
            args.adr_prices, args.global_prices, args.fx_rates, args.pairs,
        )

    # -- step 2: verify files exist -------------------------------------------
    required = {
        "adr-prices":    args.adr_prices,
        "global-prices": args.global_prices,
        "fx-rates":      args.fx_rates,
        "pairs-json":    args.pairs,
    }
    missing = {k: str(v) for k, v in required.items() if not v.exists()}
    if missing:
        for k, path in missing.items():
            log.error("missing: %s  →  %s", k, path)
        log.error(
            "Run with --fetch to pull from WRDS, or supply existing Parquet files."
        )
        return 2

    # -- step 3: auto-detect date range ---------------------------------------
    data_start, data_end = _read_date_range(args.adr_prices, "marketdate")
    log.info("ADR price data:  %s → %s", data_start, data_end)

    bt_start = args.start or data_start
    bt_end = args.end or data_end
    if bt_start >= bt_end:
        log.error("--start (%s) must be before --end (%s)", bt_start, bt_end)
        return 2
    log.info("backtest window: %s → %s", bt_start, bt_end)

    # -- step 4: load pairs and price panel -----------------------------------
    pairs = _bt.load_pairs(args.pairs)
    if not pairs:
        log.error("no pairs in registry; nothing to backtest")
        return 3

    for pair in pairs:
        pair.estimation_days = args.T
        pair.holding_days = args.H
        pair.k0 = args.k0
        pair.kc = args.kc

    cfg = {"T": args.T, "H": args.H, "k0": args.k0, "kc": args.kc,
           "max_overnight_gap": args.max_overnight_gap}
    log.info("backtest config: %s", cfg)
    log.info("starting backtest (%d pairs) ...", len(pairs))

    panel = _bt.load_panel(args.adr_prices, args.global_prices, args.fx_rates, pairs)
    if not panel:
        log.error("price panel empty; abort")
        return 4

    # -- step 5: run per-pair backtest ----------------------------------------
    all_trades: list = []
    for pair in pairs:
        df = panel.get(pair.pair_id)
        if df is None:
            continue
        trades = _bt.backtest_pair(
            pair, df, cfg,
            start=bt_start, end=bt_end,
            max_overnight_gap_days=args.max_overnight_gap,
        )
        all_trades.extend(trades)
        log.info("[%s] %d trades, %d bars", pair.pair_id, len(trades), len(df))

    # -- step 6: build report -------------------------------------------------
    report = _bt.build_report(all_trades, pairs, cfg)
    summary = report["summary"]

    # -- step 7: paper benchmark validation -----------------------------------
    validation = _validate_against_paper(summary)
    summary["validation"] = validation

    # -- step 8: write output -------------------------------------------------
    out_dir = args.out_dir or (
        _BACKTEST_BASE / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("output directory: %s", out_dir)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    (out_dir / "distribution.json").write_text(
        json.dumps(report["distribution"], indent=2, default=str)
    )
    import pandas as pd
    trades_df = pd.DataFrame(report["trades"])
    if not trades_df.empty:
        trades_df.to_csv(out_dir / "trades.csv", index=False)
        trades_df.to_parquet(out_dir / "trades.parquet", index=False)

    # -- step 9: print summary ------------------------------------------------
    log.info("=" * 64)
    log.info("BACKTEST COMPLETE — %d round-trips (%d closed)",
             summary.get("n_trades", 0), summary.get("n_closed", 0))
    log.info("=" * 64)
    skip = {"config", "validation"}
    for k, v in summary.items():
        if k not in skip:
            log.info("  %-28s %s", k, v)
    log.info("  --- paper benchmark validation (Hong & Susmel 2013) ---")
    all_pass = True
    for k, chk in validation.items():
        status = "PASS" if chk["within_tolerance"] else "FAIL"
        val = chk["value"]
        val_str = f"{val:.4f}" if val is not None else "n/a"
        log.info("  %-28s %-4s  (actual=%s  target=%.4f  tol=±%.4f)",
                 k, status, val_str, chk["target"], chk["tolerance"])
        if not chk["within_tolerance"]:
            all_pass = False
    log.info("=" * 64)
    log.info("overall: %s", "ALL PASS" if all_pass else "SOME CHECKS FAILED (see above)")
    log.info("results → %s", out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
