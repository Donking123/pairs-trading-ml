#!/usr/bin/env python3
"""
run_engine_backtest.py
======================

Orchestrates the full pipeline: optionally fetch WRDS data, then run the
production ``BacktestEngine`` (``src/asian_adr/``) over the Parquet cache.

Steps
-----
1. If ``--fetch`` is passed (or any required Parquet file is missing), invoke
   the three WRDS fetch scripts and the pair screener in sequence.
2. Auto-detect the backtest date range from the ADR price Parquet.
3. Override ``approved_date``/``expiry_date`` on every pair so the engine
   treats the full historical window as tradeable (reproduction-backtest mode).
4. Run ``BacktestEngine.from_parquet().run_and_report()`` and print the paper
   benchmark validation table.

Usage
-----
::

    # Use existing Parquet/JSON without re-fetching from WRDS
    python datastream/run_engine_backtest.py

    # Force a fresh WRDS pull before backtesting
    python datastream/run_engine_backtest.py --fetch

    # Specific trading window (rolling stats warm up within the window;
    # first ~T bars generate no signals — that is normal, not lost data)
    python datastream/run_engine_backtest.py --start 2015-01-01 --end 2023-12-31

    # Strategy parameter grid (paper defaults shown)
    python datastream/run_engine_backtest.py --k0 2.0 --kc 0.0 --T 60 --H 90

Output
------
Artefacts written to ``datastream/data/backtest/run_<timestamp>/``:

    summary.json       -- headline metrics and paper benchmark validation flags
    distribution.json  -- Table 7-B distributions (ROCE, RUCE, duration, ...)
    trades.parquet     -- per-trade records
    trades.csv
    tearsheet.html     -- self-contained HTML tearsheet (no JS)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd

# ---- import the production package -----------------------------------------
# The datastream/ scripts run from the repo root; src/ is not installed, so we
# insert it onto sys.path before any asian_adr import.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from asian_adr.backtest import BacktestConfig, BacktestEngine  # noqa: E402
from asian_adr.risk import RiskConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("run_engine_backtest")

# ---- canonical file paths --------------------------------------------------
_PARQUET = HERE / "data" / "parquet"
_ADR_PRICES = _PARQUET / "adr" / "adr_prices.parquet"
_ADR_REF = _PARQUET / "adr" / "adr_reference.parquet"
_GLOBAL_PRICES = _PARQUET / "global" / "global_prices.parquet"
_FX_RATES = _PARQUET / "fx" / "fx_rates.parquet"
_PAIRS_JSON = ROOT / "config" / "pairs" / "asian_adr_pairs.json"
_BACKTEST_BASE = HERE / "data" / "backtest"


# ---- helpers ---------------------------------------------------------------

def _read_date_range(parquet_path: Path, date_col: str) -> tuple[date, date]:
    """Return (min_date, max_date) by reading only the date column."""
    try:
        df = pd.read_parquet(parquet_path, columns=[date_col])
    except Exception:
        df = pd.read_parquet(parquet_path, columns=[date_col], engine="fastparquet")
    col = pd.to_datetime(df[date_col]).dt.date
    return col.min(), col.max()


def _run_script(script: Path, *extra_args: str) -> None:
    """Invoke a sibling datastream/ script as a subprocess."""
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
    """Run the four fetch/screening scripts in dependency order."""
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


# ---- CLI -------------------------------------------------------------------

def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch WRDS Parquet data (optional) and run BacktestEngine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # fetch
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
    # backtest window
    p.add_argument(
        "--start", type=_parse_date, default=None,
        help="Backtest start (default: data min-date + T warm-up days)",
    )
    p.add_argument(
        "--end", type=_parse_date, default=None,
        help="Backtest end (default: data max-date)",
    )
    # strategy params
    p.add_argument("--k0", type=float, default=2.0,
                   help="Entry spread multiplier (paper default: 2.0)")
    p.add_argument("--kc", type=float, default=0.0,
                   help="Exit spread multiplier (paper default: 0.0)")
    p.add_argument("--T", dest="T", type=int, default=60,
                   help="Rolling estimation window in days (paper default: 60)")
    p.add_argument("--H", dest="H", type=int, default=90,
                   help="Max holding period in days (paper default: 90)")
    p.add_argument("--notional", type=float, default=100_000,
                   help="USD notional per ADR leg")
    # file paths
    p.add_argument("--adr-prices",    type=Path, default=_ADR_PRICES)
    p.add_argument("--global-prices", type=Path, default=_GLOBAL_PRICES)
    p.add_argument("--fx-rates",      type=Path, default=_FX_RATES)
    p.add_argument("--pairs",         type=Path, default=_PAIRS_JSON)
    p.add_argument("--out-dir",       type=Path, default=None,
                   help="Output directory (default: datastream/data/backtest/run_<ts>)")
    return p.parse_args(argv)


# ---- main ------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # -- step 1: optionally fetch WRDS data ------------------------------------
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

    # -- step 2: verify files exist after optional fetch -----------------------
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
            "Run `python datastream/run_engine_backtest.py --fetch` to pull from WRDS."
        )
        return 2

    # -- step 3: auto-detect date range ----------------------------------------
    data_start, data_end = _read_date_range(args.adr_prices, "marketdate")
    log.info("ADR price data:  %s → %s", data_start, data_end)

    # No artificial warm-up offset: WelfordRollingStats warms up naturally
    # inside HongSusmelEngine (it suppresses signals until count >= T).
    # Skipping the first T bars would silently discard Parquet data rather
    # than pre-filling the rolling window, since _merge() only streams [start, end].
    bt_start = args.start or data_start
    bt_end = args.end or data_end

    if bt_start >= bt_end:
        log.error("--start (%s) must be before --end (%s)", bt_start, bt_end)
        return 2
    log.info("backtest window: %s → %s", bt_start, bt_end)

    # -- step 4: build BacktestConfig ------------------------------------------
    # param_overrides serves two purposes:
    #   - apply CLI strategy params (k0/kc/T/H) to every pair in the registry
    #   - override approved_date / expiry_date so all pairs are active across
    #     the full historical sample (reproduction-backtest mode; the screener
    #     sets these to the screening date, not the data period)
    #
    # RiskConfig: use a large AUM baseline so the drawdown kill switch does not
    # fire on normal historical volatility.  With $100K notional per leg and 20
    # simultaneous pairs, gross exposure is ~$2M; a 5% drawdown on $1M AUM
    # (the production default) would trigger on a single bad month.  $10M gives
    # ~4.4× headroom and is realistic for a multi-pair backtest.
    risk_config = RiskConfig(aum_usd=Decimal("10000000"))
    config = BacktestConfig(
        risk_config=risk_config,
        notional_per_leg=Decimal(str(int(args.notional))),
        param_overrides={
            "k0": args.k0,
            "kc": args.kc,
            "estimation_days": args.T,
            "holding_days": args.H,
            "approved_date": str(data_start),
            "expiry_date": str(data_end),
        },
    )
    engine = BacktestEngine.from_parquet(
        config,
        adr_prices=str(args.adr_prices),
        global_prices=str(args.global_prices),
        fx_rates=str(args.fx_rates),
        pairs_json=str(args.pairs),
    )

    # -- step 5: run -----------------------------------------------------------
    out_dir = args.out_dir or (
        _BACKTEST_BASE / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    log.info("output directory: %s", out_dir)
    log.info("starting backtest (k0=%.1f kc=%.1f T=%d H=%d) ...",
             args.k0, args.kc, args.T, args.H)

    results = asyncio.run(engine.run_and_report(bt_start, bt_end, str(out_dir)))

    # -- step 6: print summary -------------------------------------------------
    summary_path = Path(out_dir) / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        log.info("=" * 64)
        log.info("BACKTEST COMPLETE — %d round-trips", len(results))
        log.info("=" * 64)
        skip = {"config", "validation"}
        for k, v in summary.items():
            if k not in skip:
                log.info("  %-28s %s", k, v)
        log.info("  --- paper benchmark validation ---")
        for k, chk in summary.get("validation", {}).items():
            status = "PASS" if chk["within_tolerance"] else "FAIL"
            val = chk["value"]
            val_str = f"{val:.4f}" if val is not None else "n/a"
            log.info("  %-28s %-4s  (actual=%s  target=%.4f)",
                     k, status, val_str, chk["target"])
        log.info("=" * 64)
        log.info("tearsheet → %s/tearsheet.html", out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
