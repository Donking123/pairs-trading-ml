#!/usr/bin/env python3
"""
fetch_datastream_adr_data.py
============================

Fetch U.S.-listed ADR OHLCV from WRDS Datastream (``tr_ds2.wrds_ds2dsf``,
``type_='ADR'``) and cache to Parquet for downstream screening / backtesting.

Requires
--------
- ``wrds`` Python library installed (``pip install wrds``)
- Environment variables ``WRDS_USERNAME`` and ``WRDS_PASSWORD`` set, OR a
  ``~/.pgpass`` entry for the WRDS server (recommended for production)

Usage
-----
::

    python scripts/fetch_datastream_adr_data.py \
        --out data/parquet/adr

Output
------
``<out>/adr_prices.parquet`` with columns::

    infocode, marketdate, ticker, isin, open, high, low, close, volume, adj_factor

``<out>/adr_reference.parquet`` with the ADR -> underlying mapping and ratio.

Notes
-----
Per ADR Architecture spec §3.1 (ADR-002): the conversion ratio (``adrr``) is
the **fixed structural constant** used as the cointegrating vector. It is
read verbatim from Datastream — never OLS-estimated.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("fetch_adr")

start_date = date(2026, 4, 30) - timedelta(days=365 * 20)
end_date = date(2026, 4, 30)

WRDS_USERNAME = os.environ.get("WRDS_USERNAME")
WRDS_PASSWORD = os.environ.get("WRDS_PASSWORD")

# -----------------------------------------------------------------------------
# WRDS SQL (verbatim — keep this script independent of src/)
# -----------------------------------------------------------------------------
ADR_PRICES_SQL = """
    SELECT
        n.infocode,
        d.marketdate,
        d.close,
        d.high,
        d.low,
        d.open,
        d.volume,
        d.cumadjfactor AS adj_factor,
        n.dscode  AS ticker,
        n.isin
    FROM tr_ds_equities.wrds_ds2dsf  AS d
    JOIN tr_ds_equities.wrds_ds_names AS n
      ON d.infocode = n.infocode
    WHERE d.marketdate BETWEEN %(start_date)s AND %(end_date)s
      AND n.region   = 'US'
      AND n.typecode = 'ADR'
    ORDER BY n.infocode, d.marketdate
"""

ADR_REFERENCE_SQL = """
    SELECT
        a.dscode       AS adr_ticker,
        a.isin         AS adr_isin,
        a.infocode     AS adr_infocode,
        a.startdate    AS adr_startdate,
        a.enddate      AS adr_enddate,
        u.dscode       AS underlying_ticker,
        u.primexchmnem AS underlying_exchange,
        u.isocurrcode  AS underlying_currency,
        u.startdate    AS underlying_startdate,
        u.enddate      AS underlying_enddate,
        NULL::float    AS adr_ratio
    FROM tr_ds_equities.wrds_ds_names AS a
    JOIN tr_ds_equities.wrds_ds_names AS u
      ON a.dscompcode  = u.dscompcode
     AND a.dscompcode IS NOT NULL
     AND a.infocode   != u.infocode
     AND u.region     != 'US'
    WHERE a.region   = 'US'
      AND a.typecode = 'ADR'
"""


# -----------------------------------------------------------------------------
# WRDS connection
# -----------------------------------------------------------------------------
def require_wrds() -> "wrds.Connection":  # type: ignore[name-defined]
    """Import wrds, verify credentials, return an open connection."""
    try:
        import wrds  # type: ignore
    except ImportError as e:
        log.error("The `wrds` package is not installed.\n"
                  "  Install with: pip install wrds")
        raise SystemExit(2) from e

    log.info("connecting to WRDS as user=%s", WRDS_USERNAME)
    kwargs: dict = {"wrds_username": WRDS_USERNAME}
    if WRDS_PASSWORD is not None:
        kwargs["wrds_password"] = WRDS_PASSWORD
    return wrds.Connection(**kwargs)


def discover_tables() -> None:
    """Print all tables in tr_ds* libraries using the WRDS built-in listing."""
    db = require_wrds()
    try:
        libraries = [lib for lib in db.list_libraries() if lib.startswith("tr_ds")]
        if not libraries:
            log.warning("No tr_ds* libraries found — check your WRDS subscription.")
            return
        for lib in sorted(libraries):
            tables = db.list_tables(library=lib)
            log.info("Library %s: %s", lib, tables)
    finally:
        db.close()


def fetch_from_wrds(start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Connect to WRDS and run the ADR price + reference queries."""
    db = require_wrds()
    try:
        log.info("running ADR prices query (start=%s, end=%s)", start, end)
        prices = db.raw_sql(
            ADR_PRICES_SQL,
            params={"start_date": start.isoformat(), "end_date": end.isoformat()},
            date_cols=["marketdate"],
        )
        log.info("running ADR reference query")
        reference = db.raw_sql(
            ADR_REFERENCE_SQL,
            date_cols=["adr_startdate", "adr_enddate",
                       "underlying_startdate", "underlying_enddate"],
        )
    finally:
        db.close()

    # wrds_ds_names stores one row per name-period (a new row is added each time
    # a stock changes its name or ticker). The cross-join above therefore produces
    # N_adr_periods × N_underlying_periods rows per economic pair, many with
    # invalid or overlapping date windows. Collapse to one row per unique pair by
    # taking the union of all name-period windows for each leg independently, then
    # computing the intersection at the pair level.
    #
    # WRDS uses 2079-06-05 as a sentinel for "still active" instead of NULL.
    # Convert sentinel to NaT before aggregating so MAX() treats open-ended
    # records correctly (a real far-future date would dominate MAX wrongly).
    sentinel = pd.Timestamp("2079-06-05")
    for col in ["adr_startdate", "adr_enddate", "underlying_startdate", "underlying_enddate"]:
        reference[col] = pd.to_datetime(reference[col])
        reference.loc[reference[col] == sentinel, col] = pd.NaT

    # Per-leg union: earliest start and latest end across all name periods.
    # NaT enddate = still active; keeping skipna=True means MAX ignores NaT only
    # when at least one concrete end exists — but if ANY period is open-ended we
    # want NaT (still active) to win. Use a custom agg: NaT if any NaT, else max.
    def _max_end(s):
        return pd.NaT if s.isna().any() else s.max()

    key_cols = ["adr_ticker", "adr_isin", "adr_infocode",
                "underlying_ticker", "underlying_exchange", "underlying_currency"]
    reference = (
        reference
        .groupby(key_cols, dropna=False)
        .agg(
            adr_startdate       =("adr_startdate",        "min"),
            adr_enddate         =("adr_enddate",           _max_end),
            underlying_startdate=("underlying_startdate",  "min"),
            underlying_enddate  =("underlying_enddate",    _max_end),
            adr_ratio           =("adr_ratio",             "first"),
        )
        .reset_index()
    )

    # Pair-level active window: intersection of both legs.
    # startdate = later of the two starts (skipna=False: unknown start → NaT,
    #   meaning the pair's start is genuinely unknown; exclude it from PIT screens).
    # enddate   = earlier of the two ends (skipna=True: NaT means still active,
    #   so take the other leg's actual end; both open-ended → NaT = still active).
    reference["startdate"] = reference[
        ["adr_startdate", "underlying_startdate"]
    ].max(axis=1, skipna=False)
    reference["enddate"] = reference[
        ["adr_enddate", "underlying_enddate"]
    ].min(axis=1)
    reference = reference.drop(
        columns=["adr_startdate", "adr_enddate",
                 "underlying_startdate", "underlying_enddate"]
    )

    # Drop pairs where the computed window is invalid (start after end) — these
    # are artefacts of non-overlapping name periods between the two legs.
    valid = (
        reference["startdate"].isna() |
        reference["enddate"].isna() |
        (reference["startdate"] <= reference["enddate"])
    )
    n_invalid = (~valid).sum()
    if n_invalid:
        log.warning("dropped %d pairs with startdate > enddate (name-period artefacts)", n_invalid)
    reference = reference[valid].reset_index(drop=True)

    if prices.empty:
        log.error("WRDS returned zero ADR price rows for the requested range. "
                  "Check your subscription and date range.")
        raise SystemExit(3)
    if reference.empty:
        log.error("WRDS returned zero ADR reference rows. Check your subscription.")
        raise SystemExit(3)

    log.info("WRDS returned %d price rows, %d reference rows",
             len(prices), len(reference))
    return prices, reference


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def write_outputs(prices: pd.DataFrame, reference: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    prices_path = out / "adr_prices.parquet"
    ref_path = out / "adr_reference.parquet"

    prices.to_parquet(prices_path, index=False, compression="snappy")
    reference.to_parquet(ref_path, index=False, compression="snappy")

    log.info("wrote %s (%d rows, %.1f KB)",
             prices_path, len(prices), prices_path.stat().st_size / 1024)
    log.info("wrote %s (%d rows, %.1f KB)",
             ref_path, len(reference), ref_path.stat().st_size / 1024)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch U.S. ADR OHLCV from WRDS Datastream into Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", type=_parse_date,
                   default=start_date,
                   help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=_parse_date, default=end_date,
                   help="End date (YYYY-MM-DD)")
    p.add_argument("--out", type=Path, default=Path("datastream/data/parquet/adr"),
                   help="Output directory for Parquet files")
    p.add_argument("--discover", action="store_true",
                   help="List available tables in tr_ds* schemas and exit")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.discover:
        discover_tables()
        return 0

    if args.start >= args.end:
        log.error("--start must be strictly before --end")
        return 2

    prices, reference = fetch_from_wrds(args.start, args.end)
    write_outputs(prices, reference, args.out)
    log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
