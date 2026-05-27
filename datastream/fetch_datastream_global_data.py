#!/usr/bin/env python3
"""
fetch_datastream_global_data.py
===============================

Fetch Asian-underlying OHLCV from WRDS Datastream (``tr_ds2.wrds_ds2dsf``)
and cache to Parquet. The set of underlying tickers / exchanges is read
from the ADR reference table written by ``fetch_datastream_adr_data.py``.

Eligible Asian exchanges (per ADR Architecture spec §3.4)::

    {TSE, HKEX, KRX, BSE, NSE, ASX, SGX, TWSE, IDX, PSE, BURSA}

Requires
--------
- ``wrds`` Python library installed (``pip install wrds``)
- Environment variables ``WRDS_USERNAME`` and ``WRDS_PASSWORD`` set, OR
  a ``~/.pgpass`` entry for the WRDS server

Usage
-----
::

    python scripts/fetch_datastream_global_data.py \
        --start 2018-01-01 --end 2024-12-31 \
        --adr-reference data/parquet/adr/adr_reference.parquet \
        --out data/parquet/global

Output
------
``<out>/global_prices.parquet`` with columns::

    infocode, marketdate, ticker, exchange, currency,
    open, high, low, close, volume, adj_factor
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("fetch_global")

start_date = date(2026, 4, 30) - timedelta(days=365 * 10)
end_date = date(2026, 4, 30)

ASIAN_EXCHANGES: set[str] = {
    "TKS", "TSE",        # Japan (Tokyo)
    "HKG",               # Hong Kong
    "KRX",               # Korea
    "BOM", "BSE",        # India Bombay
    "NSE",               # India National
    "ASX",               # Australia
    "SES",               # Singapore
    "TAI",               # Taiwan
    "IDX",               # Indonesia
    "PHS",               # Philippines
    "KLS",               # Malaysia (Bursa)
}

WRDS_USERNAME = "nglei2025"
WRDS_PASSWORD = "Police@123456789"


# -----------------------------------------------------------------------------
# WRDS SQL
# -----------------------------------------------------------------------------
GLOBAL_PRICES_SQL = """
    SELECT
        n.infocode,
        d.marketdate,
        d.close        AS close,
        d.high         AS high,
        d.low          AS low,
        d.open         AS open,
        d.volume       AS volume,
        d.cumadjfactor AS adj_factor,
        n.dscode       AS ticker,
        n.primexchmnem AS exchange,
        n.isocurrcode  AS currency
    FROM tr_ds_equities.wrds_ds2dsf  AS d
    JOIN tr_ds_equities.wrds_ds_names AS n
      ON d.infocode = n.infocode
    WHERE d.marketdate BETWEEN %(start_date)s AND %(end_date)s
      AND n.primexchmnem = ANY(%(exchanges)s)
      AND n.dscode       = ANY(%(tickers)s)
    ORDER BY n.infocode, d.marketdate
"""


# -----------------------------------------------------------------------------
# WRDS connection
# -----------------------------------------------------------------------------
def require_wrds() -> "wrds.Connection":  # type: ignore[name-defined]
    try:
        import wrds  # type: ignore
    except ImportError as e:
        log.error("The `wrds` package is not installed.\n"
                  "  Install with: pip install wrds")
        raise SystemExit(2) from e

    log.info("connecting to WRDS as user=%s", WRDS_USERNAME)
    return wrds.Connection(wrds_username=WRDS_USERNAME, wrds_password=WRDS_PASSWORD)


def fetch_from_wrds(
    start: date,
    end: date,
    tickers: list[str],
    exchanges: list[str],
) -> pd.DataFrame:
    db = require_wrds()
    try:
        log.info("fetching %d underlying tickers across %d exchanges",
                 len(tickers), len(exchanges))
        df = db.raw_sql(
            GLOBAL_PRICES_SQL,
            params={
                "start_date": start.isoformat(),
                "end_date":   end.isoformat(),
                "tickers":    tickers,
                "exchanges":  exchanges,
            },
            date_cols=["marketdate"],
        )
    finally:
        db.close()

    if df.empty:
        log.error("WRDS returned zero rows for the requested underlyings. "
                  "Check tickers, exchanges, and date range.")
        raise SystemExit(3)

    returned = set(df["ticker"].astype(str).unique())
    requested = set(tickers)
    missing = sorted(requested - returned)
    if missing:
        log.warning("%d/%d requested underlyings returned no data: %s",
                    len(missing), len(requested),
                    ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))

    log.info("WRDS returned %d price rows for %d underlyings",
             len(df), len(returned))
    return df


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def load_adr_reference(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.error("ADR reference not found at %s.\n"
                  "  Run scripts/fetch_datastream_adr_data.py first.", path)
        raise SystemExit(2)
    ref = pd.read_parquet(path)
    log.info("loaded ADR reference: %d rows", len(ref))
    return ref


def filter_to_asian(adr_reference: pd.DataFrame) -> pd.DataFrame:
    mask = adr_reference["underlying_exchange"].isin(ASIAN_EXCHANGES)
    filtered = adr_reference[mask].copy()
    log.info("filtered to Asian exchanges: %d -> %d rows",
             len(adr_reference), len(filtered))
    return filtered


def write_output(df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "global_prices.parquet"
    df.to_parquet(path, index=False, compression="snappy")
    log.info("wrote %s (%d rows, %.1f KB)",
             path, len(df), path.stat().st_size / 1024)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch Asian underlying OHLCV from WRDS Datastream into Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", type=_parse_date,
                   default=start_date)
    p.add_argument("--end", type=_parse_date, default=end_date)
    p.add_argument("--adr-reference", type=Path,
                   default=Path("data/parquet/adr/adr_reference.parquet"),
                   help="Path to ADR reference Parquet (output of fetch_datastream_adr_data.py)")
    p.add_argument("--out", type=Path, default=Path("data/parquet/global"))
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.start >= args.end:
        log.error("--start must be strictly before --end")
        return 2

    adr_reference = load_adr_reference(args.adr_reference)
    asian_only = filter_to_asian(adr_reference)
    if asian_only.empty:
        log.error("No ADRs with Asian-exchange underlyings in reference; abort.")
        return 3

    tickers = asian_only["underlying_ticker"].astype(str).tolist()
    exchanges = sorted(asian_only["underlying_exchange"].unique().tolist())
    df = fetch_from_wrds(args.start, args.end, tickers, exchanges)

    write_output(df, args.out)
    log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
