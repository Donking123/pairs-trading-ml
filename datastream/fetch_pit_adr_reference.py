#!/usr/bin/env python3
"""
fetch_pit_adr_reference.py
==========================

Fetch a **point-in-time** ADR reference table from WRDS Datastream that
includes the ``startdate`` and ``enddate`` of each name record. This is the
key data needed to fix survivorship bias: instead of using a snapshot of
currently-active ADRs, the screening pipeline can filter to only pairs that
were active as of the screening date.

The existing ``adr_reference.parquet`` is a static snapshot (no date fields).
This script produces ``adr_reference_pit.parquet`` which adds:

    startdate   date   first date the ADR was active in Datastream
    enddate     date   last date (NaT = still active today)

Usage
-----
::

    python datastream/fetch_pit_adr_reference.py \
        --out datastream/data/parquet/adr/adr_reference_pit.parquet

Then pass ``--adr-reference-pit`` to run_walkforward.py / run_param_grid.py
to use point-in-time screening.

WRDS table
----------
``tr_ds_equities.wrds_ds_names`` has columns including:
    infocode, dscode, startdate, enddate, typecode, region, dscompcode,
    primexchmnem, isocurrcode
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("fetch_pit_ref")

WRDS_USERNAME = os.environ["WRDS_USERNAME"]
WRDS_PASSWORD = os.environ.get("WRDS_PASSWORD")

# Pull ADR names with their active date range.
# enddate IS NULL means still active (we'll represent as NaT).
ADR_PIT_SQL = """
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
    ORDER BY a.dscode, u.dscode
"""


def require_wrds():
    try:
        import wrds
    except ImportError as e:
        log.error("wrds package not installed. Run: pip install wrds")
        raise SystemExit(2) from e
    log.info("connecting to WRDS as %s", WRDS_USERNAME)
    return wrds.Connection(wrds_username=WRDS_USERNAME, wrds_password=WRDS_PASSWORD)


def fetch(out_path: Path) -> None:
    db = require_wrds()
    try:
        log.info("fetching point-in-time ADR reference …")
        df = db.raw_sql(ADR_PIT_SQL, date_cols=["adr_startdate", "adr_enddate",
                                                  "underlying_startdate", "underlying_enddate"])
    finally:
        db.close()

    if df.empty:
        log.error("WRDS returned zero rows — check subscription")
        raise SystemExit(3)

    log.info("fetched %d rows", len(df))

    # Pair-level active window: the pair is active only while BOTH the ADR and
    # the underlying name record are active — i.e. the intersection of the two
    # leg windows.
    #
    # startdate = later of the two starts. A NaT start means the leg's start is
    #   unknown, so the pair's start is genuinely unknown too; propagate NaT
    #   (skipna=False) rather than silently adopting the other leg's start, which
    #   would mark the pair active earlier than we can justify.
    # enddate = earlier of the two *actual* ends. Here NaT means "still active",
    #   so it is correct to skip NaT (skipna default): the pair stops being
    #   active when the FIRST leg ends; if both legs are open-ended the min is
    #   NaT (still active). Do NOT use skipna=False here, or one active leg would
    #   wrongly force the whole pair to NaT-end.
    df["startdate"] = df[["adr_startdate", "underlying_startdate"]].max(axis=1, skipna=False)
    df["enddate"] = df[["adr_enddate", "underlying_enddate"]].min(axis=1)

    out = df[["adr_ticker", "adr_isin", "adr_infocode",
              "underlying_ticker", "underlying_exchange", "underlying_currency",
              "adr_ratio", "startdate", "enddate"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False, compression="snappy")
    log.info("wrote %s (%d rows, %.1f KB)", out_path, len(out),
             out_path.stat().st_size / 1024)

    # Summary
    still_active = out["enddate"].isna().sum()
    delisted     = out["enddate"].notna().sum()
    log.info("pairs still active: %d  |  delisted/ended: %d", still_active, delisted)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch point-in-time ADR reference with startdate/enddate from WRDS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", type=Path,
                   default=Path("datastream/data/parquet/adr/adr_reference_pit.parquet"))
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    fetch(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
