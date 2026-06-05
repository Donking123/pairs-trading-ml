#!/usr/bin/env python3
"""
fetch_fx_history.py
===================

Fetch daily FX close rates for every currency in the ADR reference table
from WRDS Datastream.

All output is normalised to ``USD per 1 unit of base_currency`` (so that
``local_price * mid = USD price``).

Datastream FX tables
--------------------
FX spot rates are stored in ``trdstrm.ds2fxrate`` (daily rates keyed by
``exrateintcode``) joined with ``trdstrm.ds2fxcode`` (maps
``exrateintcode`` to ``fromcurrcode``, ``tocurrcode``, ``ratetypecode``).

All ``fromcurrcode → USD`` SPOT rows store the rate as **CCY per 1 USD**
regardless of the description wording. The script inverts each value to
produce USD-per-CCY for output.

Requires
--------
- ``wrds`` library + ``WRDS_USERNAME`` env var (+ optionally ``WRDS_PASSWORD``
  or a populated ``~/.pgpass`` file).

Usage
-----
::

    python datastream/fetch_fx_history.py \\
        --start 2018-01-01 --end 2024-12-31 \\
        --adr-reference datastream/data/parquet/adr/adr_reference.parquet \\
        --out datastream/data/parquet/fx

Output
------
``<out>/fx_rates.parquet`` with columns::

    date, base_currency, quote_currency, currency_pair, mid, provider

``mid`` is USD per 1 unit of ``base_currency``.
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
log = logging.getLogger("fetch_fx")

start_date = date(2026, 4, 30) - timedelta(days=365 * 10)
end_date = date(2026, 4, 30)

WRDS_USERNAME = os.environ["WRDS_USERNAME"]
WRDS_PASSWORD = os.environ.get("WRDS_PASSWORD")

# Pre-euro legacy currencies superseded by EUR in 1999-2002, plus the old
# Turkish Lira (TRL, replaced by TRY in 2005). Datastream no longer carries
# active SPOT series for these.
EXCLUDED_CURRENCIES: set[str] = {
    "ATS",  # Austrian Schilling
    "BEF",  # Belgian Franc
    "DEM",  # German Deutsche Mark
    "FRF",  # French Franc
    "GRD",  # Greek Drachma
    "ITL",  # Italian Lira
    "NLG",  # Dutch Guilder
    "PTE",  # Portuguese Escudo
    "TRL",  # Old Turkish Lira
}


# -----------------------------------------------------------------------------
# Datastream FX SQL
# -----------------------------------------------------------------------------
# trdstrm.ds2fxcode  — maps exrateintcode → (fromcurrcode, tocurrcode, ratetypecode)
# trdstrm.ds2fxrate  — daily midrate keyed by exrateintcode
#
# Convention: all fromcurrcode→USD SPOT rows store midrate as CCY per 1 USD.
# We invert to USD-per-CCY in fetch_from_datastream.
# -----------------------------------------------------------------------------
DATASTREAM_FX_SQL = """
    SELECT
        c.fromcurrcode  AS iso_currency,
        r.exratedate    AS marketdate,
        r.midrate       AS rate
    FROM trdstrm.ds2fxrate  AS r
    JOIN trdstrm.ds2fxcode  AS c ON r.exrateintcode = c.exrateintcode
    WHERE r.exratedate BETWEEN %(start_date)s AND %(end_date)s
      AND c.ratetypecode = 'SPOT'
      AND c.fromcurrcode = ANY(%(currencies)s)
      AND c.tocurrcode   = 'USD'
      AND r.midrate IS NOT NULL
      AND r.midrate > 0
    ORDER BY c.fromcurrcode, r.exratedate
"""


def require_wrds() -> "wrds.Connection":  # type: ignore[name-defined]
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
    else:
        log.info("WRDS_PASSWORD not set; using ~/.pgpass or interactive prompt")
    return wrds.Connection(**kwargs)


def fetch_from_datastream(
    currencies: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Query trdstrm.ds2fxrate / ds2fxcode for SPOT rates of each non-USD currency
    vs USD. Returns a long-form DataFrame normalised to USD-per-1-unit-of-base.

    All fromcurrcode→USD SPOT rows in ds2fxrate store the rate as CCY per 1 USD,
    so every value is inverted: mid_usd_per_base = 1 / midrate.
    """
    targets = [c for c in currencies if c != "USD"]
    if not targets:
        log.warning("no non-USD currencies to fetch from Datastream")
        return _empty_fx_frame()

    log.info("Datastream FX: fetching SPOT rates for %d currencies: %s",
             len(targets), sorted(targets))

    db = require_wrds()
    try:
        raw = db.raw_sql(
            DATASTREAM_FX_SQL,
            params={
                "start_date": start.isoformat(),
                "end_date":   end.isoformat(),
                "currencies": targets,
            },
            date_cols=["marketdate"],
        )
    finally:
        db.close()

    if raw.empty:
        log.warning("Datastream returned zero FX rows for requested currencies")
        return _empty_fx_frame()

    out_rows: list[dict] = []
    seen_currencies: set[str] = set()
    for _, row in raw.iterrows():
        iso = str(row["iso_currency"])
        rate = float(row["rate"])
        if rate <= 0:
            continue
        # All fromcurrcode→USD SPOT rates are CCY per 1 USD — invert for output.
        mid_usd_per_base = 1.0 / rate
        out_rows.append({
            "date":           pd.Timestamp(row["marketdate"]).date(),
            "base_currency":  iso,
            "quote_currency": "USD",
            "currency_pair":  f"{iso}_USD",
            "mid":            round(mid_usd_per_base, 10),
            "provider":       "datastream",
        })
        seen_currencies.add(iso)

    df = pd.DataFrame(out_rows)

    # Synthesise the USD identity series over the union of dates already present.
    if "USD" in currencies and not df.empty:
        all_dates = sorted(df["date"].unique())
        usd_rows = pd.DataFrame({
            "date":           all_dates,
            "base_currency":  "USD",
            "quote_currency": "USD",
            "currency_pair":  "USD_USD",
            "mid":            1.0,
            "provider":       "datastream",
        })
        df = pd.concat([df, usd_rows], ignore_index=True)

    missing = set(targets) - seen_currencies
    if missing:
        log.warning("Datastream returned NO rows for: %s", sorted(missing))

    log.info("Datastream FX rows: %d (covering %d currencies)",
             len(df), df["base_currency"].nunique())
    return df


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _empty_fx_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "date":           pd.Series(dtype="object"),
        "base_currency":  pd.Series(dtype="object"),
        "quote_currency": pd.Series(dtype="object"),
        "currency_pair":  pd.Series(dtype="object"),
        "mid":            pd.Series(dtype="float64"),
        "provider":       pd.Series(dtype="object"),
    })


def load_currencies_from_reference(path: Path) -> list[str]:
    if not path.exists():
        log.error("ADR reference not found at %s.\n"
                  "  Run fetch_datastream_adr_data.py first.", path)
        raise SystemExit(2)
    try:
        ref = pd.read_parquet(path)
    except Exception:
        try:
            import fastparquet as fp  # type: ignore
            ref = fp.ParquetFile(str(path)).to_pandas(columns=["underlying_currency"])
        except Exception as e:
            log.error("Failed to read ADR reference parquet: %s", e)
            raise SystemExit(2)
    ccys = sorted(
        c for c in set(ref["underlying_currency"].astype(str).tolist())
        if c != "None" and c not in EXCLUDED_CURRENCIES
    )
    if "USD" not in ccys:
        ccys.append("USD")
    log.info("currencies from ADR reference: %s", ccys)
    return ccys


def write_output(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        log.error("No FX rows to write — aborting.")
        raise SystemExit(3)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fx_rates.parquet"
    df = df.sort_values(["base_currency", "date"]).reset_index(drop=True)
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
        description="Fetch daily FX close rates (USD per 1 unit of base) "
                    "from WRDS Datastream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", type=_parse_date, default=start_date)
    p.add_argument("--end",   type=_parse_date, default=end_date)
    p.add_argument("--adr-reference", type=Path,
                   default=Path("datastream/data/parquet/adr/adr_reference.parquet"),
                   help="ADR reference Parquet (used to derive currency list)")
    p.add_argument("--currencies", nargs="*",
                   help="Explicit currency list; overrides --adr-reference")
    p.add_argument("--out", type=Path, default=Path("datastream/data/parquet/fx"))
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.start >= args.end:
        log.error("--start must be strictly before --end")
        return 2

    if args.currencies:
        currencies = list(args.currencies)
        if "USD" not in currencies:
            currencies.append("USD")
    else:
        currencies = load_currencies_from_reference(args.adr_reference)

    log.info("requested currencies: %s", currencies)

    df = fetch_from_datastream(currencies, args.start, args.end)
    write_output(df, args.out)
    log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
