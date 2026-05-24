"""
Phase 0 - Data spine.

Pulls the survivorship-bias-free dataset for the Rotondi & Russo (2025) pairs-trading
replication from WRDS and caches it as parquet files in data/.

Outputs (data/):
    sp500_constituents.parquet  point-in-time S&P 500 membership (permno, start, end)
    crsp_daily.parquet          daily prices, bid/ask, returns, codes for all members
    delisting.parquet           delisting dates, reason codes, delisting returns
    sp500_index.parquet         daily S&P 500 index level + return
    ff_factors.parquet          Fama-French 5 factors + momentum (+ risk-free rate)

Run:
    python src/wrds_pull.py

One-time credential setup (creates ~/.pgpass):
    python -c "import wrds; wrds.Connection().create_pgpass_file()"

Design notes (why each piece matters):
  * Survivorship bias - we take index membership from crsp.dsp500list, which records
    every stock that was *ever* in the S&P 500 with its exact entry/exit dates. Using
    today's members would silently delete every firm that went bankrupt or was
    dropped, inflating the backtest.
  * Bid/ask - crsp.dsf carries the closing bid and ask. The paper marks PnL at
    bid/ask, so the spread *is* the transaction cost; no separate commission model.
  * Adjustment - cfacpr is CRSP's cumulative price-adjustment factor; adjusted price
    = prc / cfacpr gives a continuous series across splits/dividends.
  * prc can be negative when CRSP has no closing trade and reports a bid-ask average;
    downstream code should take abs(prc).
"""
from __future__ import annotations

import pandas as pd
import wrds

from config import COMMON_SHARE_CODES, DATA_DIR, END_DATE, START_DATE


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #
def connect() -> wrds.Connection:
    """Open a WRDS connection (uses ~/.pgpass if configured, else prompts)."""
    return wrds.Connection()


# --------------------------------------------------------------------------- #
# Individual pulls
# --------------------------------------------------------------------------- #
def get_sp500_constituents(db: wrds.Connection) -> pd.DataFrame:
    """Point-in-time S&P 500 membership from crsp.dsp500list.

    Each row: a permno with the date range it was an index constituent. This is
    the survivorship-bias fix.
    """
    return db.raw_sql(
        """
        SELECT permno, start, ending
        FROM crsp.dsp500list
        WHERE ending >= %(start)s AND start <= %(end)s
        """,
        params={"start": START_DATE, "end": END_DATE},
        date_cols=["start", "ending"],
    )


def get_crsp_daily(db: wrds.Connection, permnos) -> pd.DataFrame:
    """Daily CRSP stock file (crsp.dsf) for the given permnos over the sample.

    Pulls price, bid/ask, returns, volume, shares outstanding, and the cumulative
    price-adjustment factor cfacpr.
    """
    permno_list = ", ".join(str(int(p)) for p in sorted(set(permnos)))
    return db.raw_sql(
        f"""
        SELECT permno, date, prc, ret, retx, bid, ask, vol,
               shrout, cfacpr, cfacshr, openprc
        FROM crsp.dsf
        WHERE permno IN ({permno_list})
          AND date BETWEEN %(start)s AND %(end)s
        """,
        params={"start": START_DATE, "end": END_DATE},
        date_cols=["date"],
    )


def get_crsp_names(db: wrds.Connection, permnos) -> pd.DataFrame:
    """Historical identifiers/classification from crsp.dsenames.

    shrcd (share code), exchcd (exchange) and siccd (industry) change over time,
    so each row is valid only between namedt and nameendt.
    """
    permno_list = ", ".join(str(int(p)) for p in sorted(set(permnos)))
    return db.raw_sql(
        f"""
        SELECT permno, namedt, nameendt, ticker, comnam, shrcd, exchcd, siccd
        FROM crsp.dsenames
        WHERE permno IN ({permno_list})
        """,
        date_cols=["namedt", "nameendt"],
    )


def get_delisting(db: wrds.Connection, permnos) -> pd.DataFrame:
    """Delisting events from crsp.dsedelist - date, reason code, delisting return.

    The delisting return (dlret) is the often-large, often-negative return realised
    when a stock leaves the exchange. A pair holding a delisted leg must book it.
    """
    permno_list = ", ".join(str(int(p)) for p in sorted(set(permnos)))
    return db.raw_sql(
        f"""
        SELECT permno, dlstdt, dlstcd, dlret
        FROM crsp.dsedelist
        WHERE permno IN ({permno_list})
        """,
        date_cols=["dlstdt"],
    )


def get_sp500_index(db: wrds.Connection) -> pd.DataFrame:
    """Daily S&P 500 index level and return from crsp.dsp500.

    Needed for the partial-correlation distance metric (market control) and the
    buy-and-hold benchmark.
    """
    return db.raw_sql(
        """
        SELECT caldt, spindx, sprtrn
        FROM crsp.dsp500
        WHERE caldt BETWEEN %(start)s AND %(end)s
        """,
        params={"start": START_DATE, "end": END_DATE},
        date_cols=["caldt"],
    )


def get_ff_factors(db: wrds.Connection) -> pd.DataFrame:
    """Fama-French 5 factors + momentum, daily, from the WRDS `ff` library.

    Used for the economic-significance regressions (paper Section 4.1.2) and as the
    factor set for the Phase 3 factor-beta extension.
    """
    ff5 = db.raw_sql(
        """
        SELECT date, mktrf, smb, hml, rmw, cma, rf
        FROM ff.fivefactors_daily
        WHERE date BETWEEN %(start)s AND %(end)s
        """,
        params={"start": START_DATE, "end": END_DATE},
        date_cols=["date"],
    )
    umd = db.raw_sql(
        """
        SELECT date, umd
        FROM ff.factors_daily
        WHERE date BETWEEN %(start)s AND %(end)s
        """,
        params={"start": START_DATE, "end": END_DATE},
        date_cols=["date"],
    )
    return ff5.merge(umd, on="date", how="left")


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #
def attach_classification(daily: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    """Attach the time-valid shrcd/exchcd/siccd/ticker to each daily row.

    Uses an as-of join on namedt (most recent name record on or before the trade
    date), then drops rows past that record's nameendt - an interval join.
    """
    daily = daily.sort_values("date")
    names = names.sort_values("namedt")
    merged = pd.merge_asof(
        daily,
        names,
        left_on="date",
        right_on="namedt",
        by="permno",
        direction="backward",
    )
    # keep only rows where the name record is still valid on the trade date
    return merged[merged["date"] <= merged["nameendt"]].copy()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = connect()
    try:
        print("[1/5] S&P 500 constituents ...")
        constituents = get_sp500_constituents(db)
        constituents.to_parquet(DATA_DIR / "sp500_constituents.parquet")
        permnos = constituents["permno"].unique()
        print(f"      {len(permnos)} unique permnos ever in the index")

        print("[2/5] CRSP daily file (this is the big pull) ...")
        daily = get_crsp_daily(db, permnos)
        names = get_crsp_names(db, permnos)
        daily = attach_classification(daily, names)
        before = len(daily)
        daily = daily[daily["shrcd"].isin(COMMON_SHARE_CODES)]
        daily.to_parquet(DATA_DIR / "crsp_daily.parquet")
        print(f"      {before:,} rows -> {len(daily):,} after common-share filter")

        print("[3/5] Delisting events ...")
        get_delisting(db, permnos).to_parquet(DATA_DIR / "delisting.parquet")

        print("[4/5] S&P 500 index ...")
        get_sp500_index(db).to_parquet(DATA_DIR / "sp500_index.parquet")

        print("[5/5] Fama-French factors ...")
        get_ff_factors(db).to_parquet(DATA_DIR / "ff_factors.parquet")
    finally:
        db.close()

    # --- Phase 0 checkpoint (plan target: ~6,039 trading days, 288 months) ---
    n_days = daily["date"].nunique()
    n_months = daily["date"].dt.to_period("M").nunique()
    print("\n--- Phase 0 checkpoint ---")
    print(f"  trading days : {n_days:,}   (paper: 6,039)")
    print(f"  months       : {n_months}   (paper: 288)")
    print(f"  stocks       : {daily['permno'].nunique():,}   (paper: 1,098)")
    print(f"  cached to    : {DATA_DIR}")


if __name__ == "__main__":
    main()
