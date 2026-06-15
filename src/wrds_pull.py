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

Run (default 2000-2023 → data/):
    python src/wrds_pull.py

Run an extended pull for the forward / out-of-sample test (Phase 4d), e.g. through 2025
into a separate directory so the validated 2003-2023 cache stays pristine:
    python src/wrds_pull.py --end 2025-12-31 --data-dir data_through_2025

One-time credential setup (creates ~/.pgpass):
    python -c "import wrds; wrds.Connection().create_pgpass_file()"

Design notes (why each piece matters):
  * Survivorship bias - we take index membership from crsp.dsp500list, which records
    every stock that was *ever* in the S&P 500 with its exact entry/exit dates.
  * Bid/ask - crsp.dsf carries the closing bid and ask; the spread *is* the cost.
  * prc can be negative when CRSP reports a bid-ask average; take abs(prc) downstream.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import wrds

from src.config import COMMON_SHARE_CODES, DATA_DIR, END_DATE, START_DATE


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #
def connect() -> wrds.Connection:
    """Open a WRDS connection (uses ~/.pgpass if configured, else prompts)."""
    return wrds.Connection()


# --------------------------------------------------------------------------- #
# Individual pulls (date range parametrised; defaults from config)
# --------------------------------------------------------------------------- #
def get_sp500_constituents(db, start=START_DATE, end=END_DATE) -> pd.DataFrame:
    """Point-in-time S&P 500 membership from crsp.dsp500list (survivorship-bias fix)."""
    return db.raw_sql(
        """
        SELECT permno, start, ending
        FROM crsp.dsp500list
        WHERE ending >= %(start)s AND start <= %(end)s
        """,
        params={"start": start, "end": end},
        date_cols=["start", "ending"],
    )


def get_crsp_daily(db, permnos, start=START_DATE, end=END_DATE) -> pd.DataFrame:
    """Daily CRSP stock file (crsp.dsf): price, bid/ask, returns, adjustment factors."""
    permno_list = ", ".join(str(int(p)) for p in sorted(set(permnos)))
    return db.raw_sql(
        f"""
        SELECT permno, date, prc, ret, retx, bid, ask, vol,
               shrout, cfacpr, cfacshr, openprc
        FROM crsp.dsf
        WHERE permno IN ({permno_list})
          AND date BETWEEN %(start)s AND %(end)s
        """,
        params={"start": start, "end": end},
        date_cols=["date"],
    )


def get_crsp_names(db, permnos) -> pd.DataFrame:
    """Historical identifiers/classification from crsp.dsenames (time-valid intervals)."""
    permno_list = ", ".join(str(int(p)) for p in sorted(set(permnos)))
    return db.raw_sql(
        f"""
        SELECT permno, namedt, nameendt, ticker, comnam, shrcd, exchcd, siccd
        FROM crsp.dsenames
        WHERE permno IN ({permno_list})
        """,
        date_cols=["namedt", "nameendt"],
    )


def get_delisting(db, permnos) -> pd.DataFrame:
    """Delisting events from crsp.dsedelist - date, reason code, delisting return."""
    permno_list = ", ".join(str(int(p)) for p in sorted(set(permnos)))
    return db.raw_sql(
        f"""
        SELECT permno, dlstdt, dlstcd, dlret
        FROM crsp.dsedelist
        WHERE permno IN ({permno_list})
        """,
        date_cols=["dlstdt"],
    )


def get_sp500_index(db, start=START_DATE, end=END_DATE) -> pd.DataFrame:
    """Daily S&P 500 index level and return from crsp.dsp500 (market control)."""
    return db.raw_sql(
        """
        SELECT caldt, spindx, sprtrn
        FROM crsp.dsp500
        WHERE caldt BETWEEN %(start)s AND %(end)s
        """,
        params={"start": start, "end": end},
        date_cols=["caldt"],
    )


def get_ff_factors(db, start=START_DATE, end=END_DATE) -> pd.DataFrame:
    """Fama-French 5 factors + momentum, daily, from the WRDS `ff` library."""
    ff5 = db.raw_sql(
        """
        SELECT date, mktrf, smb, hml, rmw, cma, rf
        FROM ff.fivefactors_daily
        WHERE date BETWEEN %(start)s AND %(end)s
        """,
        params={"start": start, "end": end},
        date_cols=["date"],
    )
    umd = db.raw_sql(
        """
        SELECT date, umd
        FROM ff.factors_daily
        WHERE date BETWEEN %(start)s AND %(end)s
        """,
        params={"start": start, "end": end},
        date_cols=["date"],
    )
    return ff5.merge(umd, on="date", how="left")


# --------------------------------------------------------------------------- #
# CIZ / v2 pulls (current data, incl. 2025) — map back to the legacy schema
# --------------------------------------------------------------------------- #
# CIZ common-stock filter = the official equivalent of legacy shrcd 10/11.
_CIZ_COMMON = (
    "securitytype = 'EQTY' AND securitysubtype = 'COM' AND sharetype = 'NS' "
    "AND usincflg = 'Y' AND issuertype IN ('ACOR','CORP')"
)
# S&P 500 index family in the v2 constituents table (confirmed via probe).
_SP500_INDFAM = 1100500
_CIZ_NUMERIC = ["prc", "ret", "retx", "bid", "ask", "vol", "shrout",
                "cfacpr", "cfacshr", "openprc"]


def get_sp500_constituents_v2(db, start=START_DATE, end=END_DATE) -> pd.DataFrame:
    """S&P 500 membership from the CIZ table (crsp.dsp500list_v2)."""
    return db.raw_sql(
        f"""
        SELECT permno, mbrstartdt AS start, mbrenddt AS ending
        FROM crsp.dsp500list_v2
        WHERE indfam = {_SP500_INDFAM}
          AND mbrenddt >= %(start)s AND mbrstartdt <= %(end)s
        """,
        params={"start": start, "end": end},
        date_cols=["start", "ending"],
    )


def get_crsp_daily_v2(db, permnos, start=START_DATE, end=END_DATE) -> pd.DataFrame:
    """Daily stock data from the CIZ convenience view (crsp.wrds_dsfv2_query).

    This single view carries prices, bid/ask, returns, the cumulative adjustment
    factor, and the time-valid siccd/ticker/classification — so no separate name
    join is needed. CIZ columns are aliased back to the legacy schema; a sentinel
    shrcd=11 is added (the CIZ common-stock filter is applied in SQL) so downstream
    code is unchanged.
    """
    permno_list = ", ".join(str(int(p)) for p in sorted(set(permnos)))
    daily = db.raw_sql(
        f"""
        SELECT permno, dlycaldt AS date, dlyprc AS prc, dlyret AS ret,
               dlyretx AS retx, dlybid AS bid, dlyask AS ask, dlyvol AS vol,
               shrout, dlycumfacpr AS cfacpr, dlycumfacshr AS cfacshr,
               dlyopen AS openprc, siccd, ticker, securitynm AS comnam,
               primaryexch
        FROM crsp.wrds_dsfv2_query
        WHERE permno IN ({permno_list})
          AND dlycaldt BETWEEN %(start)s AND %(end)s
          AND {_CIZ_COMMON}
        """,
        params={"start": start, "end": end},
        date_cols=["date"],
    )
    for c in _CIZ_NUMERIC:
        daily[c] = pd.to_numeric(daily[c], errors="coerce")
    daily["shrcd"] = 11           # sentinel: CIZ filter already applied in SQL
    daily["exchcd"] = 0           # legacy field, unused downstream
    return daily


def get_sp500_index_v2(db, start=START_DATE, end=END_DATE) -> pd.DataFrame:
    """Daily S&P 500 index from the CIZ table (crsp.dsp500_v2); legacy column names."""
    return db.raw_sql(
        """
        SELECT caldt, spindx, sprtrn
        FROM crsp.dsp500_v2
        WHERE caldt BETWEEN %(start)s AND %(end)s
        """,
        params={"start": start, "end": end},
        date_cols=["caldt"],
    )


def get_delisting_v2(db, permnos, start=START_DATE, end=END_DATE) -> pd.DataFrame:
    """Delisting events from CIZ daily data (dlydelflg flags a delisting day).

    CIZ integrates the delisting return into dlyret on the delisting day, so we take
    that as dlret; dlstcd is a sentinel (only used as a fallback when dlret is NaN).
    """
    permno_list = ", ".join(str(int(p)) for p in sorted(set(permnos)))
    dl = db.raw_sql(
        f"""
        SELECT permno, dlycaldt AS dlstdt, dlyret AS dlret
        FROM crsp.stkdlysecuritydata
        WHERE permno IN ({permno_list})
          AND dlydelflg <> 'N'
          AND dlycaldt BETWEEN %(start)s AND %(end)s
        """,
        params={"start": start, "end": end},
        date_cols=["dlstdt"],
    )
    dl["dlret"] = pd.to_numeric(dl["dlret"], errors="coerce")
    dl["dlstcd"] = 500            # sentinel; real return comes from dlret
    return dl


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #
def attach_classification(daily: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    """Attach the time-valid shrcd/exchcd/siccd/ticker to each daily row (as-of join)."""
    daily = daily.sort_values("date")
    names = names.sort_values("namedt")
    merged = pd.merge_asof(
        daily, names, left_on="date", right_on="namedt", by="permno",
        direction="backward",
    )
    return merged[merged["date"] <= merged["nameendt"]].copy()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main(start: str = START_DATE, end: str = END_DATE, data_dir: Path = DATA_DIR,
         source: str = "legacy") -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pull window {start} → {end}  |  source {source}  |  output {data_dir}")
    db = connect()
    try:
        if source == "v2":
            # CIZ / v2 tables — current through 2025. Schema mapped to legacy.
            print("[1/5] S&P 500 constituents (v2) ...")
            constituents = get_sp500_constituents_v2(db, start, end)
            constituents.to_parquet(data_dir / "sp500_constituents.parquet")
            permnos = constituents["permno"].unique()
            print(f"      {len(permnos)} unique permnos ever in the index")

            print("[2/5] CRSP daily file (v2 convenience view) ...")
            daily = get_crsp_daily_v2(db, permnos, start, end)
            daily.to_parquet(data_dir / "crsp_daily.parquet")
            print(f"      {len(daily):,} rows (CIZ common-stock filter applied in SQL)")

            print("[3/5] Delisting events (v2) ...")
            get_delisting_v2(db, permnos, start, end).to_parquet(
                data_dir / "delisting.parquet")

            print("[4/5] S&P 500 index (v2) ...")
            get_sp500_index_v2(db, start, end).to_parquet(
                data_dir / "sp500_index.parquet")
        else:
            print("[1/5] S&P 500 constituents ...")
            constituents = get_sp500_constituents(db, start, end)
            constituents.to_parquet(data_dir / "sp500_constituents.parquet")
            permnos = constituents["permno"].unique()
            print(f"      {len(permnos)} unique permnos ever in the index")

            print("[2/5] CRSP daily file (this is the big pull) ...")
            daily = get_crsp_daily(db, permnos, start, end)
            names = get_crsp_names(db, permnos)
            daily = attach_classification(daily, names)
            before = len(daily)
            daily = daily[daily["shrcd"].isin(COMMON_SHARE_CODES)]
            daily.to_parquet(data_dir / "crsp_daily.parquet")
            print(f"      {before:,} rows -> {len(daily):,} after common-share filter")

            print("[3/5] Delisting events ...")
            get_delisting(db, permnos).to_parquet(data_dir / "delisting.parquet")

            print("[4/5] S&P 500 index ...")
            get_sp500_index(db, start, end).to_parquet(data_dir / "sp500_index.parquet")

        print("[5/5] Fama-French factors ...")
        get_ff_factors(db, start, end).to_parquet(data_dir / "ff_factors.parquet")
    finally:
        db.close()

    n_days = daily["date"].nunique()
    n_months = daily["date"].dt.to_period("M").nunique()
    print("\n--- Pull complete ---")
    print(f"  trading days : {n_days:,}")
    print(f"  months       : {n_months}")
    print(f"  stocks       : {daily['permno'].nunique():,}")
    print(f"  cached to    : {data_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pull the CRSP/FF data spine from WRDS.")
    ap.add_argument("--start", default=START_DATE, help="pull start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=END_DATE, help="pull end date (YYYY-MM-DD)")
    ap.add_argument("--data-dir", default=str(DATA_DIR),
                    help="output directory for the parquet cache")
    ap.add_argument("--source", default="legacy", choices=["legacy", "v2"],
                    help="legacy = crsp.dsf (through 2024); v2 = CIZ tables (through 2025)")
    args = ap.parse_args()
    main(start=args.start, end=args.end, data_dir=Path(args.data_dir), source=args.source)
