"""
01_fetch_data.py
────────────────
Pulls from WRDS/CRSP and saves locally as parquet:
  1. Daily returns for S&P 500 stocks (survivorship-bias-free)
  2. Daily returns for all factor ETFs defined in config.py
  3. Stock name/metadata table (permno → ticker, SIC, exchange)

Run once. All downstream scripts read from data/raw/.
Runtime: ~5–10 minutes.
"""

import wrds
import pandas as pd
from tqdm import tqdm
from config import (
    WRDS_USERNAME, START_DATE, END_DATE,
    FF5_FACTORS, MIN_DOLLAR_VOLUME,
    DATA_RAW,
)


def connect() -> wrds.Connection:
    print("Connecting to WRDS…")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)
    print("Connected.\n")
    return db


# ── 1. S&P 500 constituent permnos ────────────────────────────────────────────
def fetch_sp500_permnos(db: wrds.Connection) -> list:
    """
    Uses CRSP dsp500list — records every stock's entry/exit date in the S&P 500.
    Pulling all permnos active at any point in [START_DATE, END_DATE] avoids
    survivorship bias: we include stocks that were later removed or delisted.
    """
    print("Fetching S&P 500 constituent history…")
    q = f"""
        SELECT DISTINCT permno
        FROM crsp.dsp500list
        WHERE start  <= '{END_DATE}'
          AND ending >= '{START_DATE}'
    """
    permnos = db.raw_sql(q)["permno"].astype(int).tolist()
    print(f"  → {len(permnos):,} unique permnos ever in S&P 500 during sample.\n")
    return permnos


# ── 2. Daily stock returns ─────────────────────────────────────────────────────
def fetch_stock_returns(db: wrds.Connection, permnos: list) -> pd.DataFrame:
    """
    Pulls ret, prc, shrout, vol from crsp.dsf in batches of 500.

    Column notes:
      ret     — daily return incl. dividends/splits; use this for P&L, not prc diff
      prc     — closing price; negative = CRSP bid-ask midpoint convention → abs()
      shrout  — shares outstanding in thousands → multiply by 1000
      vol     — daily share volume
    """
    print("Fetching daily stock returns from crsp.dsf (batched)…")
    batch_size = 500
    batches    = [permnos[i:i+batch_size] for i in range(0, len(permnos), batch_size)]
    frames     = []

    for batch in tqdm(batches, desc="  Batch"):
        ids = ",".join(str(p) for p in batch)
        q   = f"""
            SELECT permno, date, ret, prc, shrout, vol
            FROM crsp.dsf
            WHERE permno IN ({ids})
              AND date BETWEEN '{START_DATE}' AND '{END_DATE}'
        """
        frames.append(db.raw_sql(q, date_cols=["date"]))

    stocks = pd.concat(frames, ignore_index=True)
    stocks["date"]      = pd.to_datetime(stocks["date"])
    stocks["prc"]       = stocks["prc"].abs()
    stocks["shrout"]    = stocks["shrout"] * 1_000
    stocks["dollar_vol"]= stocks["prc"] * stocks["vol"].fillna(0)

    # Drop missing returns (keep -1 delisting returns — they are real losses)
    before = len(stocks)
    stocks  = stocks.dropna(subset=["ret"])
    print(f"  → Dropped {before - len(stocks):,} rows with NaN returns.")

    # Liquidity filter: require minimum average daily dollar volume
    avg_dv         = stocks.groupby("permno")["dollar_vol"].mean()
    liquid         = avg_dv[avg_dv >= MIN_DOLLAR_VOLUME].index
    stocks         = stocks[stocks["permno"].isin(liquid)]
    print(f"  → {stocks['permno'].nunique()} permnos pass liquidity filter "
          f"(${MIN_DOLLAR_VOLUME:,.0f}/day avg).")
    print(f"  → Total rows kept: {len(stocks):,}\n")

    # Pivot to wide: index=date, columns=permno, values=ret
    ret_wide = (
        stocks
        .pivot(index="date", columns="permno", values="ret")
        .sort_index()
    )
    ret_wide.columns = ret_wide.columns.astype(int)

    out = DATA_RAW / "stock_returns.parquet"
    ret_wide.to_parquet(out)
    print(f"Saved stock returns  →  {out}\n")
    return ret_wide


# ── 3. Stock metadata ─────────────────────────────────────────────────────────
def fetch_stock_names(db: wrds.Connection, permnos: list) -> pd.DataFrame:
    """
    Saves permno → ticker, company name, SIC code, exchange.
    Used for labelling pairs in results.
    """
    ids = ",".join(str(p) for p in permnos)
    q   = f"""
        SELECT permno, ticker, comnam, siccd, exchcd, namedt, nameendt
        FROM crsp.msenames
        WHERE permno IN ({ids})
    """
    names = db.raw_sql(q, date_cols=["namedt", "nameendt"])
    # Keep most recent record per permno
    names = (
        names
        .sort_values("namedt")
        .groupby("permno")
        .last()
        .reset_index()
    )
    out = DATA_RAW / "stock_names.parquet"
    names.to_parquet(out, index=False)
    print(f"Saved stock names    →  {out}\n")
    return names


# ── 4. Fama-French 5 + Momentum factor returns ────────────────────────────────
def fetch_factor_returns(db: wrds.Connection) -> pd.DataFrame:
    """
    Pulls Fama-French 5 factors + momentum (UMD) from the WRDS ff library.

    These 6 factors are orthogonal by construction, unlike the sector ETFs
    (XLF/XLK/SPY) which are highly correlated and made Ridge betas noisy.

    Sources:
      ff.fivefactors_daily  →  mktrf, smb, hml, rmw, cma
      ff.factors_daily      →  umd (momentum)
    """
    print("Fetching Fama-French 5 + momentum factors from WRDS ff library…")

    ff5 = db.raw_sql(
        f"""
        SELECT date, mktrf, smb, hml, rmw, cma
        FROM ff.fivefactors_daily
        WHERE date BETWEEN '{START_DATE}' AND '{END_DATE}'
        """,
        date_cols=["date"],
    )

    umd = db.raw_sql(
        f"""
        SELECT date, umd
        FROM ff.factors_daily
        WHERE date BETWEEN '{START_DATE}' AND '{END_DATE}'
        """,
        date_cols=["date"],
    )

    merged = ff5.merge(umd, on="date", how="left")
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values("date").set_index("date")

    # Rename columns to descriptive names matching FACTOR_NAMES in config
    merged.columns = list(FF5_FACTORS.values())   # Market, Size, Value, Profitability, Investment, Momentum

    out = DATA_RAW / "factor_returns.parquet"
    merged.to_parquet(out)
    print(f"  → FF5 factor matrix: {merged.shape[0]} days × {merged.shape[1]} factors")
    print(f"Saved factor returns →  {out}\n")
    return merged


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    db = connect()

    permnos     = fetch_sp500_permnos(db)
    stock_rets  = fetch_stock_returns(db, permnos)
    _           = fetch_stock_names(db, permnos)
    factor_rets = fetch_factor_returns(db)

    db.close()

    print("=" * 60)
    print("01_fetch_data.py complete.")
    print(f"  Stocks : {stock_rets.shape[1]:,} permnos  ×  {stock_rets.shape[0]:,} days")
    print(f"  Factors: {factor_rets.shape[1]:,} factors ×  {factor_rets.shape[0]:,} days")
    print(f"\nNext: python 02_rolling_betas.py")
