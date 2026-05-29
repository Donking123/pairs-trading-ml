"""
data_fetcher.py — Unified data access for the pairs-trading sandbox.

Goals
- Prefer robust sources (WRDS REST API, FRED, yfinance) when available
- Fall back gracefully; persist everything to ./data as parquet

Sources
- FF factors + Momentum: WRDS ff.factors_daily (token auth, no SSH/credentials)
- CRSP prices, membership, earnings: WRDS REST API
- FRED macro series: public CSV endpoint (no API key)
- yfinance: fallback for equities and factor ETFs

WRDS token: pass wrds_token= or set env var WRDS_API_TOKEN.
No username/password prompts.

Usage
    from data_fetcher import DataFetcher
    df = DataFetcher(wrds_token="your_token")
    stock_close, stock_volume, stock_returns, factor_close, factor_returns = df.load_all()
"""

from __future__ import annotations

import io
import os
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

_WRDS_BASE = "https://wrds-api.wharton.upenn.edu/data/"

# FRED series → frb.rates_daily column mapping.
# None = derived (see fetch_fred).
_FRB_MAP: dict = {
    "DGS2":        "dgs2",
    "DGS10":       "dgs10",
    "T10Y2Y":      None,             # derived: dgs10 - dgs2
    "BAMLH0A0HYM2": "bamlh0a0hym2", # ICE BofA HY OAS — exact match in WRDS
    "BAMLC0A0CM":  None,             # IG OAS not in WRDS; proxy: bamlc0a0cmey - dgs10
}

# FRED only distributes ICE BofA series from this date (licensing restriction).
# Pre-2023 BAML history must come from WRDS frb.rates_daily.
_FRED_BAML_START = "2023-05-23"
_FRED_BAML_SERIES = {"BAMLH0A0HYM2", "BAMLC0A0CM"}


@dataclass
class DataFetcher:
    data_dir: Path = Path(__file__).resolve().parent / "data"
    allow_network: bool = True
    wrds_token: str = ""

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.wrds_token:
            self.wrds_token = os.environ.get("WRDS_API_TOKEN", "")
        # Cache paths
        self.spx_close_pq   = self.data_dir / "spx_close.parquet"
        self.spx_vol_pq     = self.data_dir / "spx_volume.parquet"
        self.factor_close_pq = self.data_dir / "factor_close.parquet"
        self.ff_daily_pq    = self.data_dir / "ff_daily.parquet"
        self.fred_pq        = self.data_dir / "fred.parquet"

    # ── WRDS REST API helper ──────────────────────────────────────────────────

    def _wrds_fetch(
        self,
        table: str,
        params: Optional[dict] = None,
        date_cols: Optional[List[str]] = None,
        page_size: int = 5000,
    ) -> pd.DataFrame:
        """Fetch all pages from a WRDS REST API table.

        Handles pagination automatically. Disables SSL verification because
        the WRDS API endpoint uses a certificate that fails validation in some
        environments (wrds-api.wharton.upenn.edu).

        Args:
            table: WRDS library.table name, e.g. "crsp.dsf"
            params: Query filters, e.g. {"date__gte": "2014-01-01"}
            date_cols: Column names to parse as dates
            page_size: Records per request (max ~10000)
        """
        import requests

        if not self.wrds_token:
            raise RuntimeError(
                "WRDS token required. Pass wrds_token= or set WRDS_API_TOKEN env var."
            )
        headers = {
            "Authorization": f"Token {self.wrds_token}",
            "Accept": "application/json",
        }
        p = dict(params or {})
        p["limit"] = page_size

        url: Optional[str] = _WRDS_BASE + table + "/"
        frames = []
        page = 0
        while url:
            r = requests.get(
                url,
                params=p if page == 0 else None,
                headers=headers,
                verify=False,
                timeout=(10, 120),
            )
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            if results:
                frames.append(pd.DataFrame(results))
            url = data.get("next")
            page += 1

        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        for col in date_cols or []:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    # ── FF factors via WRDS (primary) + Ken French (fallback) ─────────────────

    def fetch_ff_daily(self, refresh: bool = False) -> pd.DataFrame:
        """FF5 + Momentum daily factors.

        Primary: WRDS ff.factors_daily (requires token, values already in decimal).
        Fallback: Ken French website zip download.
        """
        if self.ff_daily_pq.exists() and not refresh:
            return pd.read_parquet(self.ff_daily_pq)
        if not self.allow_network:
            raise FileNotFoundError("ff_daily.parquet missing and network disabled")

        ff = None

        # --- Primary: WRDS REST API ---
        if self.wrds_token:
            try:
                raw = self._wrds_fetch(
                    "ff.factors_daily",
                    params={"date__gte": "1990-01-01"},
                    date_cols=["date"],
                )
                if not raw.empty:
                    ff = raw.set_index("date").sort_index()
                    # WRDS columns: mktrf, smb, hml, rf, umd (umd = momentum)
                    ff = ff.rename(columns={
                        "mktrf": "FF_MktRF", "smb": "FF_SMB", "hml": "FF_HML",
                        "rf": "FF_RF",       "umd": "MOM",
                    })
                    for col in ff.columns:
                        ff[col] = pd.to_numeric(ff[col], errors="coerce")
            except Exception as e:
                warnings.warn(f"WRDS ff.factors_daily failed ({e}), falling back to Ken French")
                ff = None

        # --- Fallback: Ken French website ---
        if ff is None:
            ff = self._fetch_ff_kf()

        ff.to_parquet(self.ff_daily_pq)
        return ff

    def _fetch_ff_kf(self) -> pd.DataFrame:
        """Download FF5 + Momentum from Ken French's website."""
        import requests

        def _parse(text: str) -> pd.DataFrame:
            lines = text.splitlines()
            data_start = next(
                (i for i, ln in enumerate(lines)
                 if len(ln.split(",")[0].strip()) == 8 and ln.split(",")[0].strip().isdigit()),
                None,
            )
            if data_start is None:
                raise ValueError("No data rows in FF CSV")
            header_idx = data_start - 1
            while header_idx >= 0 and not lines[header_idx].strip():
                header_idx -= 1
            data_lines = []
            for ln in lines[data_start:]:
                first = ln.split(",")[0].strip()
                if not first or not (first.isdigit() and len(first) == 8):
                    break
                data_lines.append(ln)
            csv = lines[header_idx] + "\n" + "\n".join(data_lines)
            df = pd.read_csv(io.StringIO(csv))
            df.columns = [c.strip() for c in df.columns]
            df = df.rename(columns={df.columns[0]: "date"})
            df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m%d")
            for col in df.columns[1:]:
                df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce") / 100.0
            return df.set_index("date").sort_index()

        url5 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        urlm = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
        r5 = __import__("requests").get(url5, timeout=60)
        with zipfile.ZipFile(io.BytesIO(r5.content)).open(
            zipfile.ZipFile(io.BytesIO(r5.content)).namelist()[0]
        ) as f:
            df5 = _parse(f.read().decode("latin1"))

        rm = __import__("requests").get(urlm, timeout=60)
        with zipfile.ZipFile(io.BytesIO(rm.content)).open(
            zipfile.ZipFile(io.BytesIO(rm.content)).namelist()[0]
        ) as f:
            dfm = _parse(f.read().decode("latin1"))

        mom_col = next((c for c in dfm.columns if "mom" in c.lower()), dfm.columns[0])
        dfm = dfm.rename(columns={mom_col: "MOM"})
        ff = df5.join(dfm[["MOM"]], how="left")
        return ff.rename(columns={
            "Mkt-RF": "FF_MktRF", "SMB": "FF_SMB", "HML": "FF_HML",
            "RMW": "FF_RMW",     "CMA": "FF_CMA",  "RF":  "FF_RF",
        })

    # ── Macro rates: FRED API (primary) → WRDS frb.rates_daily (fallback) ───────

    def fetch_fred(
        self,
        series: Iterable[str],
        refresh: bool = False,
        start: str = "2014-01-01",
        fred_api_key: str = "",
    ) -> pd.DataFrame:
        """Fetch macro rate series via FRED API (golden source, current to today).

        Primary: FRED REST API (api.stlouisfed.org) — requires free API key.
                 Returns exact OAS values; no proxy needed. Current to today.
        Fallback: WRDS frb.rates_daily (stale ~Feb 2025) for pre-2023 BAML history.

        ICE BofA licensing: FRED only distributes BAML series from 2023-05-23.
        Pre-2023 BAML data comes from WRDS frb.rates_daily (bamlh0a0hym2 = exact HY OAS;
        BAMLC0A0CM proxied as bamlc0a0cmey − dgs10, overestimates by ~20bp on average).

        Correct series IDs: BAMLC0A0CM (IG OAS), BAMLH0A0HYM2 (HY OAS).
        Note: BAMLC0A0CM (with extra C) does not exist on FRED.

        Args:
            series:       FRED series ids e.g. ["DGS2","DGS10","BAMLC0A0CM"]
            refresh:      Re-download even if cached
            start:        Observation start date (default 2014-01-01)
            fred_api_key: Free key from fred.stlouisfed.org/docs/api/api_key.html
                          Also read from env var FRED_API_KEY if not passed.
        """
        if self.fred_pq.exists() and not refresh:
            return pd.read_parquet(self.fred_pq)
        if not self.allow_network:
            raise FileNotFoundError("fred.parquet missing and network disabled")

        import requests

        api_key = fred_api_key or os.environ.get("FRED_API_KEY", "")
        series_list = list(series)
        frames: dict[str, pd.Series] = {}

        # --- Step 1: FRED REST API (exact values, current to today) ---
        # BAML series only available from _FRED_BAML_START; others from `start`.
        if api_key:
            for sid in series_list:
                obs_start = _FRED_BAML_START if sid in _FRED_BAML_SERIES else start
                try:
                    r = requests.get(
                        "https://api.stlouisfed.org/fred/series/observations",
                        params={
                            "series_id":         sid,
                            "api_key":           api_key,
                            "file_type":         "json",
                            "observation_start": obs_start,
                        },
                        timeout=30,
                    )
                    r.raise_for_status()
                    obs = r.json().get("observations", [])
                    if obs:
                        df = pd.DataFrame(obs)[["date", "value"]].rename(columns={"value": sid})
                        df["date"] = pd.to_datetime(df["date"])
                        df[sid] = pd.to_numeric(df[sid], errors="coerce")
                        frames[sid] = df.set_index("date")[sid]
                except Exception as e:
                    warnings.warn(f"FRED API {sid} failed: {e}")

        # --- Step 2: WRDS frb.rates_daily — fills pre-2023 BAML + any gaps ---
        # For BAML series: splice WRDS (start → _FRED_BAML_START) with FRED (after).
        # For DGS/T10Y2Y: only use WRDS if FRED failed entirely.
        need_wrds = [s for s in series_list if s in _FRB_MAP and (
            s in _FRED_BAML_SERIES or s not in frames
        )]
        if need_wrds and self.wrds_token:
            try:
                raw = self._wrds_fetch(
                    "frb.rates_daily",
                    params={"date__gte": start},
                    date_cols=["date"],
                ).set_index("date").sort_index()
                for sid in need_wrds:
                    col = _FRB_MAP[sid]
                    if sid == "T10Y2Y":
                        s = pd.to_numeric(raw["dgs10"], errors="coerce") \
                          - pd.to_numeric(raw["dgs2"],  errors="coerce")
                    elif sid == "BAMLC0A0CM":
                        # IG yield spread as pre-2023 proxy (~20bp high vs true OAS)
                        s = pd.to_numeric(raw["bamlc0a0cmey"], errors="coerce") \
                          - pd.to_numeric(raw["dgs10"],        errors="coerce")
                    else:
                        s = pd.to_numeric(raw[col], errors="coerce")
                    s = s.rename(sid)

                    if sid in _FRED_BAML_SERIES and sid in frames:
                        # Splice: WRDS history + FRED from _FRED_BAML_START onward
                        wrds_part = s[s.index < _FRED_BAML_START]
                        frames[sid] = pd.concat([wrds_part, frames[sid]]).sort_index()
                    elif sid not in frames:
                        frames[sid] = s
            except Exception as e:
                warnings.warn(f"WRDS frb.rates_daily failed: {e}")

        # --- Last resort: FRED public CSV (often times out) ---
        # --- Step 3: FRED public CSV last resort (often blocked) ---
        for sid in [s for s in series_list if s not in frames]:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            try:
                r = requests.get(url, timeout=(10, 60),
                                  headers={"User-Agent": "Mozilla/5.0"}, stream=True)
                r.raise_for_status()
                content = b"".join(r.iter_content(8192)).decode()
                df = pd.read_csv(io.StringIO(content), parse_dates=["DATE"])
                df = df.rename(columns={"DATE": "date", df.columns[1]: sid})
                df = df.set_index("date").sort_index()
                frames[sid] = pd.to_numeric(df[sid], errors="coerce")
            except Exception as e:
                warnings.warn(f"FRED CSV {sid} failed: {e}")

        if not frames:
            raise RuntimeError("No macro series fetched — check WRDS token or FRED API key")
        fred = pd.concat(frames.values(), axis=1).sort_index()
        fred.to_parquet(self.fred_pq)
        return fred

    # ── WRDS / CRSP ──────────────────────────────────────────────────────────

    def fetch_crsp_spx_membership(self, refresh: bool = False) -> pd.DataFrame:
        """Historical S&P 500 membership from CRSP (crsp.msp500list).

        Returns DataFrame [permno, ticker, start_date, end_date].
        REST API field names: start → start_date, ending → end_date.
        """
        cache = self.data_dir / "crsp_spx_membership.parquet"
        if cache.exists() and not refresh:
            return pd.read_parquet(cache)

        # msp500list: permno, start, ending
        memb = self._wrds_fetch(
            "crsp.msp500list",
            date_cols=["start", "ending"],
        ).rename(columns={"start": "start_date", "ending": "end_date"})

        # stocknames for ticker mapping: permno, ticker, namedt, nameenddt
        names = self._wrds_fetch(
            "crsp.stocknames",
            date_cols=["namedt", "nameenddt"],
        )[["permno", "ticker", "namedt", "nameenddt"]]

        df = memb.merge(names, on="permno", how="left")
        df.to_parquet(cache)
        return df

    def fetch_crsp_daily(
        self,
        tickers: Optional[Iterable[str]] = None,
        start: str = "2014-01-01",
        refresh: bool = False,
        wrds_username: str = "",
        spx_only: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """CRSP adjusted daily close prices and volume via direct PostgreSQL.

        Uses wrds.Connection().raw_sql() — one SQL query, not thousands of
        paginated REST requests. Requires ~/.pgpass credential (run
        shared/wrds_connection.py::setup_pgpass() once to store it).

        Do NOT call .cursor() on the wrds Connection — it wraps SQLAlchemy,
        not raw psycopg2. Only .raw_sql(sql, date_cols=[...]) is available.

        Price adjustment: prc is negative when CRSP records a bid quote instead
        of a trade price; ABS() in SQL handles this before dividing by cfacpr.

        spx_only=True (default): filters by permno from crsp.msp500list —
        all historical S&P 500 constituents including entrants and exits.
        No survivorship bias. Do NOT pass tickers= for this use case;
        ticker-based filtering is unreliable (tickers get reused across firms).

        Returns (close_df, volume_df) with shape (dates × tickers).
        """
        close_cache  = self.data_dir / "crsp_close.parquet"
        volume_cache = self.data_dir / "crsp_volume.parquet"
        if close_cache.exists() and volume_cache.exists() and not refresh:
            return pd.read_parquet(close_cache), pd.read_parquet(volume_cache)

        import wrds as _wrds

        username = wrds_username or os.environ.get("WRDS_USERNAME", "")
        db = _wrds.Connection(wrds_username=username) if username else _wrds.Connection()

        try:
            # Build permno filter — prefer spx_only (stable permno) over tickers
            if spx_only and not tickers:
                # All historical S&P 500 members via permanent ID — no survivorship bias
                permno_clause = """
                    AND d.permno IN (
                        SELECT DISTINCT permno
                        FROM crsp.msp500list
                    )"""
            elif tickers:
                ticker_sql = ", ".join(f"'{t}'" for t in tickers)
                permno_clause = f"""
                    AND d.permno IN (
                        SELECT DISTINCT permno
                        FROM crsp.stocknames
                        WHERE ticker IN ({ticker_sql})
                    )"""
            else:
                permno_clause = ""

            # Pull total return (ret) + volume.
            # ret is CRSP's dividend-adjusted daily return (decimal).
            # We build a total-return index (TRI) = (1+ret).cumprod() so that
            # dividends are properly included in spread construction.
            # Using prc/cfacpr (split-adjusted price only) causes artificial
            # spread jumps at every dividend date → false entry signals.
            sql = f"""
                SELECT
                    d.date,
                    n.ticker,
                    d.ret,
                    d.vol
                FROM crsp.dsf AS d
                INNER JOIN crsp.stocknames AS n
                    ON  d.permno = n.permno
                    AND d.date  >= n.namedt
                    AND d.date  <= COALESCE(n.nameenddt, CURRENT_DATE)
                WHERE d.date >= '{start}'
                  {permno_clause}
            """
            # Prefer SQLAlchemy connection (stable across wrds versions);
            # fall back to raw_sql or raw DBAPI connection.
            try:
                if hasattr(db, "engine") and db.engine is not None:
                    try:
                        from sqlalchemy import text as _sql_text
                    except Exception:
                        _sql_text = None
                    # Use an explicit Connection, not Engine, to avoid any
                    # code path that tries engine.cursor
                    with db.engine.connect() as conn:
                        if _sql_text is not None:
                            dsf = pd.read_sql_query(_sql_text(sql), conn, parse_dates=["date"])
                        else:
                            dsf = pd.read_sql_query(sql, conn, parse_dates=["date"])
                else:
                    dsf = db.raw_sql(sql, date_cols=["date"])
            except Exception as e:
                # Fallback 1: use raw DBAPI connection from engine (has .cursor)
                if hasattr(db, "engine") and db.engine is not None:
                    try:
                        conn = db.engine.raw_connection()
                        try:
                            dsf = pd.read_sql_query(sql, conn, parse_dates=["date"])
                        finally:
                            conn.close()
                    except Exception:
                        # Fallback 2: try wrds raw_sql if available
                        dsf = db.raw_sql(sql, date_cols=["date"])
                else:
                    raise
        finally:
            db.close()

        # Build total-return index (TRI) from ret.
        # ret=NaN means delisted/no-trade — fill with 0 (price unchanged assumption).
        # Delisted stocks will have flat TRI after delisting and won't cointegrate
        # with active stocks, so they're naturally filtered by ADF/HL screens.
        dsf = dsf.dropna(subset=["ticker"])
        dsf["ret"] = dsf["ret"].fillna(0.0)

        ret_wide  = dsf.pivot_table(index="date", columns="ticker", values="ret").sort_index()
        close_df  = (1 + ret_wide).cumprod()   # TRI anchored to 1.0 at first date
        volume_df = dsf.pivot_table(index="date", columns="ticker", values="vol").sort_index()

        close_df.to_parquet(close_cache)
        volume_df.to_parquet(volume_cache)
        return close_df, volume_df

    # ── WRDS / IBES (earnings calendar) ──────────────────────────────────────

    def fetch_ibes_earnings_dates(
        self,
        tickers: Optional[Iterable[str]] = None,
        start: str = "2014-01-01",
        refresh: bool = False,
        wrds_username: str = "",
    ) -> pd.DataFrame:
        """IBES actual earnings announcement dates per ticker.

        When wrds_username is provided, uses direct PostgreSQL (pgpass) — one
        query, much faster than paginated REST for multi-year pulls.
        Falls back to WRDS REST API when wrds_username is empty.

        Filter: fpi=1 (quarterly actuals). Returns DataFrame [ticker, earnings_date].
        """
        cache = self.data_dir / "ibes_earnings.parquet"
        if cache.exists() and not refresh:
            return pd.read_parquet(cache)

        username = wrds_username or os.environ.get("WRDS_USERNAME", "")

        if username:
            import wrds as _wrds
            ticker_clause = ""
            if tickers:
                t_sql = ", ".join(f"'{t}'" for t in tickers)
                ticker_clause = f"AND ticker IN ({t_sql})"
            sql = f"""
                SELECT DISTINCT ticker, anndats_act
                FROM ibes.statsum_epsus
                WHERE fpi = '1'
                  AND anndats_act IS NOT NULL
                  AND anndats_act >= '{start}'
                  {ticker_clause}
                ORDER BY ticker, anndats_act
            """
            db = _wrds.Connection(wrds_username=username) if username else _wrds.Connection()
            try:
                try:
                    if hasattr(db, "engine") and db.engine is not None:
                        try:
                            from sqlalchemy import text as _sql_text
                        except Exception:
                            _sql_text = None
                        with db.engine.connect() as conn:
                            df = pd.read_sql_query(
                                _sql_text(sql) if _sql_text else sql, conn)
                    else:
                        df = db.raw_sql(sql)
                except Exception:
                    if hasattr(db, "engine") and db.engine is not None:
                        conn = db.engine.raw_connection()
                        try:
                            df = pd.read_sql_query(sql, conn)
                        finally:
                            conn.close()
                    else:
                        df = db.raw_sql(sql)
            finally:
                db.close()
            # normalise column name regardless of case returned by driver
            date_col = next((c for c in df.columns if c.lower() == "anndats_act"), None)
            if date_col is None:
                raise RuntimeError(f"anndats_act not found in IBES result. Columns: {list(df.columns)}")
            df = df.rename(columns={date_col: "earnings_date"})
            df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
        else:
            params: dict = {"anndats_act__gte": start, "fpi": "1"}
            if tickers:
                params["ticker__in"] = ",".join(tickers)
            df = self._wrds_fetch(
                "ibes.statsum_epsus",
                params=params,
                date_cols=["anndats_act"],
            )
            if df.empty:
                raise RuntimeError("No IBES data returned")
            df = df[["ticker", "anndats_act"]].rename(columns={"anndats_act": "earnings_date"})

        df = (
            df
            .drop_duplicates()
            .dropna(subset=["earnings_date"])
            .loc[lambda d: d["earnings_date"] >= start]
            .sort_values(["ticker", "earnings_date"])
            .reset_index(drop=True)
        )
        df.to_parquet(cache)
        return df

    # ── Markit borrow (stub — sfi.loan_rate not on WRDS REST API) ─────────────

    def fetch_markit_borrow(
        self,
        tickers: Optional[Iterable[str]] = None,
        start: str = "2014-01-01",
        refresh: bool = False,
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "sfi.loan_rate is not available via the WRDS REST API. "
            "Use WRDS SAS/Python client with direct PostgreSQL access."
        )

    def fetch_bbg(self):
        raise NotImplementedError("Bloomberg stub — replace with BLP/WAPI calls")

    # ── yfinance fallback ─────────────────────────────────────────────────────

    def yf_equities(self, n: int = 300, refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        from spx_data import download_spx_universe

        if not refresh and self.spx_close_pq.exists() and self.spx_vol_pq.exists():
            return pd.read_parquet(self.spx_close_pq), pd.read_parquet(self.spx_vol_pq)
        close_df, vol_df = download_spx_universe(n=n, refresh=refresh)
        close_df.to_parquet(self.spx_close_pq)
        vol_df.to_parquet(self.spx_vol_pq)
        return close_df, vol_df

    def yf_factor_etfs(self, refresh: bool = False) -> pd.DataFrame:
        from spx_data import download_factor_etfs

        if not refresh and self.factor_close_pq.exists():
            return pd.read_parquet(self.factor_close_pq)
        fc = download_factor_etfs(refresh=refresh)
        fc.to_parquet(self.factor_close_pq)
        return fc

    # ── Unified entrypoint ────────────────────────────────────────────────────

    def load_all(
        self,
        n: int = 300,
        refresh: bool = False,
        prefer_factors: str = "both",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
        """Return (stock_close, stock_volume, stock_returns, factor_close_or_None, factor_returns).

        prefer_factors: "both" | "ff_fred_first" | "etf_only"
        """
        close_df, vol_df = self.yf_equities(n=n, refresh=refresh)
        stock_returns = close_df.pct_change()

        if prefer_factors in ("ff_fred_first", "both"):
            ff = self.fetch_ff_daily(refresh=refresh)
            fred_ids = ["DGS2", "DGS10", "T10Y2Y", "BAMLC0A0CM", "BAMLH0A0HYM2"]
            try:
                fred = self.fetch_fred(fred_ids, refresh=refresh)
                F = ff.join(fred.diff().add_prefix("FRED_d"), how="outer")
            except Exception:
                F = ff
            factor_returns = F.reindex(stock_returns.index)
            factor_close = None
            if prefer_factors == "both":
                try:
                    fc = self.yf_factor_etfs(refresh=refresh)
                    etf_rets = fc.pct_change().reindex(stock_returns.index).add_prefix("ETF_")
                    factor_returns = factor_returns.join(etf_rets, how="outer")
                    factor_close = fc.reindex(stock_returns.index)
                except Exception:
                    pass
        else:
            fc = self.yf_factor_etfs(refresh=refresh)
            factor_returns = fc.pct_change().reindex(stock_returns.index)
            factor_close = fc.reindex(stock_returns.index)

        return close_df, vol_df, stock_returns, factor_close, factor_returns


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fetch all data for pairs-trading sandbox")
    ap.add_argument("--token",    default=os.environ.get("WRDS_API_TOKEN", ""),
                    help="WRDS API token for REST API calls (ff, fred, ibes)")
    ap.add_argument("--username", default=os.environ.get("WRDS_USERNAME", ""),
                    help="WRDS username for direct SQL (CRSP daily). Uses ~/.pgpass for password.")
    ap.add_argument("--wrds",    action="store_true", help="Fetch WRDS data (CRSP, IBES)")
    ap.add_argument("--crsp-only", action="store_true",
                    help="Fetch CRSP daily prices only (skips ff, fred, ibes, yfinance)")
    ap.add_argument("--no-spx-filter", action="store_true",
                    help="Disable S&P 500 permno filter — fetches all CRSP stocks (very large)")
    ap.add_argument("--refresh", action="store_true", help="Re-download even if cached")
    ap.add_argument("--start",   default="2014-01-01", help="Start date for WRDS queries")
    args = ap.parse_args()

    f = DataFetcher(allow_network=True, wrds_token=args.token)
    spx_only = not args.no_spx_filter

    print("=== Fetching project data ===\n")

    # -- CRSP-only fast path --------------------------------------------------
    if args.crsp_only:
        username = args.username or os.environ.get("WRDS_USERNAME", "")
        if not username:
            ap.error("--crsp-only requires --username or WRDS_USERNAME env var")
        try:
            c, v = f.fetch_crsp_daily(
                start=args.start,
                refresh=args.refresh,
                wrds_username=username,
                spx_only=spx_only,
            )
            print(f"[ok] crsp_close.parquet   {c.shape}  "
                  f"{c.index[0].date()} → {c.index[-1].date()}")
            print(f"[ok] crsp_volume.parquet  {v.shape}")
        except Exception as e:
            print(f"[!!] crsp_close/volume FAILED: {e}")
        print("\n=== data/ ===")
        for p in sorted(f.data_dir.glob("*.parquet")):
            print(f"  {p.name:<40} {p.stat().st_size / 1e6:6.1f} MB")
        raise SystemExit(0)

    # 1. FF factors (WRDS REST or Ken French fallback)
    try:
        ff = f.fetch_ff_daily(refresh=args.refresh)
        src = "WRDS" if args.token else "Ken French"
        print(f"[ok] ff_daily.parquet  {ff.shape}  {ff.index[0].date()} → {ff.index[-1].date()}  ({src})")
    except Exception as e:
        print(f"[!!] ff_daily FAILED: {e}")

    # 2. FRED macro
    FRED_IDS = ["DGS2", "DGS10", "T10Y2Y", "BAMLC0A0CM", "BAMLH0A0HYM2"]
    try:
        fred = f.fetch_fred(FRED_IDS, refresh=args.refresh)
        print(f"[ok] fred.parquet      {fred.shape}")
    except Exception as e:
        print(f"[!!] fred FAILED: {e}")

    # 3. Factor ETFs
    try:
        fc = f.yf_factor_etfs(refresh=args.refresh)
        print(f"[ok] factor_close.parquet  {fc.shape}")
    except Exception as e:
        print(f"[!!] factor_etfs FAILED: {e}")

    if not args.wrds:
        # Fallback equities
        try:
            eq_c, eq_v = f.yf_equities(refresh=args.refresh)
            print(f"[ok] spx_close.parquet  {eq_c.shape}  (yfinance fallback)")
        except Exception as e:
            print(f"[!!] yf_equities FAILED: {e}")
        print("\nRe-run with --wrds --token <token> to fetch CRSP/IBES.")
    else:
        # 4. CRSP membership
        try:
            memb = f.fetch_crsp_spx_membership(refresh=args.refresh)
            print(f"[ok] crsp_spx_membership.parquet  {memb.shape}")
        except Exception as e:
            print(f"[!!] crsp_spx_membership FAILED: {e}")

        # 5. CRSP daily prices (direct SQL — requires ~/.pgpass, not REST token)
        try:
            c, v = f.fetch_crsp_daily(start=args.start, refresh=args.refresh,
                                       wrds_username=args.username, spx_only=spx_only)
            print(f"[ok] crsp_close.parquet   {c.shape}")
            print(f"[ok] crsp_volume.parquet  {v.shape}")
        except Exception as e:
            print(f"[!!] crsp_close/volume FAILED: {e}")

        # 6. IBES earnings (raw SQL when --username set, REST fallback otherwise)
        try:
            ibes = f.fetch_ibes_earnings_dates(
                start=args.start, refresh=args.refresh, wrds_username=args.username)
            print(f"[ok] ibes_earnings.parquet  {ibes.shape}  ({ibes['ticker'].nunique()} tickers)")
        except Exception as e:
            print(f"[!!] ibes_earnings FAILED: {e}")

    print("\n=== data/ ===")
    for p in sorted(f.data_dir.glob("*.parquet")):
        print(f"  {p.name:<40} {p.stat().st_size / 1e6:6.1f} MB")
