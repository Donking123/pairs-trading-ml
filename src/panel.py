"""
Formation-window panel helper  [Pipeline Stage 1 / 2 bridge].

Takes the Phase-0 cached CRSP data and produces a clean price matrix for one
formation window — exactly the shape distances.py / clustering.py expect: rows =
trading days, columns = stock identifiers (permno), no NaNs.

The two key correctness properties enforced here:
  1. *Survivorship-bias-free*: only stocks continuously in the S&P 500 over the
     entire formation window are kept.
  2. *Continuously priced*: stocks missing any trading day in the window are
     dropped. (Otherwise SSD distance would be ill-defined.)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, FORMATION_YEARS


TRADING_DAYS_PER_YEAR = 252  # US equity convention


def load_crsp_daily(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the cached CRSP daily panel (long format)."""
    return pd.read_parquet(data_dir / "crsp_daily.parquet")


def load_sp500_constituents(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the S&P 500 constituent membership intervals."""
    return pd.read_parquet(data_dir / "sp500_constituents.parquet")


def load_market_returns(data_dir: Path = DATA_DIR) -> pd.Series:
    """Load the S&P 500 daily total-return series (the 'market return' for PC distance).

    Returns a Series indexed by date with daily returns (e.g. 0.012 = +1.2%).
    Sourced from `sp500_index.parquet::sprtrn` (CRSP's total-return field).
    """
    sp = pd.read_parquet(data_dir / "sp500_index.parquet")
    return pd.Series(sp["sprtrn"].to_numpy(), index=pd.DatetimeIndex(sp["caldt"]), name="mkt_ret")


def total_return_price(crsp: pd.DataFrame) -> pd.DataFrame:
    """Add a 'tr_price' column = total-return price index per stock.

    Built from CRSP's `ret` (total return, dividend-reinvested) by cumprod-ing
    within each permno. Starts at 100 on each stock's first available date.
    The level is arbitrary because SSD z-normalises before comparing — only the
    *shape* of the trajectory matters.
    """
    crsp = crsp.sort_values(["permno", "date"]).copy()
    crsp["tr_price"] = (
        crsp.groupby("permno")["ret"].transform(lambda r: 100.0 * (1 + r.fillna(0)).cumprod())
    )
    return crsp


def members_on_date(constituents: pd.DataFrame, as_of: pd.Timestamp) -> set[int]:
    """Permnos that were S&P 500 constituents on (or spanning) `as_of`."""
    mask = (constituents["start"] <= as_of) & (constituents["ending"] >= as_of)
    return set(constituents.loc[mask, "permno"].tolist())


def formation_window_panel(
    as_of: str | pd.Timestamp,
    crsp: pd.DataFrame | None = None,
    constituents: pd.DataFrame | None = None,
    formation_years: int = FORMATION_YEARS,
    use_total_return: bool = True,
) -> pd.DataFrame:
    """Build a clean formation-window price matrix ending on `as_of`.

    Parameters
    ----------
    as_of : timestamp
        Last trading day of the formation window (e.g. "2023-12-29").
    crsp, constituents : DataFrames
        Pre-loaded panels — if None we load from disk.
    formation_years : int
        Window length in years (default 3 = paper's standard).
    use_total_return : bool
        If True (default) use the dividend-reinvested total-return price index.
        If False use raw close price (CRSP `prc`, sign-corrected and split-adjusted).

    Returns
    -------
    DataFrame
        Rows = trading days (~756 if 3 years); columns = permno of stocks that were
        (a) continuously in the S&P 500 across the whole window AND
        (b) continuously priced (no NaN) across the whole window.
        No NaNs anywhere — safe to hand to distances.py / clustering.py.
    """
    if crsp is None:
        crsp = load_crsp_daily()
    if constituents is None:
        constituents = load_sp500_constituents()

    as_of = pd.Timestamp(as_of)
    n_days = formation_years * TRADING_DAYS_PER_YEAR

    # 1. Slice crsp_daily to the window: take all rows up to as_of, then keep the
    #    last n_days unique trading dates
    sub = crsp.loc[crsp["date"] <= as_of].copy()
    trading_days = sub["date"].drop_duplicates().sort_values()
    window_days = trading_days.tail(n_days)
    if len(window_days) < n_days:
        raise ValueError(
            f"only {len(window_days)} trading days available before {as_of.date()}, "
            f"need {n_days}"
        )
    window_start, window_end = window_days.iloc[0], window_days.iloc[-1]
    sub = sub.loc[sub["date"].between(window_start, window_end)]

    # 2. Survivorship filter: stocks continuously in S&P 500 over the whole window.
    #    A stock counts as 'continuously in' if its membership interval covers both
    #    window_start and window_end.
    cont_mask = (
        (constituents["start"] <= window_start) & (constituents["ending"] >= window_end)
    )
    continuously_in_index = set(constituents.loc[cont_mask, "permno"].tolist())
    sub = sub.loc[sub["permno"].isin(continuously_in_index)]

    # 3. Choose price column
    if use_total_return:
        sub = total_return_price(sub)
        price_col = "tr_price"
    else:
        # CRSP `prc` is negated when only bid/ask midpoint is available — take abs;
        # cfacpr is the cumulative price-adjustment factor (splits, special dividends)
        sub = sub.copy()
        sub["adj_price"] = sub["prc"].abs() / sub["cfacpr"].replace(0, pd.NA)
        price_col = "adj_price"

    # 4. Pivot: dates × permno
    panel = sub.pivot_table(index="date", columns="permno", values=price_col, aggfunc="first")
    panel = panel.reindex(window_days)  # ensure full date index, fill missing days with NaN

    # 5. Continuously-priced filter: drop any stock with NaN in the window
    n_before = panel.shape[1]
    panel = panel.dropna(axis=1, how="any")
    # CRSP nullable types can leave the panel as dtype=object; coerce to float64 so
    # downstream numerical code (scipy.pdist) gets a real numeric array.
    panel = panel.astype("float64")
    n_after = panel.shape[1]
    print(
        f"formation_window_panel({as_of.date()}): "
        f"{len(window_days)} days, "
        f"{n_after} stocks kept "
        f"({n_before - n_after} dropped for missing prices)"
    )

    return panel


def ticker_lookup(
    permnos: list[int] | None = None,
    crsp: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
) -> pd.Series:
    """Map permno -> ticker (most recent ticker on or before `as_of`)."""
    if crsp is None:
        crsp = load_crsp_daily()
    sub = crsp[["permno", "date", "ticker"]].dropna()
    if as_of is not None:
        sub = sub.loc[sub["date"] <= as_of]
    sub = sub.sort_values(["permno", "date"]).drop_duplicates("permno", keep="last")
    tick = sub.set_index("permno")["ticker"]
    if permnos is not None:
        tick = tick.reindex(permnos)
    return tick


def siccd_lookup(
    permnos: list[int] | None = None,
    crsp: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
) -> pd.Series:
    """Map permno -> SIC code (most recent SIC on or before `as_of`)."""
    if crsp is None:
        crsp = load_crsp_daily()
    sub = crsp[["permno", "date", "siccd"]].dropna()
    if as_of is not None:
        sub = sub.loc[sub["date"] <= as_of]
    sub = sub.sort_values(["permno", "date"]).drop_duplicates("permno", keep="last")
    sic = sub.set_index("permno")["siccd"]
    if permnos is not None:
        sic = sic.reindex(permnos)
    return sic
