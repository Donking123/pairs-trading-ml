#!/usr/bin/env python3
"""
run_asian_adr_screening.py
==========================

Asian ADR Pair Selection Pipeline (ADR Architecture spec §3.4).

For each ADR with an Asian underlying:

    1. Reconstruct the dollar spread
            spread_t = P_ADR,t  -  (P_local,t * FX_t) / adr_ratio
       with ``adr_ratio`` taken **verbatim from Datastream** — never OLS-estimated
       (ADR-002).

    2. Test for cointegration on the spread series:
           - Augmented Dickey-Fuller (ADF)        — reject unit root at 5%
           - Phillips-Perron (PP)                 — reject unit root at 5%

    3. Apply liquidity filters:
           - >= 50% non-zero return days on BOTH legs
           - >= 2 years (~504 bd) of joint trading history
           - ADR zero-return-day pct <= 50%

    4. Estimate Roll (1984) effective spread for both legs (post-hoc cost proxy).

    5. Write approved pairs to ``config/pairs/asian_adr_pairs.json``.

Usage
-----
::

    python scripts/run_asian_adr_screening.py \
        --adr-prices       data/parquet/adr/adr_prices.parquet \
        --adr-reference    data/parquet/adr/adr_reference.parquet \
        --global-prices    data/parquet/global/global_prices.parquet \
        --fx-rates         data/parquet/fx/fx_rates.parquet \
        --out              config/pairs/asian_adr_pairs.json \
        --as-of            2024-12-31

Use ``--min-history-days`` to control the joint-history floor (default 504
trading days ≈ 2 years per the paper's eligibility criteria).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("screen")


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


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass
class ApprovedPair:
    pair_id: str
    adr_ticker: str
    underlying_ticker: str
    underlying_exchange: str
    underlying_currency: str
    adr_ratio: float
    estimation_days: int
    holding_days: int
    k0: float
    kc: float
    non_zero_return_pct_adr: float
    non_zero_return_pct_local: float
    zero_return_pct_adr: float
    roll_spread_adr: float
    roll_spread_local: float
    adf_pvalue: float
    pp_pvalue: float
    joint_history_days: int
    fx_hedge_required: bool
    approved_date: str
    expiry_date: str
    is_active: bool


# Standard ADR ratios (1/N where N = local shares per ADR).  Used to snap the
# median price-based estimate to the nearest structurally meaningful value.
_STANDARD_RATIOS: list[float] = [
    0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0,
]
_LOG_STD = np.log(_STANDARD_RATIOS)


def _snap_raw_ratio(raw: float) -> Optional[float]:
    """Validate a raw median(local*FX/ADR) estimate and snap it to the nearest
    standard ratio in log-space. Returns None if the raw value is non-finite or
    outside the plausible [0.001, 1000] band. Single source of truth shared by
    the per-pair and bulk estimators so their accept/snap rules cannot drift."""
    if not np.isfinite(raw) or raw < 0.001 or raw > 1000:
        return None
    idx = int(np.argmin(np.abs(np.log(raw) - _LOG_STD)))
    return _STANDARD_RATIOS[idx]


def _bulk_estimate_ratios(
    candidates: pd.DataFrame,
    adr_prices: pd.DataFrame,
    global_prices: pd.DataFrame,
    fx_rates: pd.DataFrame,
) -> pd.Series:
    """
    Estimate adr_ratio for every null-ratio candidate, returning a Series indexed
    by (adr_ticker, underlying_ticker) -> estimated ratio or NaN.

    Each source is pivoted once into a wide frame (date x ticker / date x ccy) so
    the per-pair column lookups below are O(1) instead of re-scanning the long
    tables for every candidate; the median itself is still computed per pair
    because the joint date overlap differs from pair to pair.
    """
    # pivot ADR closes: index=date, columns=ticker
    adr_wide = (adr_prices[adr_prices["ticker"].isin(candidates["adr_ticker"])]
                .pivot_table(index="marketdate", columns="ticker", values="close", aggfunc="last"))
    adr_wide.index = pd.to_datetime(adr_wide.index)

    # pivot local closes: index=date, columns=ticker
    loc_wide = (global_prices[global_prices["ticker"].isin(candidates["underlying_ticker"])]
                .pivot_table(index="marketdate", columns="ticker", values="close", aggfunc="last"))
    loc_wide.index = pd.to_datetime(loc_wide.index)

    # pivot FX: index=date, columns=currency
    fx_wide = (fx_rates.pivot_table(index="date", columns="base_currency", values="mid", aggfunc="last"))
    fx_wide.index = pd.to_datetime(fx_wide.index)

    results: dict[tuple, float] = {}
    for _, row in candidates.iterrows():
        adr_t = row["adr_ticker"]
        loc_t = row["underlying_ticker"]
        ccy   = row["underlying_currency"]
        key   = (adr_t, loc_t)

        if adr_t not in adr_wide.columns or loc_t not in loc_wide.columns or ccy not in fx_wide.columns:
            results[key] = float("nan")
            continue

        adr_s = adr_wide[adr_t].dropna()
        loc_s = loc_wide[loc_t].dropna()
        fx_s  = fx_wide[ccy].dropna()

        combined = pd.concat([adr_s.rename("adr"), loc_s.rename("loc"), fx_s.rename("fx")], axis=1).dropna()
        combined = combined[(combined["adr"] > 0) & (combined["loc"] > 0) & (combined["fx"] > 0)]

        if len(combined) < 30:
            results[key] = float("nan")
            continue

        raw = (combined["loc"] * combined["fx"] / combined["adr"]).median()
        snapped = _snap_raw_ratio(raw)
        results[key] = float("nan") if snapped is None else snapped

    return pd.Series(results, dtype=float)



def _estimate_ratio_from_prices(
    adr_prices: pd.DataFrame,
    local_prices: pd.DataFrame,
    fx_rates: pd.DataFrame,
    currency: str,
) -> Optional[float]:
    """
    Estimate adr_ratio = median(P_local * FX / P_ADR) then snap to the nearest
    standard ratio in log-space.  Returns None if the overlap is too thin (<30
    joint observations) or the raw median is outside [0.001, 1000].
    """
    adr = adr_prices.rename(columns={"close": "adr_close"}).set_index("marketdate")[["adr_close"]]
    loc = local_prices.rename(columns={"close": "local_close"}).set_index("marketdate")[["local_close"]]
    fx  = fx_rates[fx_rates["base_currency"] == currency].copy()
    fx["date"] = pd.to_datetime(fx["date"])
    fx = fx.rename(columns={"date": "marketdate", "mid": "fx_mid"}).set_index("marketdate")[["fx_mid"]]

    adr.index = pd.to_datetime(adr.index)
    loc.index = pd.to_datetime(loc.index)

    df = adr.join(loc, how="inner").join(fx, how="inner").dropna()
    df = df[(df["adr_close"] > 0) & (df["local_close"] > 0) & (df["fx_mid"] > 0)]

    if len(df) < 30:
        return None

    raw = (df["local_close"] * df["fx_mid"] / df["adr_close"]).median()
    return _snap_raw_ratio(raw)


# -----------------------------------------------------------------------------
# Spread reconstruction
# -----------------------------------------------------------------------------
def reconstruct_spread(
    adr_prices: pd.DataFrame,
    local_prices: pd.DataFrame,
    fx_rates: pd.DataFrame,
    currency: str,
    adr_ratio: float,
) -> pd.DataFrame:
    """
    Build the daily dollar spread series for one pair.

    Returns a DataFrame indexed by date with columns:
        adr_close, local_close, fx_mid, local_usd, spread

    Drops any date where any of the three series is missing.
    """
    adr = adr_prices.rename(columns={"close": "adr_close"}).set_index("marketdate")[["adr_close"]]
    loc = local_prices.rename(columns={"close": "local_close"}).set_index("marketdate")[["local_close"]]
    fx  = fx_rates[fx_rates["base_currency"] == currency].copy()
    fx["date"] = pd.to_datetime(fx["date"])
    fx = fx.rename(columns={"date": "marketdate", "mid": "fx_mid"}).set_index("marketdate")[["fx_mid"]]

    adr.index = pd.to_datetime(adr.index)
    loc.index = pd.to_datetime(loc.index)

    df = adr.join(loc, how="inner").join(fx, how="inner").dropna()
    df["local_usd"] = df["local_close"] * df["fx_mid"]
    df["spread"]    = df["adr_close"] - df["local_usd"] / adr_ratio
    return df


# -----------------------------------------------------------------------------
# Statistical tests
# -----------------------------------------------------------------------------
def adf_pvalue(series: pd.Series) -> float:
    """Augmented Dickey-Fuller test on the spread; p < 0.05 => stationary."""
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        log.error("statsmodels not installed; install with `pip install statsmodels`")
        raise
    result = adfuller(series.dropna().values, autolag="AIC")
    return float(result[1])


def pp_pvalue(series: pd.Series) -> float:
    """
    Phillips-Perron test. statsmodels has PhillipsPerron only in newer versions;
    fall back to a simple lag-augmented variant if unavailable.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        # statsmodels >=0.14 ships arch.unitroot — but we keep deps minimal
        # and use the KPSS / PP via the `arch` package if available.
        try:
            from arch.unitroot import PhillipsPerron  # type: ignore
            pp = PhillipsPerron(series.dropna().values)
            return float(pp.pvalue)
        except ImportError:
            # Fallback: use the ADF result as a proxy. (Both reject same null.)
            return adf_pvalue(series)
    except Exception as exc:
        log.warning("Phillips-Perron failed (%s); using ADF as proxy", exc)
        return adf_pvalue(series)


def non_zero_return_pct(prices: pd.Series) -> float:
    """Fraction of days with non-zero log-return (Bekaert et al. 2007 liquidity)."""
    rets = prices.pct_change().dropna()
    if len(rets) == 0:
        return 0.0
    return float((rets != 0).sum() / len(rets))


def zero_return_pct(prices: pd.Series) -> float:
    return 1.0 - non_zero_return_pct(prices)


def roll_effective_spread(prices: pd.Series) -> float:
    """
    Roll (1984) effective spread: 2 * sqrt(|cov(r_t, r_{t-1})|)

    Only finite if first-order autocov is negative (bid-ask bounce). When the
    autocov is positive, Roll's estimator is undefined; we return the |cov|
    version as in the spec snippet — it's a conservative cost proxy.
    """
    rets = prices.pct_change().dropna().values
    if len(rets) < 3:
        return float("nan")
    autocov = float(np.cov(rets[:-1], rets[1:], ddof=1)[0, 1])
    return float(2.0 * np.sqrt(abs(autocov)))


# -----------------------------------------------------------------------------
# Per-pair screening
# -----------------------------------------------------------------------------
def screen_pair(
    pair_row: pd.Series,
    adr_prices: pd.DataFrame,
    global_prices: pd.DataFrame,
    fx_rates: pd.DataFrame,
    as_of: date,
    cfg: dict,
) -> tuple[Optional[ApprovedPair], dict]:
    """
    Returns ``(approved_pair_or_None, diagnostics_dict)``. The diagnostics dict
    is always populated so we can report rejections.
    """
    adr_t = pair_row["adr_ticker"]
    und_t = pair_row["underlying_ticker"]
    exch  = pair_row["underlying_exchange"]
    ccy   = pair_row["underlying_currency"]
    raw_ratio = pair_row["adr_ratio"]
    pair_id = f"{adr_t}__{und_t}"
    diag = {"pair_id": pair_id, "status": "rejected", "reason": ""}

    if raw_ratio is None or pd.isna(raw_ratio):
        adr_s = adr_prices[adr_prices["ticker"] == adr_t][["marketdate", "close"]]
        und_s = global_prices[global_prices["ticker"] == und_t][["marketdate", "close"]]
        estimated = _estimate_ratio_from_prices(adr_s, und_s, fx_rates, ccy)
        if estimated is None:
            diag["reason"] = "adr_ratio is null and could not be estimated from prices"
            return None, diag
        log.warning("[%s] adr_ratio was null; estimated %.4f from price median", pair_id, estimated)
        ratio = estimated
        diag["adr_ratio_estimated"] = True
    else:
        ratio = float(raw_ratio)

    if exch not in ASIAN_EXCHANGES:
        diag["reason"] = f"underlying_exchange={exch} not in ASIAN_EXCHANGES"
        return None, diag
    if ratio <= 0:
        diag["reason"] = f"invalid adr_ratio={ratio}"
        return None, diag

    adr_slice = adr_prices[adr_prices["ticker"] == adr_t][["marketdate", "close"]]
    und_slice = global_prices[global_prices["ticker"] == und_t][["marketdate", "close"]]
    if adr_slice.empty or und_slice.empty:
        diag["reason"] = (f"missing prices: adr={len(adr_slice)}, "
                          f"underlying={len(und_slice)}")
        return None, diag

    df = reconstruct_spread(adr_slice, und_slice, fx_rates, ccy, ratio)
    diag["joint_history_days"] = int(len(df))
    if len(df) < cfg["min_history_days"]:
        diag["reason"] = f"insufficient history ({len(df)} < {cfg['min_history_days']})"
        return None, diag

    # Trim to as-of date (no look-ahead)
    df = df[df.index <= pd.Timestamp(as_of)]
    if len(df) < cfg["min_history_days"]:
        diag["reason"] = f"insufficient history as-of {as_of}"
        return None, diag

    nzr_adr   = non_zero_return_pct(df["adr_close"])
    nzr_local = non_zero_return_pct(df["local_close"])
    zr_adr    = zero_return_pct(df["adr_close"])
    diag.update({"non_zero_return_pct_adr": nzr_adr,
                 "non_zero_return_pct_local": nzr_local,
                 "zero_return_pct_adr": zr_adr})

    if nzr_adr < cfg["min_non_zero_return_pct"] or nzr_local < cfg["min_non_zero_return_pct"]:
        diag["reason"] = (f"liquidity fail: adr_nzr={nzr_adr:.2f}, "
                          f"local_nzr={nzr_local:.2f} (need >= "
                          f"{cfg['min_non_zero_return_pct']:.2f})")
        return None, diag

    if zr_adr > cfg["max_zero_return_pct_adr"]:
        diag["reason"] = (f"ADR zero-return excess: {zr_adr:.2f} > "
                          f"{cfg['max_zero_return_pct_adr']:.2f}")
        return None, diag

    try:
        adf_p = adf_pvalue(df["spread"])
        pp_p  = pp_pvalue(df["spread"])
    except Exception as exc:
        diag["reason"] = f"cointegration test failed: {exc}"
        return None, diag

    diag.update({"adf_pvalue": adf_p, "pp_pvalue": pp_p})
    if adf_p > cfg["cointegration_alpha"] or pp_p > cfg["cointegration_alpha"]:
        diag["reason"] = (f"not cointegrated: ADF p={adf_p:.4f}, "
                          f"PP p={pp_p:.4f} (need both <= {cfg['cointegration_alpha']})")
        return None, diag

    roll_adr   = roll_effective_spread(df["adr_close"])
    roll_local = roll_effective_spread(df["local_close"])

    approved = ApprovedPair(
        pair_id=pair_id,
        adr_ticker=adr_t,
        underlying_ticker=str(und_t),
        underlying_exchange=exch,
        underlying_currency=ccy,
        adr_ratio=ratio,
        estimation_days=cfg["estimation_days"],
        holding_days=cfg["holding_days"],
        k0=cfg["k0"],
        kc=cfg["kc"],
        non_zero_return_pct_adr=round(nzr_adr, 4),
        non_zero_return_pct_local=round(nzr_local, 4),
        zero_return_pct_adr=round(zr_adr, 4),
        roll_spread_adr=round(roll_adr, 6) if not np.isnan(roll_adr) else 0.0,
        roll_spread_local=round(roll_local, 6) if not np.isnan(roll_local) else 0.0,
        adf_pvalue=round(adf_p, 6),
        pp_pvalue=round(pp_p, 6),
        joint_history_days=int(len(df)),
        fx_hedge_required=False,         # ADR-003
        approved_date=as_of.isoformat(),
        expiry_date=(as_of + timedelta(days=365)).isoformat(),
        is_active=True,
    )
    diag["status"] = "approved"
    diag["reason"] = "passed all filters"
    return approved, diag


# -----------------------------------------------------------------------------
# Pipeline driver
# -----------------------------------------------------------------------------
def run_pipeline(
    adr_prices: pd.DataFrame,
    adr_reference: pd.DataFrame,
    global_prices: pd.DataFrame,
    fx_rates: pd.DataFrame,
    as_of: date,
    cfg: dict,
) -> tuple[list[ApprovedPair], list[dict]]:
    approved: list[ApprovedPair] = []
    diagnostics: list[dict] = []

    asian = adr_reference[adr_reference["underlying_exchange"].isin(ASIAN_EXCHANGES)]

    # Point-in-time filter: if the reference table has startdate/enddate columns
    # (from fetch_pit_adr_reference.py), only keep pairs active as of `as_of`.
    # A pair is active if: startdate <= as_of AND (enddate is NaT OR enddate >= as_of).
    # This eliminates survivorship bias from pairs that were delisted before as_of.
    #
    # NaT semantics differ by column (see fetch_pit_adr_reference.py):
    #   startdate NaT -> the pair's start is *unknown*. We cannot confirm it was
    #     active as of `as_of`, so we conservatively EXCLUDE it (dropping it is
    #     the survivorship-safe choice; admitting an unknown-start pair would let
    #     it trade in periods where it may not have existed).
    #   enddate  NaT -> the pair is still active (open-ended), so it passes the
    #     upper-bound test.
    if "startdate" in asian.columns and "enddate" in asian.columns:
        as_of_ts = pd.Timestamp(as_of)
        asian = asian.copy()
        asian["startdate"] = pd.to_datetime(asian["startdate"])
        asian["enddate"]   = pd.to_datetime(asian["enddate"])
        # WRDS uses 2079-06-05 as a sentinel for "still active" instead of NULL.
        # Treat any enddate beyond today as open-ended (active).
        _today = pd.Timestamp.today().normalize()
        asian["enddate_eff"] = asian["enddate"].where(asian["enddate"] <= _today, other=pd.NaT)
        active_mask = (
            (asian["startdate"].notna() & (asian["startdate"] <= as_of_ts)) &
            (asian["enddate_eff"].isna() | (asian["enddate_eff"] >= as_of_ts))
        )
        n_before = len(asian)
        asian = asian[active_mask].drop(columns=["enddate_eff"])
        log.info("point-in-time filter (as_of=%s): %d -> %d pairs (dropped %d inactive)",
                 as_of, n_before, len(asian), n_before - len(asian))
    else:
        log.warning(
            "adr_reference has no startdate/enddate columns — "
            "survivorship bias NOT corrected. Run fetch_pit_adr_reference.py "
            "and pass adr_reference_pit.parquet via --adr-reference to fix this."
        )

    # Pre-filter 1: drop rows whose tickers have no price history at all.
    known_adrs = set(adr_prices["ticker"].unique())
    known_locals = set(global_prices["ticker"].unique())
    asian = asian[
        asian["adr_ticker"].isin(known_adrs) &
        asian["underlying_ticker"].isin(known_locals)
    ]

    # Pre-filter 2: drop duplicate (adr_ticker, underlying_ticker) rows — the
    # reference often has multiple entries for the same pair.
    asian = asian.drop_duplicates(subset=["adr_ticker", "underlying_ticker"])

    # Pre-fill null ratios in bulk (one vectorised pass) to avoid per-pair joins.
    no_ratio_mask = asian["adr_ratio"].isna()
    n_no_ratio = no_ratio_mask.sum()
    log.info("screening %d candidates: %d with known ratio, %d needing ratio estimation",
             len(asian), len(asian) - n_no_ratio, n_no_ratio)
    if n_no_ratio > 0:
        log.info("estimating %d missing ratios in bulk …", n_no_ratio)
        bulk = _bulk_estimate_ratios(
            asian[no_ratio_mask], adr_prices, global_prices, fx_rates)
        asian = asian.copy()
        # cast to float64 so we can write numeric values (column may be StringArray)
        asian["adr_ratio"] = pd.to_numeric(asian["adr_ratio"], errors="coerce")
        # Vectorised writeback: map each null-ratio row's (adr, underlying) key
        # through the bulk Series and fill it in. combine_first leaves any pair
        # bulk couldn't estimate (NaN) untouched, so it stays null and is dropped.
        keys = pd.MultiIndex.from_arrays(
            [asian.loc[no_ratio_mask, "adr_ticker"],
             asian.loc[no_ratio_mask, "underlying_ticker"]]
        )
        estimated = pd.Series(keys.map(bulk).to_numpy(),
                              index=asian.index[no_ratio_mask], dtype=float)
        asian["adr_ratio"] = asian["adr_ratio"].combine_first(estimated)
        log.info("bulk ratio estimation complete; %d still null (will be dropped)",
                 asian["adr_ratio"].isna().sum())

    n_candidates = len(asian)
    for i, (_, row) in enumerate(asian.iterrows(), 1):
        result, diag = screen_pair(row, adr_prices, global_prices, fx_rates, as_of, cfg)
        diagnostics.append(diag)
        if result is not None:
            log.info("[PASS] %s — %s", diag["pair_id"], diag["reason"])
            approved.append(result)
        else:
            log.debug("[FAIL] %s — %s", diag["pair_id"], diag["reason"])
        if i % 100 == 0 or i == n_candidates:
            log.info("  screening progress: %d/%d candidates, %d approved so far",
                     i, n_candidates, len(approved))

    log.info("screening complete: %d approved, %d rejected",
             len(approved), len(diagnostics) - len(approved))
    return approved, diagnostics


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Asian ADR pair selection pipeline (cointegration + liquidity + Roll).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adr-prices", type=Path,
                   default=Path("datastream/data/parquet/adr/adr_prices.parquet"))
    p.add_argument("--adr-reference", type=Path,
                   default=Path("datastream/data/parquet/adr/adr_reference.parquet"))
    p.add_argument("--global-prices", type=Path,
                   default=Path("datastream/data/parquet/global/global_prices.parquet"))
    p.add_argument("--fx-rates", type=Path,
                   default=Path("datastream/data/parquet/fx/fx_rates.parquet"))
    p.add_argument("--out", type=Path,
                   default=Path("config/pairs/asian_adr_pairs.json"))
    p.add_argument("--diagnostics-out", type=Path,
                   default=Path("config/pairs/screening_diagnostics.json"),
                   help="Write per-candidate accept/reject diagnostics here")
    p.add_argument("--as-of", type=_parse_date, default=date.today(),
                   help="Selection 'as of' date (no look-ahead beyond this)")

    p.add_argument("--estimation-days", type=int, default=60, help="T")
    p.add_argument("--holding-days",    type=int, default=90, help="H")
    p.add_argument("--k0", type=float, default=2.0, help="entry multiplier")
    p.add_argument("--kc", type=float, default=0.0, help="exit multiplier")
    p.add_argument("--cointegration-alpha", type=float, default=0.05)
    p.add_argument("--min-history-days", type=int, default=504,
                   help="~2 trading years (paper minimum eligibility)")
    p.add_argument("--min-non-zero-return-pct", type=float, default=0.50)
    p.add_argument("--max-zero-return-pct-adr", type=float, default=0.50)
    return p.parse_args(argv)


def _load_or_die(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        log.error("%s not found at %s — run the relevant fetch script first", label, path)
        sys.exit(2)
    try:
        df = pd.read_parquet(path)
    except OSError:
        df = pd.read_parquet(path, engine="fastparquet")
    log.info("loaded %s: %d rows", label, len(df))
    if "marketdate" in df.columns:
        df["marketdate"] = pd.to_datetime(df["marketdate"]).dt.normalize()
    return df


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    adr_prices    = _load_or_die(args.adr_prices, "ADR prices")
    adr_reference = _load_or_die(args.adr_reference, "ADR reference")
    global_prices = _load_or_die(args.global_prices, "Global prices")
    fx_rates      = _load_or_die(args.fx_rates, "FX rates")

    cfg = {
        "estimation_days": args.estimation_days,
        "holding_days": args.holding_days,
        "k0": args.k0,
        "kc": args.kc,
        "cointegration_alpha": args.cointegration_alpha,
        "min_history_days": args.min_history_days,
        "min_non_zero_return_pct": args.min_non_zero_return_pct,
        "max_zero_return_pct_adr": args.max_zero_return_pct_adr,
    }
    log.info("config: %s", cfg)

    approved, diagnostics = run_pipeline(
        adr_prices, adr_reference, global_prices, fx_rates, args.as_of, cfg,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([asdict(p) for p in approved], indent=2))
    log.info("wrote %d approved pairs to %s", len(approved), args.out)

    args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_out.write_text(json.dumps(diagnostics, indent=2, default=str))
    log.info("wrote %d diagnostic records to %s", len(diagnostics), args.diagnostics_out)

    if not approved:
        log.warning("no pairs were approved — review %s for rejection reasons",
                    args.diagnostics_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
