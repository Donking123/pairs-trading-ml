#!/usr/bin/env python3
"""
run_walkforward.py
==================

Out-of-sample (walk-forward) evaluation of the Asian ADR
pairs strategy.

Why this exists
---------------
The one-shot pipeline selects pairs (cointegration + liquidity) over a history
and then ``run_backtest.py`` trades that *same* history. Pairs are therefore
chosen with knowledge of the very prices they are then traded on — an in-sample
look-ahead bias that makes the headline ROCE/RUCE optimistic.

This driver removes that bias. For each fold it:

    1. SELECT  pairs using only data in the TRAIN window  [train_start, split]
       (re-runs the existing screening pipeline with ``as_of = split``).
    2. TRADE   those pairs only in the TEST window         (split, test_end]
       (rolling stats still warm up on pre-split history, but no trade is
        opened or closed before ``split`` — so every recorded trade is OOS).
    3. AGGREGATE the test-window trades across folds into a single OOS result.

A single fold ("holdout") is the default. Pass ``--folds N`` for an expanding
walk-forward: the train window grows, the test window rolls forward.

This script imports the screening and backtest logic verbatim from
``datastream/`` — it adds no new strategy logic, only the train/test discipline.

Usage
-----
::

    python datastream/run_walkforward.py \
        --adr-prices    datastream/data/parquet/adr/adr_prices.parquet \
        --adr-reference datastream/data/parquet/adr/adr_reference.parquet \
        --global-prices datastream/data/parquet/global/global_prices.parquet \
        --fx-rates      datastream/data/parquet/fx/fx_rates.parquet \
        --train-start   2010-01-01 \
        --split         2020-12-31 \
        --test-end      2025-12-31 \
        --out-dir       research/output/walkforward_$(date +%Y%m%d_%H%M%S)

For an expanding walk-forward with 3 OOS folds between ``--split`` and
``--test-end``::

    python datastream/run_walkforward.py ... --folds 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("walkforward")


# -----------------------------------------------------------------------------
# Import the existing datastream/ scripts as modules (they live one dir up and
# are not a package, so load them by path).
# -----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASTREAM = _REPO_ROOT / "datastream"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_bt = _load_module("ds_backtest", _DATASTREAM / "run_backtest.py")

# -----------------------------------------------------------------------------
# Pair-selection engine (relocated from the former standalone screening script).
# Asian ADR screening: cointegration (ADF/PP) + liquidity filters + Roll spread,
# the adr_ratio estimators, and the parquet loader. run_fold() drives run_pipeline
# on each train window; review/ scripts import these helpers from this module.
# -----------------------------------------------------------------------------

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


def _adj_prices(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """Return price-return-adjusted prices: adj = raw * adj_factor.
    In this Datastream extract adj_factor is a MULTIPLICATIVE adjustment: at a
    split/redenomination the raw price jumps and adj_factor moves inversely so
    that raw * adj_factor stays continuous across the corporate action. (Must
    match run_backtest._adj_prices.) Falls back to raw prices when adj_factor
    is absent or invalid."""
    p = df[price_col].copy().astype(float)
    if "adj_factor" in df.columns:
        f = pd.to_numeric(df["adj_factor"], errors="coerce")
        f = f.where(f > 0, other=1.0).fillna(1.0)
        p = p * f
    return p


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
    # apply cumadjfactor before pivoting so ratio estimates are based on
    # split-adjusted prices (consistent with what the spread formula will use)
    _adr = adr_prices[adr_prices["ticker"].isin(candidates["adr_ticker"])].copy()
    if "adj_factor" in _adr.columns:
        _f = pd.to_numeric(_adr["adj_factor"], errors="coerce").where(lambda x: x > 0, other=1.0).fillna(1.0)
        _adr = _adr.copy()
        _adr["close"] = _adr["close"].astype(float) * _f.values

    # pivot ADR adjusted closes: index=date, columns=ticker
    adr_wide = _adr.pivot_table(index="marketdate", columns="ticker", values="close", aggfunc="last")
    adr_wide.index = pd.to_datetime(adr_wide.index)

    _loc = global_prices[global_prices["ticker"].isin(candidates["underlying_ticker"])].copy()
    if "adj_factor" in _loc.columns:
        _f = pd.to_numeric(_loc["adj_factor"], errors="coerce").where(lambda x: x > 0, other=1.0).fillna(1.0)
        _loc = _loc.copy()
        _loc["close"] = _loc["close"].astype(float) * _f.values

    # pivot local adjusted closes: index=date, columns=ticker
    loc_wide = _loc.pivot_table(index="marketdate", columns="ticker", values="close", aggfunc="last")
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
    adr_df = adr_prices.set_index("marketdate")
    loc_df = local_prices.set_index("marketdate")
    adr = _adj_prices(adr_df, "close").rename("adr_close").to_frame()
    loc = _adj_prices(loc_df, "close").rename("local_close").to_frame()
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
    adr_df = adr_prices.set_index("marketdate")
    loc_df = local_prices.set_index("marketdate")
    adr = _adj_prices(adr_df, "close").rename("adr_close").to_frame()
    loc = _adj_prices(loc_df, "close").rename("local_close").to_frame()
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
    version — it's a conservative cost proxy.
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

    # include adj_factor in per-pair slices so downstream functions can apply
    # price-return adjustment (cumadjfactor) to remove corporate-action noise
    _adr_cols = ["marketdate", "close"] + (
        ["adj_factor"] if "adj_factor" in adr_prices.columns else [])
    _und_cols = ["marketdate", "close"] + (
        ["adj_factor"] if "adj_factor" in global_prices.columns else [])

    if raw_ratio is None or pd.isna(raw_ratio):
        adr_s = adr_prices[adr_prices["ticker"] == adr_t][_adr_cols]
        und_s = global_prices[global_prices["ticker"] == und_t][_und_cols]
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

    adr_slice = adr_prices[adr_prices["ticker"] == adr_t][_adr_cols]
    und_slice = global_prices[global_prices["ticker"] == und_t][_und_cols]
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


# -----------------------------------------------------------------------------
# Fold construction
# -----------------------------------------------------------------------------
def build_folds(
    train_start: date,
    split: date,
    test_end: date,
    n_folds: int,
) -> list[tuple[date, date, date]]:
    """
    Return a list of ``(train_start, train_end, test_end)`` tuples.

    Single fold (n_folds=1): one holdout — train on [train_start, split],
    test on (split, test_end].

    Expanding walk-forward (n_folds>1): the test span (split, test_end] is cut
    into ``n_folds`` equal calendar slices. Fold k trains on everything up to
    the start of test-slice k and tests on that slice — so the training window
    expands and the OOS window rolls forward without ever overlapping training.
    """
    if test_end <= split:
        raise ValueError("--test-end must be after --split")
    if split <= train_start:
        raise ValueError("--split must be after --train-start")

    if n_folds <= 1:
        return [(train_start, split, test_end)]

    total_days = (test_end - split).days
    step = total_days / n_folds
    folds: list[tuple[date, date, date]] = []
    for k in range(n_folds):
        slice_start = split + timedelta(days=round(k * step))
        slice_end = split + timedelta(days=round((k + 1) * step))
        # train up to the day before this OOS slice begins
        folds.append((train_start, slice_start, slice_end))
    return folds


# -----------------------------------------------------------------------------
# Single fold
# -----------------------------------------------------------------------------
def run_fold(
    fold_idx: int,
    train_start: date,
    train_end: date,
    test_end: date,
    adr_prices: pd.DataFrame,
    adr_reference: pd.DataFrame,
    global_prices: pd.DataFrame,
    fx_rates: pd.DataFrame,
    cfg: dict,
) -> tuple[list, list[dict]]:
    """
    Select pairs on [train_start, train_end], then backtest them on
    (train_end, test_end]. Returns ``(trades, approved_pair_dicts)``.
    """
    log.info(
        "fold %d: TRAIN [%s .. %s]  ->  TEST (%s .. %s]",
        fold_idx, train_start, train_end, train_end, test_end,
    )

    # --- 1. SELECT on the train window only (as_of = train_end) ---------------
    # The screening pipeline already trims each pair's history to <= as_of, so
    # cointegration/liquidity decisions use no post-split data.
    train_adr = adr_prices[adr_prices["marketdate"] >= pd.Timestamp(train_start)]
    train_glb = global_prices[global_prices["marketdate"] >= pd.Timestamp(train_start)]
    approved, _ = run_pipeline(
        train_adr, adr_reference, train_glb, fx_rates, train_end, cfg,
    )
    log.info("fold %d: %d pairs approved on train window", fold_idx, len(approved))
    if not approved:
        return [], []

    approved_dicts = [asdict(p) for p in approved]

    # --- 2. Build the price panel and TRADE the test window only -------------
    pairs = [
        _bt.Pair(
            pair_id=p.pair_id,
            adr_ticker=p.adr_ticker,
            underlying_ticker=p.underlying_ticker,
            underlying_exchange=p.underlying_exchange,
            underlying_currency=p.underlying_currency,
            adr_ratio=float(p.adr_ratio),
            estimation_days=cfg["estimation_days"],
            holding_days=cfg["holding_days"],
            k0=cfg["k0"],
            kc=cfg["kc"],
            roll_spread_adr=float(p.roll_spread_adr),
            roll_spread_local=float(p.roll_spread_local),
        )
        for p in approved
    ]

    # load_panel reads from disk; we already have the frames in memory, so build
    # the panel directly to avoid re-reading. Mirror load_panel's join logic.
    panel = _build_panel(adr_prices, global_prices, fx_rates, pairs)

    trades: list = []
    test_start = train_end + timedelta(days=1)
    for pair in pairs:
        df = panel.get(pair.pair_id)
        if df is None:
            continue
        pair_trades = _bt.backtest_pair(
            pair, df, start=test_start, end=test_end,
            max_overnight_gap_days=cfg.get("max_overnight_gap", 4),
            cost_per_side=cfg.get("cost_bps", 0.0) / 1e4,
        )
        trades.extend(pair_trades)

    log.info("fold %d: %d OOS trades", fold_idx, len(trades))
    return trades, approved_dicts


def _build_panel(
    adr_prices: pd.DataFrame,
    global_prices: pd.DataFrame,
    fx_rates: pd.DataFrame,
    pairs: list,
) -> dict[str, pd.DataFrame]:
    """In-memory equivalent of run_backtest.load_panel (no disk re-read).
    Mirrors load_panel: applies cumadjfactor adjustment and includes local_open_usd."""
    fx_pivot = fx_rates.pivot_table(
        index="date", columns="base_currency", values="mid", aggfunc="last"
    )
    fx_pivot.index = pd.to_datetime(fx_pivot.index).normalize()

    panel: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        adr_raw = adr_prices[adr_prices["ticker"] == pair.adr_ticker].set_index("marketdate")
        loc_raw = global_prices[global_prices["ticker"] == pair.underlying_ticker].set_index("marketdate")
        if adr_raw.empty or loc_raw.empty:
            continue
        if pair.underlying_currency not in fx_pivot.columns:
            continue
        fx_series = fx_pivot[[pair.underlying_currency]].rename(
            columns={pair.underlying_currency: "fx_mid"}
        )

        # apply price-return adjustment (adj = raw * adj_factor)
        adr_slice = _bt._adj_prices(adr_raw, "close").rename("adr_close").to_frame()
        loc_close_adj = _bt._adj_prices(loc_raw, "close").rename("local_close")

        df = (adr_slice
              .join(loc_close_adj, how="inner")
              .join(fx_series, how="inner")
              .dropna())
        if df.empty:
            continue
        df = df.sort_index()
        df["local_close_usd"] = df["local_close"] * df["fx_mid"]
        df["spread"] = df["adr_close"] - df["local_close_usd"] / pair.adr_ratio

        if "open" in loc_raw.columns:
            loc_open_adj = _bt._adj_prices(loc_raw, "open").rename("local_open")
            df = df.join(loc_open_adj, how="left")
            df["local_open_usd"] = df["local_open"] * df["fx_mid"]

        # carry the (cleaned) adjustment factors so backtest_pair can flag trades
        # whose window straddles a corporate action — identical to load_panel.
        # Without these columns spans_adj_change is always False and the OOS path
        # silently keeps the split/redenomination trades that run_backtest drops.
        adr_af = _bt._adj_factor_series(adr_raw)
        loc_af = _bt._adj_factor_series(loc_raw)
        if adr_af is not None:
            adr_af = adr_af[~adr_af.index.duplicated(keep="last")]
            df["adr_adj_factor"] = adr_af.reindex(df.index).ffill().fillna(1.0)
        if loc_af is not None:
            loc_af = loc_af[~loc_af.index.duplicated(keep="last")]
            df["local_adj_factor"] = loc_af.reindex(df.index).ffill().fillna(1.0)

        panel[pair.pair_id] = df
    return panel


# -----------------------------------------------------------------------------
# Per-floor liquidity slicing (folded in from run_walkforward_by_floor.py)
# -----------------------------------------------------------------------------
def fmt_floor(v: int) -> str:
    """Short label for an ADR share-volume floor (e.g. 0->none, 50000->50k, 1e6->1M)."""
    if v == 0:
        return "none"
    if v >= 1_000_000:
        return f"{v // 1_000_000}M"
    if v >= 1_000:
        return f"{v // 1_000}k"
    return str(v)


def train_window_median_volume(
    adr_prices: pd.DataFrame, train_start: date, train_end: date,
) -> pd.Series:
    """Median daily ADR share volume per ticker over the TRAIN window only
    [train_start, train_end] — no post-split data, matching the selection-time
    liquidity screen."""
    win = adr_prices[(adr_prices["marketdate"] >= pd.Timestamp(train_start)) &
                     (adr_prices["marketdate"] <= pd.Timestamp(train_end))]
    return win.groupby("ticker")["volume"].median()


def write_floor_dirs(
    all_trades: list, med_vol: pd.Series, floors: list[int], cfg: dict, out_dir: Path,
) -> list[tuple[int, str, Path, int]]:
    """For each share-volume floor, keep only trades whose ADR's train-window
    median volume exceeds the floor, then materialise that subset's
    ``trades_oos.csv`` + ``summary.json`` into ``out_dir/floor_<label>/`` exactly
    as ``main`` writes the aggregate run — so ``run_walkforward_portfolio.py
    --by-floor`` can build a persistent per-floor equity curve.

    Filtering the single base run is identical to re-screening with the floor:
    ``run_pipeline`` screens each pair independently and ``backtest_pair`` trades
    each pair independently, so dropping other ADRs never changes a surviving
    pair's trades. Returns ``(floor, label, sub_dir, n_trades)`` per written floor.
    """
    written: list[tuple[int, str, Path, int]] = []
    for th in floors:
        if th > 0:
            eligible = set(med_vol[med_vol > th].index)
            trades = [t for t in all_trades if t.adr_ticker in eligible]
        else:
            trades = list(all_trades)
        label = fmt_floor(th)
        sub = out_dir / f"floor_{label}"
        sub.mkdir(parents=True, exist_ok=True)
        if not trades:
            log.warning("floor %s: no trades after filter — skipping", label)
            continue
        report = _bt.build_report(trades, [], cfg)
        pd.DataFrame(report["trades"]).to_csv(sub / "trades_oos.csv", index=False)
        (sub / "summary.json").write_text(json.dumps(report["summary"], default=str))
        log.info("floor %s: %d trades -> %s", label, len(trades), sub / "trades_oos.csv")
        written.append((th, label, sub, len(trades)))
    return written


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward (out-of-sample) backtest of the Asian ADR pairs strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adr-prices", type=Path,
                   default=_DATASTREAM / "data/parquet/adr/adr_prices.parquet")
    p.add_argument("--adr-reference", type=Path,
                   default=_DATASTREAM / "data/parquet/adr/adr_reference.parquet")
    p.add_argument("--global-prices", type=Path,
                   default=_DATASTREAM / "data/parquet/global/global_prices.parquet")
    p.add_argument("--fx-rates", type=Path,
                   default=_DATASTREAM / "data/parquet/fx/fx_rates.parquet")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output dir (default: a timestamped dir under "
                        "datastream/data/walkforward_output, or "
                        "review/output/equity_by_floor when --floors is set)")
    p.add_argument("--floors", type=int, nargs="+", default=None,
                   help="ADR median train-window share-volume floors. When set, "
                        "write a per-floor floor_<label>/{trades_oos.csv,summary.json} "
                        "under --out-dir instead of a single aggregate OOS run; feed "
                        "--out-dir to 'run_walkforward_portfolio.py --by-floor' to "
                        "build the per-floor equity curves (e.g. --floors 0 50000 "
                        "250000 1000000).")

    p.add_argument("--train-start", type=_parse_date, default=_parse_date("2010-01-01"),
                   help="first date used for pair selection")
    p.add_argument("--split", type=_parse_date, default=_parse_date("2020-12-31"),
                   help="train/test boundary: selection uses <= split, trading uses > split")
    p.add_argument("--test-end", type=_parse_date, default=_parse_date("2025-12-31"),
                   help="last date traded out-of-sample")
    p.add_argument("--folds", type=int, default=1,
                   help="1 = single holdout; >1 = expanding walk-forward")

    # strategy params (passed through to screening + backtest)
    # Defaults are the optimum from datastream/run_research_permutations.py
    # (T=90, k0=2.50, kc=0.0, H=90): k0=2.50 and kc=0.0 are the robust signals;
    # T=90 vs T=60 is within noise.
    p.add_argument("--estimation-days", type=int, default=90, help="T")
    p.add_argument("--holding-days", type=int, default=90, help="H")
    p.add_argument("--k0", type=float, default=2.5)
    p.add_argument("--kc", type=float, default=0.0)
    p.add_argument("--cointegration-alpha", type=float, default=0.05)
    p.add_argument("--min-history-days", type=int, default=504)
    p.add_argument("--min-non-zero-return-pct", type=float, default=0.50)
    p.add_argument("--max-zero-return-pct-adr", type=float, default=0.50)
    p.add_argument("--max-overnight-gap", type=int, default=4,
                   dest="max_overnight_gap",
                   help="skip two-bar fills where next joint bar is more than this "
                        "many calendar days after the ADR short day")
    p.add_argument("--cost-bps", type=float, default=0.0, dest="cost_bps",
                   help="incremental transaction cost in bps PER SIDE PER LEG "
                        "(slippage/impact on top of the registry's Roll spread); "
                        "a round-trip pair trade pays this on all 4 fills "
                        "(matches run_backtest.py --cost-bps; previously the OOS "
                        "path applied no incremental cost)")
    p.add_argument("--keep-adj-change", action="store_true", dest="keep_adj_change",
                   help="keep (rather than drop) trades whose window straddles an "
                        "adj_factor change; trades remain flagged via spans_adj_change "
                        "either way (matches run_backtest.py; default: drop)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.out_dir is None:
        if args.floors:
            args.out_dir = _REPO_ROOT / "review" / "output" / "equity_by_floor"
        else:
            args.out_dir = (_DATASTREAM / "data" / "walkforward_output"
                            / f"walkforward_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    adr_prices    = _load_or_die(args.adr_prices, "ADR prices")
    adr_reference = _load_or_die(args.adr_reference, "ADR reference")
    global_prices = _load_or_die(args.global_prices, "Global prices")
    fx_rates      = _load_or_die(args.fx_rates, "FX rates")
    # normalise FX date column once (screening filters by base_currency on the fly)
    fx_rates = fx_rates.copy()
    fx_rates["date"] = pd.to_datetime(fx_rates["date"]).dt.normalize()

    cfg = {
        "estimation_days": args.estimation_days,
        "holding_days": args.holding_days,
        "k0": args.k0,
        "kc": args.kc,
        "cointegration_alpha": args.cointegration_alpha,
        "min_history_days": args.min_history_days,
        "min_non_zero_return_pct": args.min_non_zero_return_pct,
        "max_zero_return_pct_adr": args.max_zero_return_pct_adr,
        "max_overnight_gap": args.max_overnight_gap,
        "cost_bps": args.cost_bps,
    }

    folds = build_folds(args.train_start, args.split, args.test_end, args.folds)
    log.info("running %d fold(s)", len(folds))

    all_trades: list = []
    fold_records: list[dict] = []
    n_flagged_total = 0
    n_dropped_total = 0
    for k, (tr_start, tr_end, te_end) in enumerate(folds):
        trades, approved = run_fold(
            k, tr_start, tr_end, te_end,
            adr_prices, adr_reference, global_prices, fx_rates, cfg,
        )
        # Defensive backstop (mirrors run_backtest.main): drop trades whose window
        # straddles an adj_factor change unless --keep-adj-change is set. Applied
        # per fold so both the fold stats below and the aggregate use the same
        # filtered set.
        n_flagged = sum(t.spans_adj_change for t in trades)
        if args.keep_adj_change:
            kept = trades
        else:
            kept = [t for t in trades if not t.spans_adj_change]
        n_flagged_total += n_flagged
        n_dropped_total += len(trades) - len(kept)

        all_trades.extend(kept)
        closed = [t for t in kept if not t.was_aborted]
        fold_records.append({
            "fold": k,
            "train_start": tr_start.isoformat(),
            "train_end": tr_end.isoformat(),
            "test_end": te_end.isoformat(),
            "n_pairs_selected": len(approved),
            "n_trades": len(kept),
            "n_closed": len(closed),
            "median_ruce_net": (round(float(np.median([t.ruce_net for t in closed])), 6)
                                if closed else None),
            "median_roce_net": (round(float(np.median([t.roce_net for t in closed])), 6)
                                if closed else None),
        })

    # --- per-floor mode: slice the base run by ADR liquidity floor ------------
    # Instead of one aggregate OOS tearsheet, write a filtered trades_oos.csv per
    # share-volume floor so the portfolio engine can persist a per-floor equity
    # curve (folded in from run_walkforward_by_floor.py).
    if args.floors:
        if not all_trades:
            log.error("base walk-forward produced no trades; nothing to slice")
            return 1
        med_vol = train_window_median_volume(adr_prices, args.train_start, args.split)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        written = write_floor_dirs(all_trades, med_vol, args.floors, cfg, args.out_dir)
        log.info("=" * 72)
        log.info("WROTE %d PER-FLOOR TRADE SET(S) under %s", len(written), args.out_dir)
        for _, label, sub, n in written:
            log.info("  floor %-6s %6d trades -> %s", label, n, sub)
        log.info("next: python review/run_walkforward_portfolio.py --by-floor %s",
                 args.out_dir)
        log.info("=" * 72)
        return 0

    # --- aggregate OOS report (reuse the backtest's report builder) -----------
    report = _bt.build_report(all_trades, [], cfg)
    report["summary"]["evaluation"] = "out_of_sample_walkforward"
    report["summary"]["n_folds"] = len(folds)
    report["summary"]["n_adj_change_flagged"] = int(n_flagged_total)
    report["summary"]["n_adj_change_dropped"] = int(n_dropped_total)
    report["folds"] = fold_records
    log.info("adj_factor backstop: flagged %d, %s %d trades spanning a corporate action",
             n_flagged_total,
             "kept" if args.keep_adj_change else "dropped",
             n_dropped_total)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(report["summary"], indent=2, default=str))
    (args.out_dir / "distribution.json").write_text(
        json.dumps(report["distribution"], indent=2, default=str))
    (args.out_dir / "folds.json").write_text(
        json.dumps(fold_records, indent=2, default=str))
    trades_df = pd.DataFrame(report["trades"])
    if not trades_df.empty:
        trades_df.to_csv(args.out_dir / "trades_oos.csv", index=False)

    log.info("=" * 72)
    log.info("WALK-FORWARD (OUT-OF-SAMPLE) SUMMARY")
    log.info("=" * 72)
    for key, val in report["summary"].items():
        if key != "config":
            log.info("  %-22s %s", key, val)
    log.info("=" * 72)
    log.info("PER-FOLD")
    for fr in fold_records:
        log.info("  fold %s: %s pairs, %s trades, median RUCE-net=%s",
                 fr["fold"], fr["n_pairs_selected"], fr["n_trades"],
                 fr["median_ruce_net"])
    log.info("=" * 72)
    log.info("DISTRIBUTION\n%s",
             _bt.format_distribution_table(report["distribution"]))
    log.info("results in %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
