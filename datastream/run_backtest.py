#!/usr/bin/env python3
"""
run_backtest.py
===============

End-to-end backtest of the Hong & Susmel (2013) Asian ADR pairs strategy.

The script implements the entire signal -> sequencer -> ROCE/RUCE pipeline
inline (single-file) so it stays runnable without `src/asian_adr/`. The logic
mirrors the spec:

    Signal layer (per pair):
        spread_t  =  P_ADR,t  -  (P_local,t * FX_t) / adr_ratio
        mu_t      =  mean(spread[t-T : t])
        sigma_t   =  std(spread[t-T : t], ddof=1)
        kappa_open  = mu_t + k0 * sigma_t
        kappa_close = mu_t + kc * sigma_t

    Execution sequencer (next-bar fills; overnight gap modelled):
        Day D U.S. close     -> SHORT_ADR  (ADR sold at close)
        Day D+1 Asia open    -> if spread still > kappa_close: BUY local
                                else: BUY ADR cover (overnight abort)

    Close (any day K):
        spread_K < kappa_close OR days_held >= H  ->  SELL local, BUY ADR cover

    Returns per trade (closed round-trip only):
        local_ret = (loc_close - loc_open) / loc_open      [USD prices]
        adr_ret   = (adr_open  - adr_cover) / adr_open
        ROCE  = local_ret + adr_ret
        RUCE  = local_ret + 2 * adr_ret                     # Reg-T 50% margin

Usage
-----
::

    python scripts/run_backtest.py \
        --adr-prices    data/parquet/adr/adr_prices.parquet \
        --global-prices data/parquet/global/global_prices.parquet \
        --fx-rates      data/parquet/fx/fx_rates.parquet \
        --pairs         config/pairs/asian_adr_pairs.json \
        --out-dir       data/backtest/run_$(date +%Y%m%d_%H%M%S) \
        --k0 2.0 --kc 0.0 --T 60 --H 90
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("backtest")


# -----------------------------------------------------------------------------
# Domain types (inlined; mirror src/asian_adr/core/events.py)
# -----------------------------------------------------------------------------
class HSPosition(str, Enum):
    FLAT = "flat"
    AWAITING_LOCAL = "awaiting_local"   # ADR sold, local leg not yet bought
    OPEN = "open"


class CloseReason(str, Enum):
    CONVERGENCE = "convergence"
    FORCE_CLOSE = "force_close"
    OVERNIGHT_ABORT = "overnight_abort"


@dataclass
class Pair:
    pair_id: str
    adr_ticker: str
    underlying_ticker: str
    underlying_exchange: str
    underlying_currency: str
    adr_ratio: float
    estimation_days: int = 60
    holding_days: int = 90
    k0: float = 2.0
    kc: float = 0.0
    roll_spread_adr: float = 0.0
    roll_spread_local: float = 0.0
    borrow_rate: float = 0.01             # annualised ADR short borrow cost (default 1%)


@dataclass
class Trade:
    pair_id: str
    adr_ticker: str
    underlying_ticker: str
    open_date: date
    close_date: date
    duration_days: int
    adr_open_price: float    # USD
    adr_cover_price: float   # USD
    local_open_price_usd: float
    local_close_price_usd: float
    local_return: float
    adr_return: float
    roce: float
    ruce: float
    roll_cost_pct: float          # bid-ask round-trip (both legs), already baked into effective prices
    borrow_cost: float            # ADR short borrow, prorated by duration
    fx_open: float                # FX rate at local leg entry
    fx_close: float               # FX rate at local leg exit
    fx_pnl: float                 # FX move embedded in local leg (diagnostic; already inside ruce)
    roce_net: float               # roce - borrow (roll already in effective prices)
    ruce_net: float               # ruce - borrow (roll already in effective prices); includes FX
    ruce_net_fx: float            # FX-hedged ruce_net (local leg in local ccy); ruce_net - ruce_net_fx = FX contribution
    close_reason: CloseReason
    was_aborted: bool
    spans_adj_change: bool = False  # trade window straddles an adj_factor change (corporate action); see defensive backstop


@dataclass
class PairState:
    position: HSPosition = HSPosition.FLAT
    spread_history: list[float] = field(default_factory=list)
    # Trade-in-flight tracking
    entry_date: Optional[date] = None     # day D: ADR shorted
    local_open_date: Optional[date] = None  # day D+1: local bought
    adr_open_price: float = 0.0           # USD, filled at day D close
    local_open_price_usd: float = 0.0     # USD, filled at day D+1 close
    fx_open: float = 0.0                  # FX rate at day D+1 (local leg entry)
    entry_idx: Optional[int] = None       # bar index of ADR short (day D)
    local_open_idx: Optional[int] = None  # bar index of local buy (day D+1)


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
def load_pairs(path: Path) -> list[Pair]:
    if not path.exists():
        log.error("pair registry not found: %s — run run_asian_adr_screening.py first", path)
        sys.exit(2)
    raw = json.loads(path.read_text())
    pairs = []
    for r in raw:
        pairs.append(Pair(
            pair_id=r["pair_id"],
            adr_ticker=r["adr_ticker"],
            underlying_ticker=r["underlying_ticker"],
            underlying_exchange=r["underlying_exchange"],
            underlying_currency=r["underlying_currency"],
            adr_ratio=float(r["adr_ratio"]),
            estimation_days=int(r.get("estimation_days", 60)),
            holding_days=int(r.get("holding_days", 90)),
            k0=float(r.get("k0", 2.0)),
            kc=float(r.get("kc", 0.0)),
            roll_spread_adr=float(r.get("roll_spread_adr", 0.0)),
            roll_spread_local=float(r.get("roll_spread_local", 0.0)),
        ))
    log.info("loaded %d pairs from %s", len(pairs), path)
    return pairs


def _adj_prices(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """Return price-return-adjusted prices: adj = raw * adj_factor.
    In this Datastream extract adj_factor is a MULTIPLICATIVE adjustment:
    at a split/redenomination the raw price jumps (e.g. 10x) and adj_factor
    moves inversely (e.g. 2.0 -> 0.2) so that raw * adj_factor stays
    continuous across the corporate action. Dividing (the previous behaviour)
    compounded the jump instead of cancelling it, producing 10x-1000x cliffs
    in the adjusted series and spurious +9000% / -99% trade returns whenever a
    trade straddled a factor-change date. Falls back to raw prices when
    adj_factor is absent, zero, or NaN."""
    p = df[price_col].copy().astype(float)
    if "adj_factor" in df.columns:
        f = pd.to_numeric(df["adj_factor"], errors="coerce")
        f = f.where(f > 0, other=1.0).fillna(1.0)
        p = p * f
    return p


def _adj_factor_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return the cleaned adj_factor (same cleaning as _adj_prices) or None when
    absent. Carried into the panel so the backtest can detect trades whose window
    straddles a corporate action (defensive backstop)."""
    if "adj_factor" not in df.columns:
        return None
    f = pd.to_numeric(df["adj_factor"], errors="coerce")
    return f.where(f > 0, other=1.0).fillna(1.0)


def load_panel(
    adr_prices_path: Path,
    global_prices_path: Path,
    fx_rates_path: Path,
    pairs: list[Pair],
) -> dict[str, pd.DataFrame]:
    """
    Build, for each pair, a tidy DataFrame indexed by date with columns:
        adr_close, local_close, fx_mid, local_close_usd, local_open_usd, spread
    Prices are adjusted for splits and bonus issues via cumadjfactor.
    local_open_usd is the adjusted local open price in USD (for realistic
    Asia-open fills); falls back to close-based fill if open is unavailable.
    Inner-joined across the three sources so each row is fully valid.
    """
    def _read(path: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(path)
        except OSError:
            return pd.read_parquet(path, engine="fastparquet")

    adr = _read(adr_prices_path)
    glb = _read(global_prices_path)
    fx  = _read(fx_rates_path)

    adr["marketdate"] = pd.to_datetime(adr["marketdate"]).dt.normalize()
    glb["marketdate"] = pd.to_datetime(glb["marketdate"]).dt.normalize()
    fx["date"]        = pd.to_datetime(fx["date"]).dt.normalize()

    fx_pivot = fx.pivot_table(
        index="date", columns="base_currency", values="mid", aggfunc="last"
    )

    panel: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        adr_raw = adr[adr["ticker"] == pair.adr_ticker].set_index("marketdate")
        loc_raw = glb[glb["ticker"] == pair.underlying_ticker].set_index("marketdate")
        if adr_raw.empty or loc_raw.empty:
            log.warning("skipping %s: missing price series", pair.pair_id)
            continue
        if pair.underlying_currency not in fx_pivot.columns:
            log.warning("skipping %s: missing FX series for %s",
                        pair.pair_id, pair.underlying_currency)
            continue
        fx_series = fx_pivot[[pair.underlying_currency]].rename(
            columns={pair.underlying_currency: "fx_mid"}
        )

        # apply price-return adjustment (adj = raw * adj_factor) to remove
        # split and bonus-issue discontinuities from the spread signal
        adr_slice = _adj_prices(adr_raw, "close").rename("adr_close").to_frame()
        loc_close_adj = _adj_prices(loc_raw, "close").rename("local_close")

        df = (adr_slice
              .join(loc_close_adj, how="inner")
              .join(fx_series, how="inner")
              .dropna())
        if df.empty:
            log.warning("skipping %s: no overlapping dates", pair.pair_id)
            continue
        df = df.sort_index()
        df["local_close_usd"] = df["local_close"] * df["fx_mid"]
        df["spread"] = df["adr_close"] - df["local_close_usd"] / pair.adr_ratio

        # add adjusted local open price for realistic Asia-open fills (Issue 3)
        if "open" in loc_raw.columns:
            loc_open_adj = _adj_prices(loc_raw, "open").rename("local_open")
            df = df.join(loc_open_adj, how="left")
            df["local_open_usd"] = df["local_open"] * df["fx_mid"]

        # carry the (cleaned) adjustment factors, aligned to the joint calendar,
        # so backtest_pair can flag/skip trades straddling a corporate action.
        adr_af = _adj_factor_series(adr_raw)
        loc_af = _adj_factor_series(loc_raw)
        if adr_af is not None:
            adr_af = adr_af[~adr_af.index.duplicated(keep="last")]
            df["adr_adj_factor"] = adr_af.reindex(df.index).ffill().fillna(1.0)
        if loc_af is not None:
            loc_af = loc_af[~loc_af.index.duplicated(keep="last")]
            df["local_adj_factor"] = loc_af.reindex(df.index).ffill().fillna(1.0)

        panel[pair.pair_id] = df

    log.info("built panel for %d pairs (skipped %d)", len(panel), len(pairs) - len(panel))
    return panel


# -----------------------------------------------------------------------------
# Rolling stats (Welford-equivalent ddof=1)
# -----------------------------------------------------------------------------
def rolling_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu_t, sigma_t) computed over the trailing `window` values (excl. t).

    Indexing convention: mu[i] uses values[i-window:i]; sigma[i] same. The first
    `window` entries are NaN. Matches the spec's stale-leg-safe "as of yesterday"
    interpretation when evaluating spread at index i.
    """
    n = len(values)
    mu = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    if n <= window:
        return mu, sd
    series = pd.Series(values).rolling(window=window, min_periods=window)
    # Use the previous window (shift by 1) so we don't use today's value:
    mu_full = series.mean().shift(1).to_numpy()
    sd_full = series.std(ddof=1).shift(1).to_numpy()
    return mu_full, sd_full


# -----------------------------------------------------------------------------
# Single-pair backtest with overnight gap modelling
# -----------------------------------------------------------------------------
def _factor_spans_change(af: Optional[np.ndarray], a: Optional[int], b: Optional[int]) -> bool:
    """True if the (cleaned) adjustment factor takes more than one value over the
    inclusive bar range [a, b] — i.e. a split/redenomination falls inside the leg's
    hold. Even with the multiplicative adjustment, a factor change can leave a
    residual price discontinuity (imperfect cancellation, data error), so trades
    whose window spans one are flagged and skipped by default (defensive backstop)."""
    if af is None or a is None or b is None or b < a:
        return False
    seg = af[a:b + 1]
    if len(seg) == 0:
        return False
    return bool(np.nanmax(seg) != np.nanmin(seg))


def backtest_pair(
    pair: Pair,
    df: pd.DataFrame,
    start: Optional[date] = None,
    end: Optional[date] = None,
    max_overnight_gap_days: int = 4,
    cost_per_side: float = 0.0,
) -> list[Trade]:
    """
    Single-pair event loop with realistic execution modelling.

    Two-bar fill (points 1-2):
        Day D U.S. close  -> SHORT ADR at adr_close[D]. Roll cost (ADR half)
                             deducted upfront from the ADR leg.
        Day D+1 Asia open -> If spread still above kappa_close: BUY local at
                             local_close_usd[D+1]. Roll cost (local half) deducted
                             upfront. FX rate recorded. Position = OPEN.
                             Else: overnight abort — cover ADR at adr_close[D+1].

    Borrow cost (point 3):
        Annualised rate ``pair.borrow_rate`` prorated by duration_days/365.
        Applied to closed round-trips only (not aborts).

    FX P&L (point 4):
        Local prices are stored in USD, so the FX move is ALREADY inside the
        local-leg return (and hence inside roce/ruce/ruce_net). ``fx_pnl`` is
        reported only as a diagnostic of that embedded component. ``ruce_net_fx``
        is the FX-HEDGED variant (local leg re-priced in local currency), so
        ``ruce_net - ruce_net_fx`` isolates the FX contribution.

    ``start`` / ``end`` restrict the trading window. Rolling stats are computed
    over the FULL supplied df so the bars at/after ``start`` already have warmed-up
    mu/sigma from pre-start history (walk-forward discipline); only entry into new
    trades is gated to the [start, end] window, never the warm-up data itself.
    """
    trades: list[Trade] = []
    # Clip only the END of the data (no future bars), but keep ALL pre-start
    # history so rolling stats are warm by the time trading is allowed to start.
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    if len(df) < pair.estimation_days + 5:
        return trades

    spreads  = df["spread"].to_numpy()
    adr_close = df["adr_close"].to_numpy()
    local_usd = df["local_close_usd"].to_numpy()
    # open prices for Asia-open fills (Issue 3); None when parquet lacks open column
    local_open_usd = (df["local_open_usd"].to_numpy()
                      if "local_open_usd" in df.columns else None)
    fx_rates  = df["fx_mid"].to_numpy()
    # adjustment-factor paths for the defensive backstop; None when absent
    adr_af = df["adr_adj_factor"].to_numpy() if "adr_adj_factor" in df.columns else None
    loc_af = df["local_adj_factor"].to_numpy() if "local_adj_factor" in df.columns else None
    dates = df.index.to_pydatetime() if hasattr(df.index, "to_pydatetime") else df.index
    if isinstance(dates, np.ndarray):
        dates = [pd.Timestamp(d).date() for d in dates]
    else:
        dates = [d.date() if hasattr(d, "date") else d for d in dates]

    # Trading window: entries are only opened on bars at/after `start`. Warm-up
    # bars before `start` still feed the rolling stats above.
    start_date = pd.Timestamp(start).date() if start is not None else None

    mu, sd = rolling_mean_std(spreads, pair.estimation_days)
    k0 = pair.k0
    kc = pair.kc

    # Per-side transaction cost on each leg = half the Roll bid/ask spread (split
    # evenly between entry and exit) PLUS an optional incremental cost
    # (slippage/impact/borrow-scarcity) from ``cost_per_side``. Applied with the
    # correct sign for each leg below: a SHORT sells low at entry and covers high
    # at exit; a LONG buys high at entry and sells low at exit — every fill moves
    # the effective price against the trade.
    half_roll_adr   = float(pair.roll_spread_adr)   / 2.0
    half_roll_local = float(pair.roll_spread_local) / 2.0
    adr_side        = half_roll_adr   + float(cost_per_side)   # ADR per-side cost
    loc_side        = half_roll_local + float(cost_per_side)   # local per-side cost
    roll_cost_pct   = float(pair.roll_spread_adr + pair.roll_spread_local)

    state = PairState()
    n = len(df)

    i = 0
    while i < n:
        if np.isnan(mu[i]) or np.isnan(sd[i]) or sd[i] == 0:
            i += 1
            continue

        kappa_open  = mu[i] + k0 * sd[i]
        kappa_close = mu[i] + kc * sd[i]
        spread_i = spreads[i]

        # ── FLAT: look for entry ──────────────────────────────────────────────
        if state.position == HSPosition.FLAT:
            # Only open new trades inside the trading window; pre-start bars are
            # warm-up only. Already-open trades are still allowed to close below.
            if start_date is not None and dates[i] < start_date:
                i += 1
                continue
            if spread_i > kappa_open:
                # Day D: SHORT ADR at close. A short SELLS, so the cost LOWERS the
                # effective entry price (sell at the bid, minus incremental cost).
                adr_open_eff = float(adr_close[i]) * (1.0 - adr_side)
                state.position      = HSPosition.AWAITING_LOCAL
                state.adr_open_price = adr_open_eff
                state.entry_date    = dates[i]
                state.entry_idx     = i

                # Day D+1: check if local leg can be entered
                j = i + 1
                if j >= n:
                    state.position = HSPosition.FLAT
                    i += 1
                    continue

                # Issue 4: skip entries when the inner-join "next bar" is separated
                # from day D by more than max_overnight_gap_days calendar days.
                # Gaps this large (e.g. Asian holiday weeks) violate the overnight
                # timing assumption and leave the ADR short unhedged for too long.
                if (dates[j] - dates[i]).days > max_overnight_gap_days:
                    log.debug(
                        "skipping entry on %s [%s]: next joint bar is %d days later "
                        "(holiday gap > %d)",
                        dates[i], pair.pair_id,
                        (dates[j] - dates[i]).days, max_overnight_gap_days,
                    )
                    state.position = HSPosition.FLAT
                    i += 1
                    continue

                spread_d1 = spreads[j]
                if spread_d1 > kappa_close:
                    # Issue 3: BUY local at D+1 OPEN (realistic Asia-open fill).
                    # Fall back to close if open prices are unavailable or NaN.
                    _loc_px = (local_open_usd[j]
                               if local_open_usd is not None
                               and not np.isnan(local_open_usd[j])
                               else local_usd[j])
                    loc_open_eff = float(_loc_px) * (1.0 + loc_side)
                    state.position             = HSPosition.OPEN
                    state.local_open_price_usd = loc_open_eff
                    state.local_open_date      = dates[j]
                    state.local_open_idx       = j
                    state.fx_open              = float(fx_rates[j])
                    i = j
                else:
                    # Overnight abort: cover ADR at D+1 close. Covering BUYS, so the
                    # cost RAISES the effective cover price (buy at the ask + cost).
                    adr_cover_eff = float(adr_close[j]) * (1.0 + adr_side)
                    adr_ret = (state.adr_open_price - adr_cover_eff) / state.adr_open_price
                    roce = adr_ret
                    ruce = 2.0 * adr_ret
                    trades.append(Trade(
                        pair_id=pair.pair_id,
                        adr_ticker=pair.adr_ticker,
                        underlying_ticker=pair.underlying_ticker,
                        open_date=state.entry_date,
                        close_date=dates[j],
                        duration_days=1,
                        adr_open_price=state.adr_open_price,
                        adr_cover_price=adr_cover_eff,
                        local_open_price_usd=0.0,
                        local_close_price_usd=0.0,
                        local_return=0.0,
                        adr_return=adr_ret,
                        roce=roce,
                        ruce=ruce,
                        roll_cost_pct=roll_cost_pct,
                        borrow_cost=0.0,
                        fx_open=0.0,
                        fx_close=0.0,
                        fx_pnl=0.0,
                        roce_net=roce,   # roll already in effective prices
                        ruce_net=ruce,
                        ruce_net_fx=ruce,
                        close_reason=CloseReason.OVERNIGHT_ABORT,
                        was_aborted=True,
                        # only the ADR leg trades on an abort; check its [i, j] span
                        spans_adj_change=_factor_spans_change(adr_af, i, j),
                    ))
                    state = PairState()
                    i = j + 1
                    continue

        # ── OPEN: look for exit ───────────────────────────────────────────────
        elif state.position == HSPosition.OPEN:
            # Duration is measured from entry_date (day D, when the ADR short and
            # its borrow begin) so it is consistent with the reported open_date and
            # charges borrow over the whole period the short is held.
            days_held = (dates[i] - state.entry_date).days
            should_close = (spread_i < kappa_close) or (days_held >= pair.holding_days)
            if should_close:
                reason = (CloseReason.FORCE_CLOSE if days_held >= pair.holding_days
                          else CloseReason.CONVERGENCE)

                # Exit fills mirror the entry's two-bar, time-zone-aware
                # sequencing (Issue 3, exit side):
                #   Day i   U.S. close -> COVER ADR at adr_close[i]. The exit signal
                #                         is known at the U.S. close and the ADR is
                #                         tradeable market-on-close, exactly as the
                #                         entry shorts it at adr_close[D].
                #   Day i+1 Asia open  -> SELL local at local_open_usd[i+1]. The
                #                         Asian market for day i closed ~13h BEFORE
                #                         the exit signal existed, so the local leg
                #                         can only be unwound at the next Asia open
                #                         (symmetric with the D+1 entry fill). Using
                #                         local_close[i] here was look-ahead: it sold
                #                         the local at a price that printed before the
                #                         signal. Falls back to the local close if the
                #                         open is missing, or to day i if i is the last
                #                         bar. Roll costs deducted upfront as before.
                adr_cover_eff = float(adr_close[i]) * (1.0 + adr_side)
                j_exit = i + 1
                if j_exit < n:
                    _loc_exit_px = (local_open_usd[j_exit]
                                    if local_open_usd is not None
                                    and not np.isnan(local_open_usd[j_exit])
                                    else local_usd[j_exit])
                    fx_close_val = float(fx_rates[j_exit])
                    loc_exit_idx = j_exit
                else:
                    _loc_exit_px = float(local_usd[i])
                    fx_close_val = float(fx_rates[i])
                    loc_exit_idx = i
                loc_close_eff = float(_loc_exit_px) * (1.0 - loc_side)

                local_ret = (loc_close_eff - state.local_open_price_usd) / state.local_open_price_usd
                adr_ret   = (state.adr_open_price - adr_cover_eff) / state.adr_open_price
                roce = local_ret + adr_ret
                ruce = local_ret + 2.0 * adr_ret

                # Borrow cost: annualised rate prorated by holding period
                borrow_cost = pair.borrow_rate * (days_held / 365.0)

                # FX move embedded in the (USD) local leg, reported for transparency:
                # local prices are USD (local_close_usd = local_close * fx_mid), so
                # local_ret ALREADY contains this — it is not added on top.
                fx_pnl = (fx_close_val - state.fx_open) / state.fx_open if state.fx_open > 0 else 0.0

                roce_net = roce - borrow_cost   # roll already in effective prices
                ruce_net = ruce - borrow_cost

                # FX-hedged RUCE: re-price the long local leg in LOCAL currency
                # (strip the FX move) so (ruce_net − ruce_net_fx) isolates the FX
                # contribution. Divide the effective USD prices back out by their
                # entry/exit FX rates.
                if state.fx_open > 0 and fx_close_val > 0:
                    loc_open_local  = state.local_open_price_usd / state.fx_open
                    loc_close_local = loc_close_eff / fx_close_val
                    local_ret_ex_fx = (loc_close_local - loc_open_local) / loc_open_local
                    ruce_ex_fx      = local_ret_ex_fx + 2.0 * adr_ret
                    ruce_net_fx     = ruce_ex_fx - borrow_cost
                else:
                    ruce_net_fx = ruce_net

                trades.append(Trade(
                    pair_id=pair.pair_id,
                    adr_ticker=pair.adr_ticker,
                    underlying_ticker=pair.underlying_ticker,
                    open_date=state.entry_date,
                    close_date=dates[i],
                    duration_days=days_held,
                    adr_open_price=state.adr_open_price,
                    adr_cover_price=adr_cover_eff,
                    local_open_price_usd=state.local_open_price_usd,
                    local_close_price_usd=loc_close_eff,
                    local_return=local_ret,
                    adr_return=adr_ret,
                    roce=roce,
                    ruce=ruce,
                    roll_cost_pct=roll_cost_pct,
                    borrow_cost=round(borrow_cost, 6),
                    fx_open=state.fx_open,
                    fx_close=fx_close_val,
                    fx_pnl=round(fx_pnl, 6),
                    roce_net=roce_net,
                    ruce_net=ruce_net,
                    ruce_net_fx=ruce_net_fx,
                    close_reason=reason,
                    was_aborted=False,
                    # flag if a corporate action falls inside either leg's hold:
                    # ADR over [entry_idx, i], local over [local_open_idx, loc_exit_idx]
                    spans_adj_change=(
                        _factor_spans_change(adr_af, state.entry_idx, i)
                        or _factor_spans_change(loc_af, state.local_open_idx, loc_exit_idx)
                    ),
                ))
                state = PairState()
        i += 1

    return trades


# -----------------------------------------------------------------------------
# Aggregation & paper Table 7-B-style distribution
# -----------------------------------------------------------------------------
DIST_PCTILES = [("mean", None), ("std", None), ("max", 1.0),
                ("p90", 0.90), ("p75", 0.75), ("median", 0.50),
                ("p25", 0.25), ("p10", 0.10), ("min", 0.0)]


def distribution_row(name: str, values: np.ndarray) -> dict:
    """Compute the paper Table 7-B distribution stats for a metric."""
    if len(values) == 0:
        return {"metric": name} | {k: float("nan") for k, _ in DIST_PCTILES}
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"metric": name} | {k: float("nan") for k, _ in DIST_PCTILES}
    row: dict = {"metric": name}
    for label, q in DIST_PCTILES:
        if label == "mean":
            row[label] = float(np.mean(arr))
        elif label == "std":
            row[label] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        elif label == "max":
            row[label] = float(np.max(arr))
        elif label == "min":
            row[label] = float(np.min(arr))
        else:
            row[label] = float(np.quantile(arr, q))
    return row


def build_report(trades: list[Trade], pairs: list[Pair], cfg: dict) -> dict:
    if not trades:
        return {"summary": {"n_trades": 0}, "trades": [], "distribution": []}

    df = pd.DataFrame([asdict(t) for t in trades])
    df["close_reason"] = df["close_reason"].astype(str)

    distribution = []
    for name, col in [("ROCE per trade", "roce"),
                      ("RUCE per trade", "ruce"),
                      ("ROCE net per trade", "roce_net"),
                      ("RUCE net per trade", "ruce_net"),
                      ("RUCE net+FX per trade", "ruce_net_fx"),
                      ("Duration (days)", "duration_days"),
                      ("Roll cost (round trip)", "roll_cost_pct"),
                      ("Borrow cost", "borrow_cost"),
                      ("FX P&L", "fx_pnl")]:
        if col in df.columns:
            distribution.append(distribution_row(name, df[col].to_numpy()))

    closed = df[~df["was_aborted"]]
    aborted_count = int(df["was_aborted"].sum())
    force_close_count = int((df["close_reason"] == CloseReason.FORCE_CLOSE.value).sum())

    summary = {
        "n_trades":         int(len(df)),
        "n_closed":         int(len(closed)),
        "n_overnight_abort": aborted_count,
        "n_force_close":    force_close_count,
        "abort_rate":       round(aborted_count / max(len(df), 1), 4),
        "median_roce":      round(float(closed["roce"].median()), 6) if len(closed) else None,
        "median_ruce":      round(float(closed["ruce"].median()), 6) if len(closed) else None,
        "median_roce_net":      round(float(closed["roce_net"].median()), 6) if len(closed) else None,
        "median_ruce_net":      round(float(closed["ruce_net"].median()), 6) if len(closed) else None,
        "median_ruce_net_fx":   round(float(closed["ruce_net_fx"].median()), 6) if len(closed) else None,
        "median_borrow_cost":   round(float(closed["borrow_cost"].median()), 6) if len(closed) else None,
        "median_fx_pnl":        round(float(closed["fx_pnl"].median()), 6) if len(closed) else None,
        "median_duration":  round(float(closed["duration_days"].median()), 2) if len(closed) else None,
        "n_pairs_traded":   int(df["pair_id"].nunique()),
        "n_pairs_loaded":   len(pairs),
        "config":           cfg,
    }

    return {
        "summary": summary,
        "distribution": distribution,
        "trades": df.to_dict(orient="records"),
    }


def format_distribution_table(dist: list[dict]) -> str:
    if not dist:
        return "(no trades)"
    cols = ["metric"] + [k for k, _ in DIST_PCTILES]
    widths = {c: max(len(c), 12) for c in cols}
    lines = [" ".join(c.ljust(widths[c]) for c in cols),
             " ".join("-" * widths[c] for c in cols)]
    for row in dist:
        cells = []
        for c in cols:
            val = row.get(c)
            if c == "metric":
                cells.append(str(val).ljust(widths[c]))
            elif val is None or (isinstance(val, float) and np.isnan(val)):
                cells.append("nan".ljust(widths[c]))
            else:
                cells.append(f"{val:.4f}".ljust(widths[c]))
        lines.append(" ".join(cells))
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest the Hong & Susmel Asian ADR pairs strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adr-prices",    type=Path, default=Path("datastream/data/parquet/adr/adr_prices.parquet"))
    p.add_argument("--global-prices", type=Path, default=Path("datastream/data/parquet/global/global_prices.parquet"))
    p.add_argument("--fx-rates",      type=Path, default=Path("datastream/data/parquet/fx/fx_rates.parquet"))
    p.add_argument("--pairs",         type=Path, default=Path("config/pairs/asian_adr_pairs.json"))
    p.add_argument("--out-dir",       type=Path,
                   default=Path("datastream/data/backtest")
                            / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    p.add_argument("--T",  dest="T",  type=int, default=60, help="estimation window (days)")
    p.add_argument("--H",  dest="H",  type=int, default=90, help="max holding period (days)")
    p.add_argument("--k0", type=float, default=2.0, help="entry multiplier")
    p.add_argument("--kc", type=float, default=0.0, help="exit multiplier")
    p.add_argument("--max-overnight-gap", type=int, default=4,
                   dest="max_overnight_gap",
                   help="skip two-bar fills where next joint bar is more than this "
                        "many calendar days after the ADR short day (holiday-gap filter)")
    p.add_argument("--keep-adj-change", action="store_true", dest="keep_adj_change",
                   help="keep (rather than drop) trades whose window straddles an "
                        "adj_factor change; trades remain flagged via spans_adj_change "
                        "either way (defensive backstop; default: drop)")
    p.add_argument("--cost-bps", type=float, default=0.0, dest="cost_bps",
                   help="incremental transaction cost in bps PER SIDE PER LEG "
                        "(slippage/impact on top of the registry's Roll spread); "
                        "a round-trip pair trade pays this on all 4 fills")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    pairs = load_pairs(args.pairs)
    if not pairs:
        log.error("no pairs in registry; nothing to backtest")
        return 3

    # CLI overrides
    for p in pairs:
        p.estimation_days = args.T
        p.holding_days = args.H
        p.k0 = args.k0
        p.kc = args.kc

    cfg = {"T": args.T, "H": args.H, "k0": args.k0, "kc": args.kc,
           "max_overnight_gap": args.max_overnight_gap,
           "keep_adj_change": args.keep_adj_change,
           "cost_bps_per_side": args.cost_bps}
    log.info("backtest config: %s", cfg)

    panel = load_panel(args.adr_prices, args.global_prices, args.fx_rates, pairs)
    if not panel:
        log.error("price panel empty; abort")
        return 4

    all_trades: list[Trade] = []
    for pair in pairs:
        df = panel.get(pair.pair_id)
        if df is None:
            continue
        trades = backtest_pair(pair, df,
                               max_overnight_gap_days=args.max_overnight_gap,
                               cost_per_side=args.cost_bps / 1e4)
        all_trades.extend(trades)
        log.info("[%s] %d trades, %d bars", pair.pair_id, len(trades), len(df))

    # Defensive backstop: drop trades whose window straddles an adj_factor change
    # (corporate action) unless --keep-adj-change is set. They stay flagged via
    # spans_adj_change either way.
    n_flagged = sum(t.spans_adj_change for t in all_trades)
    if args.keep_adj_change:
        kept_trades, n_dropped = all_trades, 0
        log.info("adj_factor backstop: keeping %d flagged trades (--keep-adj-change)", n_flagged)
    else:
        kept_trades = [t for t in all_trades if not t.spans_adj_change]
        n_dropped = len(all_trades) - len(kept_trades)
        log.info("adj_factor backstop: dropped %d trades spanning a corporate action", n_dropped)

    report = build_report(kept_trades, pairs, cfg)
    report["summary"]["n_adj_change_flagged"] = int(n_flagged)
    report["summary"]["n_adj_change_dropped"] = int(n_dropped)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(report["summary"], indent=2, default=str))
    (args.out_dir / "distribution.json").write_text(
        json.dumps(report["distribution"], indent=2, default=str)
    )
    trades_df = pd.DataFrame(report["trades"])
    if not trades_df.empty:
        trades_df.to_csv(args.out_dir / "trades.csv", index=False)
        trades_df.to_parquet(args.out_dir / "trades.parquet", index=False)

    table = format_distribution_table(report["distribution"])
    log.info("=" * 72)
    log.info("BACKTEST SUMMARY")
    log.info("=" * 72)
    for k, v in report["summary"].items():
        if k != "config":
            log.info("  %-22s %s", k, v)
    log.info("=" * 72)
    log.info("DISTRIBUTION (paper Table 7-B format)\n%s", table)
    log.info("=" * 72)
    log.info("results in %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
