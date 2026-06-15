"""
Rolling monthly backtest  [Pipeline Stage 4].

Ties the whole pipeline together. For each month from 2003-01 to 2023-12:
  1. formation window (3-yr panel ending the last day before the trading month)
  2. distances + clustering → candidate pairs
  3. fit_hedge_ratio per pair (γ frozen for the trading month)
  4. simulate every pair through the trading month (~21 days):
       - entry  : |z| >= entry_sigma and no position
       - exit   : z crosses zero (zero-cross)  [paper-faithful]
       - stop   : |z| >= stop_sigma  [realism variant only]
       - delisting: close at CRSP dlret (or code-dependent fallback)
       - month-end: force-close any remaining positions
  5. Aggregate: portfolio daily return = mean across currently-open pairs;
                monthly return = compound of daily returns.

Two variants run from the same code path:
  * core      : no costs, no stop-loss        →  match paper Sharpe target 0.88
  * realism   : bid/ask spread + 35 bps borrow + 3.5σ stop  →  "could we run it?"

Position convention (paper / Gatev-Goetzmann):
  * Equal-dollar long/short at entry: $0.50 long leg, $0.50 short leg.
  * γ enters only via the spread & z-score signal; sizing is dollar-equal.
  * Daily pair return = position * 0.5 * (ret_a - ret_b) on close-to-close prices.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.clustering import (
    cluster_hdbscan,
    cluster_hierarchical,
    cluster_optics,
    clusters_to_pairs,
)
from src.cointegration import filter_cointegrated_pairs
from src.costs import (
    RealismConfig,
    borrow_cost_daily,
    build_spread_panel,
    transaction_cost,
)
from src.config import (
    COINTEGRATION_P_THRESHOLD,
    DATA_DIR,
    ENTRY_THRESHOLD,
    EXIT_THRESHOLD,
    FORMATION_YEARS,
    HALF_LIFE_BOUNDS,
    HIER_LINKAGE,
    HIER_QUANTILE,
    MAX_CARRY_MONTHS,
    OPTICS_MIN_CLUSTER_SIZE,
    OPTICS_MIN_SAMPLES,
    OPTICS_XI,
    OPTICS_XI_FACTOR,
    OPTICS_XI_PC,
    RIDGE_ALPHA,
    STOP_LOSS_SIGMA,
    ZSCORE_WINDOW_MONTHS,
)
from src.distances import factor_beta_distance, pc_distance, ssd_distance
from src.factors import build_factor_panel
from src.panel import (
    formation_window_panel,
    load_crsp_daily,
    load_market_returns,
    load_sp500_constituents,
    siccd_lookup,
)
from src.spread import (
    fit_hedge_ratio,
    fit_hedge_ratio_rlm,
    rolling_zscore,
    spread_series,
)


ExitReason = Literal["reversion", "stop_loss", "force_close", "delisting"]


# ────────────────────────────────────────────────────────────────────────────────
# data classes
# ────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Trade:
    """One round-trip trade record."""
    permno_a: int
    permno_b: int
    direction: int                      # +1 long spread, -1 short spread
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_z: float
    exit_z: float
    round_trip_return: float            # pair P&L over the trade
    exit_reason: ExitReason


@dataclass
class CarryState:
    """A position held open ACROSS a month boundary (Phase 5 carry-over).

    The position keeps the γ it was OPENED with (`gamma_frozen`). Equal-dollar sizing
    makes γ irrelevant to P&L, so freezing it just keeps the z-score exit signal
    consistent with the spread actually entered, and keeps the position byte-identical
    under the lookahead audit.

    `cum_pnl` is the gross (borrow-adjusted, pre-transaction-cost) P&L accumulated over
    the position's life so far. It feeds the eventual `Trade.round_trip_return` ONLY —
    portfolio P&L is computed independently from each month's daily return series, so
    carry bookkeeping can never move the headline Sharpe.

    `months_carried` = number of month boundaries this position has already crossed
    (0 = still in its origin month). A position may be open during at most
    MAX_CARRY_MONTHS distinct months.
    """
    permno_a: int
    permno_b: int
    direction: int
    gamma_frozen: float
    entry_date: pd.Timestamp
    entry_z: float
    cum_pnl: float
    months_carried: int


@dataclass
class MonthResult:
    """Aggregated outputs for one month."""
    month_end: pd.Timestamp
    n_candidate_pairs: int
    n_pairs_traded: int                 # pairs that opened at least one trade
    n_trades: int                       # total round-trips
    avg_pairs_open: float               # avg active pairs per trading day
    monthly_return: float               # compound of daily portfolio returns
    daily_returns: pd.Series            # per trading day in this month
    trades: list[Trade] = field(default_factory=list)
    # Phase 5: positions still open at month-end, keyed by (permno_a, permno_b), to be
    # carried into the next month. Empty whenever carry-over is off.
    carry_out: dict = field(default_factory=dict)


def _carry_to_forced_trade(
    cs: CarryState, exit_date: pd.Timestamp, reason: ExitReason = "force_close"
) -> Trade:
    """Close a carried position 'on paper' (it dropped out of the candidate set, lost a
    leg, or the backtest ended). The position was already marked through its last
    simulated month-end, so this adds no portfolio P&L — only a diagnostic Trade record.
    """
    return Trade(
        permno_a=cs.permno_a, permno_b=cs.permno_b, direction=cs.direction,
        entry_date=cs.entry_date, exit_date=exit_date,
        entry_z=cs.entry_z, exit_z=0.0,
        round_trip_return=cs.cum_pnl, exit_reason=reason,
    )


# ────────────────────────────────────────────────────────────────────────────────
# delisting handling — Option B (code-dependent fallback)
# ────────────────────────────────────────────────────────────────────────────────


def _delisting_fallback_return(dlstcd: int) -> float:
    """When CRSP `dlret` is missing, infer a reasonable return from the code.

    Code ranges (CRSP convention):
        200–299  M&A / acquisition          →  0%   (don't penalise — usually neutral/positive)
        300–399  liquidation                →  -30% (Shumway 1997 conservative default)
        400–499  voluntary drop / sponsor   →  -30%
        500–599  exchange-related (OTC etc.)→  -5%
        600+     other / unknown             →  0%   (no information → neutral)
    """
    if 200 <= dlstcd < 300:
        return 0.0
    if 300 <= dlstcd < 500:
        return -0.30
    if 500 <= dlstcd < 600:
        return -0.05
    return 0.0


def _delisting_fallback_return_fixed(dlstcd: int) -> float:
    """Phase 6 (D6.2) — fallback map with CORRECT CRSP code semantics.

    The original map shifted the CRSP buckets by one: it labelled the 300s
    "liquidation" (they are *exchanges* — the issue is swapped in a reorg/merger,
    roughly neutral) and the 500s "exchange-related OTC" at -5% (they are *dropped
    by the exchange* — the bankruptcy / insufficient-capital / price-too-low class
    where dlret is most often missing and Shumway (1997) estimates ≈ -30%).

    Corrected ranges (CRSP convention):
        200–299  mergers / acquisitions        →   0%
        300–399  exchanges (issue swapped)     →   0%
        400–499  liquidations                  →  -30%
        500      dropped, reason unavailable   →  -30%  (Shumway; also the v2-pull
                                                          sentinel for missing dlret)
        501–519  moved to another exchange     →   0%   (neutral relisting)
        520–584  dropped, performance-related  →  -30%  (Shumway 1997)
        other    no information                →   0%
    """
    if 200 <= dlstcd < 400:
        return 0.0
    if 400 <= dlstcd < 500:
        return -0.30
    if dlstcd == 500 or 520 <= dlstcd <= 584:
        return -0.30
    return 0.0


def _delisting_day_return(market_ret: float, dlret: float, delisting_fix: bool) -> float:
    """Effective return for the delisting day (Phase 6 D6.2).

    Default (delisting_fix=False, Phases 1–5): OVERWRITE the day's close-to-close
    return with dlret. CRSP measures dlret AFTER the last close, so the correct
    treatment compounds the two: (1+ret)·(1+dlret)−1 — enabled by delisting_fix.
    Falls back to dlret alone when the market return is unavailable (NaN price).
    """
    if not delisting_fix:
        return dlret
    if np.isfinite(market_ret):
        return (1.0 + market_ret) * (1.0 + dlret) - 1.0
    return dlret


def load_delisting(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "delisting.parquet")


def get_delisting_returns(
    permnos: list[int],
    delisting_df: pd.DataFrame,
    fixed_codes: bool = False,
) -> dict[int, tuple[pd.Timestamp, float]]:
    """For each permno, return (delisting_date, return) if it delists; else absent.

    Uses dlret if present, otherwise the code-dependent fallback.
    `fixed_codes=True` (Phase 6, D6.2) uses the corrected CRSP code map.
    """
    fallback = _delisting_fallback_return_fixed if fixed_codes else _delisting_fallback_return
    out: dict[int, tuple[pd.Timestamp, float]] = {}
    sub = delisting_df.loc[delisting_df["permno"].isin(permnos)]
    for _, row in sub.iterrows():
        if pd.isna(row["dlstdt"]):
            continue
        dlret = row.get("dlret")
        if pd.notna(dlret):
            ret = float(dlret)
        else:
            ret = fallback(int(row["dlstcd"]))
        out[int(row["permno"])] = (pd.Timestamp(row["dlstdt"]), ret)
    return out


# ────────────────────────────────────────────────────────────────────────────────
# IBES earnings blackout helper
# ────────────────────────────────────────────────────────────────────────────────


def load_ibes_earnings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load pre-fetched IBES earnings dates (ticker + earnings_date columns)."""
    return pd.read_parquet(data_dir / "ibes_earnings.parquet")


def _load_permno_ticker_map(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load permno↔ticker mapping from crsp_spx_membership (has both columns)."""
    mem = pd.read_parquet(data_dir / "crsp_spx_membership.parquet")
    return mem[["permno", "ticker"]].dropna().drop_duplicates()


def _build_permno_blackout(
    ibes_df: pd.DataFrame,
    date_index: pd.DatetimeIndex,
    n_days: int,
    data_dir: Path = DATA_DIR,
) -> dict[int, frozenset]:
    """Build {permno: frozenset[Timestamp]} of earnings blackout dates.

    Joins IBES ticker-keyed earnings dates to CRSP permnos via the S&P 500
    constituent membership table.  Blocks a symmetric ±n_days calendar-day
    window around each announcement and returns only dates present in date_index.
    """
    mapping = _load_permno_ticker_map(data_dir)
    merged = ibes_df.merge(mapping, on="ticker", how="inner")
    if merged.empty:
        return {}

    date_arr = date_index.to_numpy()  # datetime64[ns]
    blocked: dict[int, frozenset] = {}
    for permno, grp in merged.groupby("permno"):
        ann_arr = pd.DatetimeIndex(grp["earnings_date"]).to_numpy()
        diff_days = (
            np.abs(
                date_arr[:, None].astype("int64") - ann_arr[None, :].astype("int64")
            )
            / 86_400_000_000_000  # nanoseconds → days
        )
        mask = (diff_days <= n_days).any(axis=1)
        dates = frozenset(date_index[mask])
        if dates:
            blocked[int(permno)] = dates
    return blocked


# ────────────────────────────────────────────────────────────────────────────────
# HMM regime helpers
# ────────────────────────────────────────────────────────────────────────────────


def load_regime_scale(data_dir: Path = DATA_DIR) -> pd.Series:
    """Load pre-computed HMM regime scale (date-indexed, values in [0, 1])."""
    df = pd.read_parquet(data_dir / "hmm_regimes.parquet")
    return df["regime_scale"]


# ────────────────────────────────────────────────────────────────────────────────
# pair simulation
# ────────────────────────────────────────────────────────────────────────────────


def _equal_dollar_daily_return(position: int, ret_a: float, ret_b: float) -> float:
    """Daily P&L for an equal-dollar long-short pair held at `position`.

    position = +1 (long spread):  long $0.50 A, short $0.50 B  →  0.5*(ret_a - ret_b)
    position = -1 (short spread): short $0.50 A, long $0.50 B  →  0.5*(ret_b - ret_a)
    """
    return position * 0.5 * (ret_a - ret_b)


def simulate_pair_in_month(
    permno_a: int,
    permno_b: int,
    gamma: float,
    panel: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    entry_sigma: float,
    exit_sigma: float,
    stop_sigma: float | None,
    zscore_window: int,
    delisting_events: dict[int, tuple[pd.Timestamp, float]],
    realism: RealismConfig = RealismConfig(),
    spread_panel: pd.DataFrame | None = None,
    carry: CarryState | None = None,
    carry_over: bool = False,
    max_carry_months: int = 1,
    execution_delay: int = 0,
    stop_cooldown: bool = False,
    block_last_day_entry: bool = False,
    delisting_fix: bool = False,
    frozen_sigma: float | None = None,
    entry_confirm: bool = False,
    structural_exits: bool = False,
    blackout: dict[int, frozenset] | None = None,
    regime_scale: pd.Series | None = None,
) -> tuple[list[Trade], pd.Series, pd.Series, int, CarryState | None]:
    """Simulate ONE pair across ONE trading month.

    Parameters
    ----------
    panel : DataFrame
        Price panel spanning formation + trading window (so z-score lookback works).
    trading_dates : DatetimeIndex
        The subset of `panel.index` that constitutes the trading month.
    delisting_events : dict
        Permno -> (dlst_date, dlret) for any stock that delists.
    carry : CarryState | None
        Phase 5. If set, the pair starts the month ALREADY in this position (carried
        from the prior month). `gamma` must be `carry.gamma_frozen`. The entry cost is
        not re-charged (it was paid when the position first opened).
    carry_over : bool
        Phase 5. If True, a position still open at month-end is returned as a CarryState
        instead of being force-closed — provided it has not yet been open for
        `max_carry_months` distinct months.
    max_carry_months : int
        Phase 5. Cap on the number of distinct months a single position may stay open.
    execution_delay : int
        Phase 6 (D6.5). 0 (default, Phases 1–5): a signal observed at close t fills at
        that same close — optimistic, since you cannot observe a close and trade at it.
        1: the signal fills at the NEXT close (t+1); the position's first P&L day is
        then t+2. Delisting closes are mechanical and never delayed; a signal pending
        at month-end lapses (month-end force-close/carry applies as usual).
    stop_cooldown : bool
        Phase 6 (D6.3). After a stop-loss exit, re-entry is blocked until |z| has come
        back INSIDE the entry band (|z| < entry_sigma) once. Without this, a stop at
        3.5σ is typically followed by |z| ≥ 2σ the very next day, re-opening the same
        position and converting the stop into pure close/reopen churn.
    block_last_day_entry : bool
        Phase 6 (D6.4). Skip entries on the month's final trading day. Such positions
        can never earn a return in-month (they force-close at the same close they
        entered) yet are charged entry+exit costs — costs the aggregation layer then
        drops because the pair has zero days in position.
    delisting_fix : bool
        Phase 6 (D6.2). Compound the delisting return with the day's market return
        instead of overwriting it, and force-close a delisting position even when the
        z-score is NaN that day (default engine skips the close and accrues NaNs).

    Returns
    -------
    (trades, daily_returns, daily_weight, days_in_position, carry_out)
        daily_returns is indexed by trading_dates; 0.0 on flat days.
        daily_weight is |entry_z| of the position open that day (0.0 when flat) —
        used by the |entry-z|-weighted allocation variant (Phase 3c).
        days_in_position is used to weight the per-month average open-pair count.
        carry_out is a CarryState if the position is held into next month, else None.
    """
    prices_a = panel[permno_a]
    prices_b = panel[permno_b]
    spread = spread_series(prices_a, prices_b, gamma)
    z = rolling_zscore(spread, window=zscore_window, frozen_sigma=frozen_sigma)

    # find delisting events that fall on or before the end of the trading month
    dlst_a = delisting_events.get(permno_a)
    dlst_b = delisting_events.get(permno_b)
    last_trading_day = trading_dates[-1]
    if dlst_a and dlst_a[0] > last_trading_day:
        dlst_a = None
    if dlst_b and dlst_b[0] > last_trading_day:
        dlst_b = None

    # ── initial state: carried in from last month, or flat ───────────────
    if carry is not None:
        position = carry.direction
        entry_date: pd.Timestamp | None = carry.entry_date
        entry_z = carry.entry_z
        current_trade_pnl = carry.cum_pnl     # gross P&L accumulated over prior months
        is_carried_position = True            # the open trade originated in a prior month
    else:
        position = 0
        entry_date = None
        entry_z = 0.0
        current_trade_pnl = 0.0
        is_carried_position = False

    trades: list[Trade] = []
    daily_pnl: list[float] = []
    daily_weight: list[float] = []   # |entry_z| of the open position that day, else 0
    days_in_position = 0

    # realism (Phase 4a) — all zero/no-op when realism is frictionless
    borrow_daily = borrow_cost_daily(realism.borrow_bps_annual)

    def _entry_exit_cost(date: pd.Timestamp) -> float:
        if not realism.transaction_costs:
            return 0.0
        return realism.spread_cost_multiplier * transaction_cost(
            spread_panel, date, permno_a, permno_b, realism.default_spread_bps
        )

    def _close_trade(exit_date: pd.Timestamp, exit_z: float, reason: ExitReason) -> None:
        """Emit the round-trip Trade for the currently-open position. round_trip_return
        is the gross P&L over the whole trade (across months if carried), excluding the
        entry/exit transaction costs — matching the pre-Phase-5 diagnostic convention."""
        trades.append(Trade(
            permno_a=permno_a, permno_b=permno_b, direction=position,
            entry_date=entry_date, exit_date=exit_date,
            entry_z=entry_z, exit_z=exit_z,
            round_trip_return=current_trade_pnl, exit_reason=reason,
        ))

    # we need the previous close to compute today's return; for the first trading day
    # this is the last formation-window close
    first_trading_idx = panel.index.get_loc(trading_dates[0])
    prev_close_a = float(prices_a.iloc[first_trading_idx - 1])
    prev_close_b = float(prices_b.iloc[first_trading_idx - 1])

    # Phase 6 (D6.5) execution-delay state: a signal observed at close t executes at
    # close t+1 when execution_delay=1. Tuples hold the SIGNAL-day z (the dislocation
    # that triggered the action). Both stay None when execution_delay=0.
    pending_entry: tuple[int, float] | None = None       # (direction, signal_z)
    pending_exit: tuple[ExitReason, float] | None = None  # (reason, signal_z)
    # Phase 6 (D6.3) stop cooldown: disarmed after a stop-out, re-armed once |z| is
    # back inside the entry band. Always True when stop_cooldown=False.
    armed = True

    # Feature 1: entry confirmation — track previous z for turning-point filter
    prev_z: float = float("nan")
    # Feature 3: structural break — 5-day rolling return buffers (one deque per leg)
    recent_ret_a: deque = deque(maxlen=5)
    recent_ret_b: deque = deque(maxlen=5)
    # Feature 4: regime scale locked at trade entry; 1.0 = no scaling
    entry_scale: float = 1.0

    for t in trading_dates:
        is_last_day = t == trading_dates[-1]
        close_a = float(prices_a.loc[t])
        close_b = float(prices_b.loc[t])

        # ── 1. mark today's P&L based on position carried into today ──────
        delist_today = False
        if position != 0:
            # if either leg delists today, replace (default) or compound (D6.2) its
            # return with the delisting return (Option B) and force-close at close[t]
            ret_a = close_a / prev_close_a - 1.0
            ret_b = close_b / prev_close_b - 1.0
            if dlst_a and dlst_a[0] == t:
                ret_a = _delisting_day_return(ret_a, dlst_a[1], delisting_fix)
                delist_today = True
            if dlst_b and dlst_b[0] == t:
                ret_b = _delisting_day_return(ret_b, dlst_b[1], delisting_fix)
                delist_today = True
            day_ret = _equal_dollar_daily_return(position, ret_a, ret_b)
            day_ret *= entry_scale        # Feature 4: regime scale (locked at entry, 1.0 if off)
            day_ret -= borrow_daily       # borrow fee on the short leg (0 if frictionless)
            current_trade_pnl += day_ret  # accumulate gross P&L for this trade
            days_in_position += 1
            daily_weight.append(abs(entry_z))   # weight by entry dislocation
            # Feature 3: feed rolling buffers for structural-break detection
            if structural_exits:
                recent_ret_a.append(ret_a)
                recent_ret_b.append(ret_b)
        else:
            day_ret = 0.0
            daily_weight.append(0.0)
        daily_pnl.append(day_ret)
        prev_close_a = close_a
        prev_close_b = close_b

        z_t = z.loc[t]

        # ── 2. execute actions signalled YESTERDAY (execution_delay=1 only) ──
        if pending_exit is not None:
            if position != 0:                 # today's P&L above still accrued to us
                reason, signal_z = pending_exit
                _close_trade(t, signal_z, reason)
                daily_pnl[-1] -= _entry_exit_cost(t)   # exit transaction cost
                position = 0
                current_trade_pnl = 0.0
                is_carried_position = False
                if reason == "stop_loss" and stop_cooldown:
                    armed = False
            pending_exit = None
        if pending_entry is not None:
            if position == 0 and not (block_last_day_entry and is_last_day):
                direction, signal_z = pending_entry
                position = direction
                entry_date = t
                entry_z = signal_z
                current_trade_pnl = 0.0          # fresh trade
                is_carried_position = False
                daily_pnl[-1] -= _entry_exit_cost(t)   # entry transaction cost
            pending_entry = None

        # ── 3. compute signal at close ─────────────────────────────────────
        if pd.isna(z_t):
            if delisting_fix and position != 0 and delist_today:
                # D6.2: a delisting close is mechanical — it must not be skipped
                # just because the z-score is undefined today.
                _close_trade(t, 0.0, "delisting")
                daily_pnl[-1] -= _entry_exit_cost(t)   # exit transaction cost
                position = 0
                current_trade_pnl = 0.0
                is_carried_position = False
                pending_entry = None
                pending_exit = None
            continue

        # D6.3: re-arm once z is back inside the entry band
        if stop_cooldown and not armed and abs(float(z_t)) < entry_sigma:
            armed = True

        # forced exit: delisting today (mechanical — never delayed)
        if position != 0 and delist_today:
            _close_trade(t, float(z_t), "delisting")
            daily_pnl[-1] -= _entry_exit_cost(t)   # exit transaction cost
            position = 0
            current_trade_pnl = 0.0
            is_carried_position = False
            pending_entry = None
            pending_exit = None
            prev_z = float(z_t)
            continue

        # stop-loss (realism variant only): z moved further against us
        if (
            position != 0
            and pending_exit is None
            and stop_sigma is not None
            and position * float(z_t) <= -stop_sigma
        ):
            if execution_delay == 0:
                _close_trade(t, float(z_t), "stop_loss")
                daily_pnl[-1] -= _entry_exit_cost(t)   # exit transaction cost
                position = 0
                current_trade_pnl = 0.0
                is_carried_position = False
                if stop_cooldown:
                    armed = False
                recent_ret_a.clear()
                recent_ret_b.clear()
            else:
                pending_exit = ("stop_loss", float(z_t))
            prev_z = float(z_t)
            continue

        # reversion exit: z crossed back through zero (or ±0.5 with Feature 5)
        if position != 0 and pending_exit is None and position * float(z_t) >= exit_sigma:
            if execution_delay == 0:
                _close_trade(t, float(z_t), "reversion")
                daily_pnl[-1] -= _entry_exit_cost(t)   # exit transaction cost
                position = 0
                current_trade_pnl = 0.0
                is_carried_position = False
                recent_ret_a.clear()
                recent_ret_b.clear()
            else:
                pending_exit = ("reversion", float(z_t))
            prev_z = float(z_t)
            continue

        # Feature 3: structural break exit — 5-day rolling correlation collapse
        if structural_exits and position != 0 and pending_exit is None:
            if len(recent_ret_a) >= 5:
                c = np.corrcoef(list(recent_ret_a), list(recent_ret_b))[0, 1]
                if np.isfinite(c) and c < 0.5:
                    if execution_delay == 0:
                        _close_trade(t, float(z_t), "reversion")
                        daily_pnl[-1] -= _entry_exit_cost(t)
                        position = 0
                        current_trade_pnl = 0.0
                        is_carried_position = False
                        recent_ret_a.clear()
                        recent_ret_b.clear()
                    else:
                        pending_exit = ("reversion", float(z_t))
                    prev_z = float(z_t)
                    continue

        # Feature 2: earnings blackout exit — force-close if either leg enters window
        if blackout and position != 0 and pending_exit is None:
            if (t in blackout.get(permno_a, frozenset())
                    or t in blackout.get(permno_b, frozenset())):
                if execution_delay == 0:
                    _close_trade(t, float(z_t), "reversion")
                    daily_pnl[-1] -= _entry_exit_cost(t)
                    position = 0
                    current_trade_pnl = 0.0
                    is_carried_position = False
                    recent_ret_a.clear()
                    recent_ret_b.clear()
                else:
                    pending_exit = ("reversion", float(z_t))
                prev_z = float(z_t)
                continue

        # entry (only if flat, armed, no pending action, and not blocked last-day)
        if position == 0 and pending_entry is None and pending_exit is None and armed:
            if block_last_day_entry and is_last_day:
                pass    # D6.4: a last-day entry can never earn an in-month return
            # Feature 2: earnings blackout — block new entries near announcements
            elif blackout and (t in blackout.get(permno_a, frozenset())
                               or t in blackout.get(permno_b, frozenset())):
                pass    # skip entry; prev_z not updated so confirmation stays valid
            else:
                # Feature 1: entry confirmation — require z turning back toward zero
                if entry_confirm:
                    confirming_short = np.isfinite(prev_z) and float(z_t) < prev_z
                    confirming_long  = np.isfinite(prev_z) and float(z_t) > prev_z
                else:
                    confirming_short = confirming_long = True

                if float(z_t) >= entry_sigma and confirming_short:
                    # Feature 4: lock regime scale at entry
                    entry_scale = (
                        float(regime_scale.loc[t])
                        if regime_scale is not None and t in regime_scale.index
                        else 1.0
                    )
                    if execution_delay == 0:
                        position = -1   # short spread
                        entry_date = t
                        entry_z = float(z_t)
                        current_trade_pnl = 0.0          # fresh trade
                        is_carried_position = False
                        daily_pnl[-1] -= _entry_exit_cost(t)   # entry transaction cost
                    else:
                        pending_entry = (-1, float(z_t))
                elif float(z_t) <= -entry_sigma and confirming_long:
                    # Feature 4: lock regime scale at entry
                    entry_scale = (
                        float(regime_scale.loc[t])
                        if regime_scale is not None and t in regime_scale.index
                        else 1.0
                    )
                    if execution_delay == 0:
                        position = +1   # long spread
                        entry_date = t
                        entry_z = float(z_t)
                        current_trade_pnl = 0.0          # fresh trade
                        is_carried_position = False
                        daily_pnl[-1] -= _entry_exit_cost(t)   # entry transaction cost
                    else:
                        pending_entry = (+1, float(z_t))
        prev_z = float(z_t)

    # ── 3. month end: carry the position into next month, or force-close ──
    carry_out: CarryState | None = None
    if position != 0:
        # months this position has already been open = boundaries it has crossed.
        # A fresh (non-carried) open trade has crossed 0; a re-entry within a carried
        # month is also fresh (is_carried_position flipped False on re-entry).
        cur_m = carry.months_carried if (carry is not None and is_carried_position) else 0
        can_carry = carry_over and (cur_m + 1 < max_carry_months)
        last_z = float(z.loc[trading_dates[-1]])
        if can_carry:
            carry_out = CarryState(
                permno_a=permno_a, permno_b=permno_b, direction=position,
                gamma_frozen=float(gamma), entry_date=entry_date, entry_z=entry_z,
                cum_pnl=current_trade_pnl, months_carried=cur_m + 1,
            )
        else:
            _close_trade(trading_dates[-1],
                         last_z if not np.isnan(last_z) else 0.0, "force_close")
            daily_pnl[-1] -= _entry_exit_cost(trading_dates[-1])   # exit transaction cost

    name = f"{permno_a}_{permno_b}"
    return (
        trades,
        pd.Series(daily_pnl, index=trading_dates, name=name),
        pd.Series(daily_weight, index=trading_dates, name=name),
        days_in_position,
        carry_out,
    )


# ────────────────────────────────────────────────────────────────────────────────
# monthly orchestrator
# ────────────────────────────────────────────────────────────────────────────────


def run_one_month(
    formation_end: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
    crsp: pd.DataFrame,
    constituents: pd.DataFrame,
    delisting_df: pd.DataFrame,
    entry_sigma: float = ENTRY_THRESHOLD,
    exit_sigma: float = EXIT_THRESHOLD,
    stop_sigma: float | None = STOP_LOSS_SIGMA,
    zscore_window: int = ZSCORE_WINDOW_MONTHS * 21,
    formation_years: int = FORMATION_YEARS,
    metric: Literal["ssd", "pc", "factor"] = "ssd",
    cointegration_filter: bool = False,
    market_returns: pd.Series | None = None,
    clusterer: Literal["optics", "hdbscan", "hierarchical", "gg"] = "optics",
    hedge_method: Literal["ols", "rlm"] = "ols",
    allocation: Literal["equal", "zweight"] = "equal",
    realism: RealismConfig = RealismConfig(),
    spread_panel: pd.DataFrame | None = None,
    style_factors: pd.DataFrame | None = None,
    carry_in: dict[tuple[int, int], CarryState] | None = None,
    carry_over: bool = False,
    max_carry_months: int = MAX_CARRY_MONTHS,
    coint_pvalue: Literal["adf", "mackinnon"] = "adf",
    use_coint_gamma: bool = False,
    execution_delay: int = 0,
    stop_cooldown: bool = False,
    block_last_day_entry: bool = False,
    delisting_fix: bool = False,
    use_frozen_sigma: bool = False,
    entry_confirm: bool = False,
    structural_exits: bool = False,
    blackout_days: int = 0,
    ibes_df: pd.DataFrame | None = None,
    regime_scale: pd.Series | None = None,
    top_k_pairs: int | None = None,
    print_pairs: bool = False,
    gg_top_n: int = 20,
    gg_dist_floor: float = 0.0,
) -> MonthResult:
    """Run the full pipeline for one month.

    `formation_end` is the LAST formation-window day (= last trading day BEFORE
    the trading month starts).
    `trading_dates` is the index of trading days in the month being simulated.

    Phase 2 / 2.5 new args (all default to Phase 1 behavior):
      metric : "ssd" (Phase 1 default), "pc" (Phase 2), or "factor" (Phase 2.5
               factor-beta clustering extension).
      cointegration_filter : if True, apply Engle-Granger + half-life filter
                             between clustering and γ-fit.
      market_returns : required when metric="pc". Series of daily market returns
                       (e.g. S&P 500). If None and metric="pc", loaded automatically.

    Phase 3 new args (default to prior behavior):
      clusterer : "optics" (default), "hdbscan", or "hierarchical" (robustness 3a).
      hedge_method : "ols" (default) or "rlm" (robust hedge ratio, 3b).
      allocation : "equal" (default) or "zweight" (|entry-z|-weighted sizing, 3c).

    Phase 6 corrections (each behind a flag; ALL defaults reproduce the Phase 5
    engine bit-identically — see phases/phase6/decisions.md):
      coint_pvalue : "adf" (Phase 2 behaviour) or "mackinnon" (D6.1 — proper
          Engle-Granger p-values; only meaningful with cointegration_filter=True).
      use_coint_gamma : D6.6 — trade the spread in the direction the cointegration
          filter actually validated, using its γ (only with cointegration_filter).
      execution_delay : D6.5 — 0 = fill at the signal close (default), 1 = fill at
          the next close.
      stop_cooldown : D6.3 — after a stop-out, block re-entry until |z| < entry_sigma.
      block_last_day_entry : D6.4 — no entries on the month's final trading day.
      delisting_fix : D6.2 — corrected delisting-code fallback map, compounded
          delisting-day return, weekend/holiday delisting dates snapped to the next
          trading day, and delisting closes no longer skipped on NaN z.
      use_frozen_sigma : if True, use the formation-window OLS residual std (from
          fit_hedge_ratio) as the fixed z-score denominator instead of the default
          rolling std. Anchors volatility to what was observed before trading began,
          so a spread that widens mid-month doesn't shrink the z-score.
    """
    fit_gamma = fit_hedge_ratio if hedge_method == "ols" else fit_hedge_ratio_rlm
    # ── formation ─────────────────────────────────────────────────────────
    formation_panel = formation_window_panel(
        formation_end, crsp=crsp, constituents=constituents,
        formation_years=formation_years,
    )

    # ── distance matrix + clustering ─────────────────────────────────────
    if metric == "ssd":
        dmat = ssd_distance(formation_panel)
        xi = OPTICS_XI
    elif metric == "pc":
        if market_returns is None:
            market_returns = load_market_returns()
        dmat = pc_distance(formation_panel, market_returns)
        xi = OPTICS_XI_PC
    elif metric == "factor":
        sic_map = siccd_lookup(
            list(formation_panel.columns), crsp=crsp, as_of=formation_end
        )
        factor_panel = build_factor_panel(
            formation_panel, sic_map, style_factors=style_factors
        )
        dmat = factor_beta_distance(
            formation_panel, factor_panel, ridge_alpha=RIDGE_ALPHA
        )
        xi = OPTICS_XI_FACTOR
    else:
        raise ValueError(
            f"unknown metric {metric!r}; expected 'ssd', 'pc', or 'factor'"
        )

    if clusterer == "gg":
        # Gatev-Goetzmann style: rank ALL possible pairs by SSD distance globally.
        # No clustering — no sector awareness. Direct analog of the original paper.
        # gg_dist_floor excludes trivially-perfect pairs (dual-class shares etc.)
        # that have near-zero distance and almost never trigger z-score entries.
        perms = list(dmat.index)
        n = len(perms)
        all_pairs: list[tuple] = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = perms[i], perms[j]
                d = float(dmat.iloc[i, j])
                if d >= gg_dist_floor:
                    all_pairs.append((a, b, d))
        all_pairs.sort(key=lambda x: x[2])
        candidate_pairs = [(a, b) for a, b, _ in all_pairs[:gg_top_n]]
        n_candidate_pairs_pre_filter = len(candidate_pairs)
    else:
        if clusterer == "optics":
            labels = cluster_optics(
                dmat,
                min_samples=OPTICS_MIN_SAMPLES,
                xi=xi,
                min_cluster_size=OPTICS_MIN_CLUSTER_SIZE,
            )
        elif clusterer == "hdbscan":
            labels = cluster_hdbscan(dmat, min_cluster_size=OPTICS_MIN_CLUSTER_SIZE)
        elif clusterer == "hierarchical":
            labels = cluster_hierarchical(
                dmat, distance_quantile=HIER_QUANTILE, linkage=HIER_LINKAGE,
                min_cluster_size=OPTICS_MIN_CLUSTER_SIZE,
            )
        else:
            raise ValueError(
                f"unknown clusterer {clusterer!r}; expected 'optics', 'hdbscan', 'hierarchical', or 'gg'"
            )
        candidate_pairs = clusters_to_pairs(labels)
        n_candidate_pairs_pre_filter = len(candidate_pairs)

    # ── (optional) top-k by minimum distance ─────────────────────────────
    # Gatev-Goetzmann (2006) use the 20 pairs with smallest SSD. When top_k_pairs
    # is set, rank all within-cluster pairs by their formation-window distance and
    # keep only the closest k. This concentrates exposure on the most cointegrated
    # pairs and reduces pair count from 100-150 → k.
    if top_k_pairs is not None and len(candidate_pairs) > top_k_pairs:
        dm_vals = dmat  # square distance matrix, index/columns = permno
        pair_dists = sorted(
            candidate_pairs,
            key=lambda p: float(dm_vals.loc[p[0], p[1]])
        )
        candidate_pairs = pair_dists[:top_k_pairs]

    # ── (optional) print pairs for this month ────────────────────────────
    if print_pairs and candidate_pairs:
        from src.panel import ticker_lookup
        tick = ticker_lookup(
            permnos=[p for pair in candidate_pairs for p in pair],
            crsp=crsp, as_of=formation_end,
        )
        n_show = min(len(candidate_pairs), top_k_pairs or len(candidate_pairs))
        print(f"\n  [{trading_dates[0].date()} → {trading_dates[-1].date()}]  "
              f"{n_show} candidate pairs (ranked by SSD distance):")
        to_show = sorted(candidate_pairs, key=lambda p: float(dmat.loc[p[0], p[1]]))[:n_show]
        for rank, (a, b) in enumerate(to_show, 1):
            _ta = tick.get(a, a); ta = str(_ta.iloc[0] if hasattr(_ta, "iloc") else _ta)
            _tb = tick.get(b, b); tb = str(_tb.iloc[0] if hasattr(_tb, "iloc") else _tb)
            d = float(dmat.loc[a, b])
            print(f"    {rank:3d}. {ta:<6s} / {tb:<6s}  (permno {a}/{b}, dist={d:.4f})")

    # ── (optional Phase 2) cointegration filter ───────────────────────────
    coint_results: dict[tuple[int, int], object] = {}
    if cointegration_filter and candidate_pairs:
        candidate_pairs, coint_results = filter_cointegrated_pairs(
            candidate_pairs,
            formation_panel,
            p_threshold=COINTEGRATION_P_THRESHOLD,
            half_life_bounds=HALF_LIFE_BOUNDS,
            pvalue_method=coint_pvalue,
        )

    carry_in = carry_in or {}
    if not candidate_pairs:
        # No tradable pairs this month → close out everything we were carrying.
        forced = [_carry_to_forced_trade(cs, trading_dates[0]) for cs in carry_in.values()]
        return MonthResult(
            month_end=trading_dates[-1],
            n_candidate_pairs=n_candidate_pairs_pre_filter,
            n_pairs_traded=0, n_trades=len(forced),
            avg_pairs_open=0.0, monthly_return=0.0,
            daily_returns=pd.Series(0.0, index=trading_dates),
            trades=forced, carry_out={},
        )

    # ── extend panel with the trading window for spread/z-score ──────────
    universe = list(formation_panel.columns)
    trading_panel = (
        crsp.loc[
            (crsp["permno"].isin(universe))
            & (crsp["date"] > formation_end)
            & (crsp["date"] <= trading_dates[-1])
        ]
        .sort_values(["permno", "date"])
    )
    # build total-return prices for the trading-window slice using the same convention
    # as panel.py: cumprod of (1+ret) starting from the formation's last value
    trading_panel = trading_panel.copy()
    trading_panel["tr_price"] = (
        trading_panel.groupby("permno")["ret"]
        .transform(lambda r: (1 + r.fillna(0)).cumprod())
    )
    # scale to continue from formation-window's last value
    last_formation = formation_panel.iloc[-1]
    pivot = trading_panel.pivot_table(
        index="date", columns="permno", values="tr_price", aggfunc="first"
    )
    pivot = pivot.reindex(columns=universe)
    pivot = pivot.multiply(last_formation, axis=1).astype("float64")
    # full panel = formation + trading
    full_panel = pd.concat([formation_panel, pivot])

    # ── delisting events for the trading window ──────────────────────────
    delisting_events = get_delisting_returns(
        universe, delisting_df, fixed_codes=delisting_fix
    )
    delisting_events = {
        p: (d, r) for p, (d, r) in delisting_events.items()
        if trading_dates[0] <= d <= trading_dates[-1]
    }
    if delisting_fix:
        # D6.2: a dlstdt on a weekend/holiday never matches a panel date, so the
        # delisting return would silently never be applied — snap to the next
        # trading day inside the window.
        snapped: dict[int, tuple[pd.Timestamp, float]] = {}
        for p, (d, r) in delisting_events.items():
            pos = trading_dates.searchsorted(d, side="left")
            snapped[p] = (trading_dates[pos], r)
        delisting_events = snapped

    # Feature 2: compute earnings blackout dict for this month's trading dates
    blackout: dict[int, frozenset] = {}
    if blackout_days > 0 and ibes_df is not None:
        blackout = _build_permno_blackout(
            ibes_df, trading_dates, blackout_days
        )

    # ── fit γ for each pair and simulate ─────────────────────────────────
    trades: list[Trade] = []
    pair_returns: dict[tuple[int, int], pd.Series] = {}
    pair_weights: dict[tuple[int, int], pd.Series] = {}
    pair_days_open: dict[tuple[int, int], int] = {}
    new_carry: dict[tuple[int, int], CarryState] = {}
    n_pairs_traded = 0

    # Phase 5: carried positions whose pair dropped out of THIS month's candidate set are
    # force-closed at the month's first trading day (the cluster no longer endorses them).
    candidate_set = set(candidate_pairs)
    for key, cs in carry_in.items():
        if key not in candidate_set:
            trades.append(_carry_to_forced_trade(cs, trading_dates[0]))

    for permno_a, permno_b in candidate_pairs:
        key = (permno_a, permno_b)
        cs = carry_in.get(key)   # Phase 5: carried-in position for this pair, if any

        # a carried pair that can no longer be simulated (leg left the universe / data
        # gap that isn't a known delisting) is force-closed at the month's first day.
        if permno_a not in formation_panel.columns or permno_b not in formation_panel.columns:
            if cs is not None:
                trades.append(_carry_to_forced_trade(cs, trading_dates[0]))
            continue
        # need full trading-month prices too (some pairs lose a leg mid-month)
        if pivot[permno_a].isna().any() or pivot[permno_b].isna().any():
            # one leg disappears during the trading month — could be a delisting
            # we'll let simulate_pair_in_month handle it via delisting_events; if
            # it's not in our events table, skip (data gap)
            if (permno_a not in delisting_events) and (permno_b not in delisting_events):
                if cs is not None:
                    trades.append(_carry_to_forced_trade(cs, trading_dates[0]))
                continue

        # γ + spread orientation:
        #   carried position → frozen γ and frozen orientation (from CarryState);
        #   D6.6 (use_coint_gamma) → the WINNING direction's γ from the filter,
        #     with the spread oriented the way the filter validated it;
        #   otherwise → freshly fit A-on-B OLS (Phases 1–5 behaviour).
        sim_a, sim_b = permno_a, permno_b
        pair_frozen_sigma: float | None = None
        if cs is not None:
            sim_a, sim_b = cs.permno_a, cs.permno_b
            gamma = cs.gamma_frozen
        elif use_coint_gamma and key in coint_results:
            res = coint_results[key]
            if res.direction == f"{permno_b}_on_{permno_a}":
                sim_a, sim_b = permno_b, permno_a
            gamma = res.gamma
            if use_frozen_sigma:
                s = getattr(res, "residual_std", float("nan"))
                pair_frozen_sigma = s if (s == s and s > 0) else None  # NaN-safe
        else:
            try:
                hr = fit_gamma(
                    formation_panel[permno_a], formation_panel[permno_b]
                )
                gamma = hr.gamma
                if use_frozen_sigma and hr.residual_std > 0:
                    pair_frozen_sigma = hr.residual_std
            except ValueError:
                continue

        pair_trades, pair_ret, pair_wt, days_open, carry_out = simulate_pair_in_month(
            sim_a, sim_b, gamma,
            full_panel, trading_dates,
            entry_sigma=entry_sigma, exit_sigma=exit_sigma,
            stop_sigma=stop_sigma, zscore_window=zscore_window,
            delisting_events=delisting_events,
            realism=realism, spread_panel=spread_panel,
            carry=cs, carry_over=carry_over, max_carry_months=max_carry_months,
            execution_delay=execution_delay, stop_cooldown=stop_cooldown,
            block_last_day_entry=block_last_day_entry, delisting_fix=delisting_fix,
            frozen_sigma=pair_frozen_sigma,
            entry_confirm=entry_confirm, structural_exits=structural_exits,
            blackout=blackout or None, regime_scale=regime_scale,
        )
        if carry_out is not None:
            new_carry[key] = carry_out
        if pair_trades:
            n_pairs_traded += 1
        if days_open > 0:
            pair_returns[key] = pair_ret
            pair_weights[key] = pair_wt
            pair_days_open[key] = days_open
            trades.extend(pair_trades)

    # ── portfolio aggregation across currently-open pairs ────────────────
    # "equal"  : daily return = equal-weight mean across pairs open TODAY.
    # "zweight": weight each open pair by |entry-z| of its open trade, renormalized
    #            daily (bet more on stronger dislocations) — Phase 3c.
    # A pair that's flat (return 0) contributes nothing either way.
    if not pair_returns:
        daily_portfolio = pd.Series(0.0, index=trading_dates)
        avg_pairs_open = 0.0
    else:
        all_returns = pd.DataFrame(pair_returns)
        # Phase 6 (D6.7) NaN guard: a NaN pair-day return means a data problem
        # (e.g. a delisting date that never matched a panel date). Previously these
        # were SILENTLY skipped by .mean(skipna) while still counting as "open".
        # Surface them loudly; numerics are unchanged (NaN→0 is excluded from the
        # open-pair mean either way).
        if all_returns.isna().to_numpy().any():
            _bad = all_returns.columns[all_returns.isna().any()].tolist()
            print(
                f"  WARNING [{trading_dates[-1].date()}]: NaN daily returns for "
                f"{len(_bad)} pair(s) {_bad[:5]}{' ...' if len(_bad) > 5 else ''} "
                f"— treated as 0 (flat); check delisting/data coverage"
            )
            all_returns = all_returns.fillna(0.0)
        active = (all_returns != 0).astype(int)
        n_open_per_day = active.sum(axis=1)
        avg_pairs_open = float(n_open_per_day.mean())

        if allocation == "equal":
            daily_portfolio = (
                all_returns.where(all_returns != 0).mean(axis=1).fillna(0.0)
            )
        elif allocation == "zweight":
            w = pd.DataFrame(pair_weights).reindex_like(all_returns).fillna(0.0)
            w = w.where(all_returns != 0, 0.0)          # only weight open pairs
            wsum = w.sum(axis=1)
            daily_portfolio = (
                (all_returns * w).sum(axis=1) / wsum.where(wsum != 0)
            ).fillna(0.0)
        else:
            raise ValueError(
                f"unknown allocation {allocation!r}; expected 'equal' or 'zweight'"
            )

    monthly_return = float((1 + daily_portfolio).prod() - 1)

    return MonthResult(
        month_end=trading_dates[-1],
        # report pre-filter count for comparability; "n_pairs_traded" already
        # captures how many *actually* opened trades after filtering
        n_candidate_pairs=n_candidate_pairs_pre_filter,
        n_pairs_traded=n_pairs_traded,
        n_trades=len(trades),
        avg_pairs_open=avg_pairs_open,
        monthly_return=monthly_return,
        daily_returns=daily_portfolio,
        trades=trades,
        carry_out=new_carry,
    )


# ────────────────────────────────────────────────────────────────────────────────
# full backtest driver
# ────────────────────────────────────────────────────────────────────────────────


def get_month_end_grid(crsp: pd.DataFrame, start: str, end: str) -> list[pd.Timestamp]:
    """Get the list of trading-month-end dates between `start` and `end` inclusive."""
    all_dates = pd.DatetimeIndex(crsp["date"].drop_duplicates().sort_values())
    all_dates = all_dates[(all_dates >= pd.Timestamp(start)) & (all_dates <= pd.Timestamp(end))]
    month_grouper = all_dates.to_series().groupby(all_dates.to_period("M"))
    return [pd.Timestamp(g.iloc[-1]) for _, g in month_grouper]


def run_backtest(
    start: str = "2003-01-01",
    end: str = "2023-12-31",
    entry_sigma: float = ENTRY_THRESHOLD,
    exit_sigma: float = EXIT_THRESHOLD,
    stop_sigma: float | None = STOP_LOSS_SIGMA,
    zscore_window: int | None = None,
    formation_years: int = FORMATION_YEARS,
    crsp: pd.DataFrame | None = None,
    constituents: pd.DataFrame | None = None,
    delisting_df: pd.DataFrame | None = None,
    market_returns: pd.Series | None = None,
    metric: Literal["ssd", "pc", "factor"] = "ssd",
    cointegration_filter: bool = False,
    clusterer: Literal["optics", "hdbscan", "hierarchical", "gg"] = "optics",
    hedge_method: Literal["ols", "rlm"] = "ols",
    allocation: Literal["equal", "zweight"] = "equal",
    realism: RealismConfig | None = None,
    style_factors: pd.DataFrame | None = None,
    carry_over: bool = False,
    max_carry_months: int = MAX_CARRY_MONTHS,
    coint_pvalue: Literal["adf", "mackinnon"] = "adf",
    use_coint_gamma: bool = False,
    execution_delay: int = 0,
    stop_cooldown: bool = False,
    block_last_day_entry: bool = False,
    delisting_fix: bool = False,
    use_frozen_sigma: bool = False,
    entry_confirm: bool = False,
    structural_exits: bool = False,
    blackout_days: int = 0,
    ibes_df: pd.DataFrame | None = None,
    use_regime_scale: bool = False,
    regime_scale: pd.Series | None = None,
    top_k_pairs: int | None = None,
    print_pairs: bool = False,
    gg_top_n: int = 20,
    gg_dist_floor: float = 0.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[Trade]]:
    """Run the rolling monthly backtest.

    Phase 2 new args (defaults reproduce Phase 1 behavior bit-identically):
      metric : "ssd" (default — Phase 1) or "pc" (Phase 2)
      cointegration_filter : if True, Engle-Granger + half-life filter is
          applied each month between clustering and γ-fit
      market_returns : Series of daily market returns (required when
          metric="pc"; loaded automatically if None)

    Phase 5 new args (default reproduces pre-Phase-5 behavior bit-identically):
      carry_over : if True, a position still open at month-end is HELD into the next
          month instead of force-closed, provided its pair is still in the candidate set
          and it has not been open for `max_carry_months` distinct months.
      max_carry_months : cap on how many distinct months one position may stay open.

    Phase 6 corrections (defaults reproduce the Phase 5 engine bit-identically;
    see `run_one_month` and phases/phase6/decisions.md D6.1–D6.7):
      coint_pvalue, use_coint_gamma, execution_delay, stop_cooldown,
      block_last_day_entry, delisting_fix.

    Returns
    -------
    monthly : DataFrame
        One row per month with portfolio diagnostics + return.
    all_trades : list[Trade]
        Every round-trip trade across the full sample.
    """
    if crsp is None:
        crsp = load_crsp_daily()
    if constituents is None:
        constituents = load_sp500_constituents()
    if delisting_df is None:
        delisting_df = load_delisting()
    if zscore_window is None:
        zscore_window = ZSCORE_WINDOW_MONTHS * 21
    # Feature 2: auto-load IBES if blackout requested
    if blackout_days > 0 and ibes_df is None:
        ibes_df = load_ibes_earnings()
    # Feature 4: auto-load pre-computed regime scale if requested
    if use_regime_scale and regime_scale is None:
        regime_scale = load_regime_scale()
    if metric == "pc" and market_returns is None:
        market_returns = load_market_returns()
    if realism is None:
        realism = RealismConfig()
    # Build the bid/ask spread panel once, only when transaction costs are on.
    spread_panel = build_spread_panel(crsp) if realism.transaction_costs else None

    month_ends = get_month_end_grid(crsp, start, end)
    if verbose:
        filt_str = " + cointegration filter" if cointegration_filter else ""
        print(f"Backtest [{metric.upper()}{filt_str}]: {len(month_ends)} months  "
              f"({month_ends[0].date()} → {month_ends[-1].date()})")

    rows: list[dict] = []
    all_trades: list[Trade] = []
    # Phase 5: positions carried from the prior month into the current one. Always empty
    # when carry_over is False → the loop is then bit-identical to the pre-Phase-5 engine.
    carry_positions: dict[tuple[int, int], CarryState] = {}

    for i, current_month_end in enumerate(month_ends):
        # we need a prior month to use as the formation_end
        if i == 0:
            continue
        formation_end = month_ends[i - 1]
        # trading days within current month
        trading_days = pd.DatetimeIndex(crsp["date"].drop_duplicates().sort_values())
        trading_days = trading_days[
            (trading_days > formation_end) & (trading_days <= current_month_end)
        ]
        if len(trading_days) == 0:
            continue

        try:
            res = run_one_month(
                formation_end=formation_end,
                trading_dates=trading_days,
                crsp=crsp,
                constituents=constituents,
                delisting_df=delisting_df,
                entry_sigma=entry_sigma,
                exit_sigma=exit_sigma,
                stop_sigma=stop_sigma,
                zscore_window=zscore_window,
                formation_years=formation_years,
                metric=metric,
                cointegration_filter=cointegration_filter,
                market_returns=market_returns,
                clusterer=clusterer,
                hedge_method=hedge_method,
                allocation=allocation,
                realism=realism,
                spread_panel=spread_panel,
                style_factors=style_factors,
                carry_in=carry_positions,
                carry_over=carry_over,
                max_carry_months=max_carry_months,
                coint_pvalue=coint_pvalue,
                use_coint_gamma=use_coint_gamma,
                execution_delay=execution_delay,
                stop_cooldown=stop_cooldown,
                block_last_day_entry=block_last_day_entry,
                delisting_fix=delisting_fix,
                use_frozen_sigma=use_frozen_sigma,
                entry_confirm=entry_confirm,
                structural_exits=structural_exits,
                blackout_days=blackout_days,
                ibes_df=ibes_df,
                regime_scale=regime_scale,
                top_k_pairs=top_k_pairs,
                print_pairs=print_pairs,
                gg_top_n=gg_top_n,
                gg_dist_floor=gg_dist_floor,
            )
        except ValueError as e:
            if verbose:
                print(f"  {current_month_end.date()}: skipped ({e})")
            continue

        carry_positions = res.carry_out   # hand open positions to the next month

        rows.append({
            "month_end": res.month_end,
            "monthly_return": res.monthly_return,
            "n_candidate_pairs": res.n_candidate_pairs,
            "n_pairs_traded": res.n_pairs_traded,
            "n_trades": res.n_trades,
            "avg_pairs_open": res.avg_pairs_open,
        })
        all_trades.extend(res.trades)

        if verbose and (i % 12 == 1):
            ann_ret = (1 + res.monthly_return) ** 12 - 1
            print(
                f"  {res.month_end.date()}: "
                f"ret={res.monthly_return:+.3%} (annualised ≈ {ann_ret:+.1%}), "
                f"pairs(cand/traded)={res.n_candidate_pairs}/{res.n_pairs_traded}, "
                f"trades={res.n_trades}, "
                f"avg_open={res.avg_pairs_open:.1f}"
            )

    # Phase 5: close out any positions still carried past the last simulated month. Their
    # P&L was already booked through that month's daily returns, so this only writes the
    # diagnostic Trade records (no effect on the monthly return series / Sharpe).
    if carry_positions:
        for cs in carry_positions.values():
            all_trades.append(_carry_to_forced_trade(cs, month_ends[-1]))

    monthly = pd.DataFrame(rows).set_index("month_end")
    if verbose:
        total = (1 + monthly["monthly_return"]).prod() - 1
        ann = (1 + total) ** (12 / len(monthly)) - 1
        print(
            f"\nDone. {len(monthly)} months. "
            f"total return = {total:+.1%}, annualised = {ann:+.1%}, "
            f"trades = {len(all_trades):,}"
        )
    return monthly, all_trades
