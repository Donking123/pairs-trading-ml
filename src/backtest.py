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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.clustering import cluster_optics, clusters_to_pairs
from src.cointegration import filter_cointegrated_pairs
from src.config import (
    COINTEGRATION_P_THRESHOLD,
    DATA_DIR,
    ENTRY_THRESHOLD,
    EXIT_THRESHOLD,
    FORMATION_YEARS,
    HALF_LIFE_BOUNDS,
    OPTICS_MIN_CLUSTER_SIZE,
    OPTICS_MIN_SAMPLES,
    OPTICS_XI,
    OPTICS_XI_PC,
    STOP_LOSS_SIGMA,
    ZSCORE_WINDOW_MONTHS,
)
from src.distances import pc_distance, ssd_distance
from src.panel import (
    formation_window_panel,
    load_crsp_daily,
    load_market_returns,
    load_sp500_constituents,
)
from src.spread import fit_hedge_ratio, rolling_zscore, spread_series


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


def load_delisting(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "delisting.parquet")


def get_delisting_returns(
    permnos: list[int],
    delisting_df: pd.DataFrame,
) -> dict[int, tuple[pd.Timestamp, float]]:
    """For each permno, return (delisting_date, return) if it delists; else absent.

    Uses dlret if present, otherwise the code-dependent fallback.
    """
    out: dict[int, tuple[pd.Timestamp, float]] = {}
    sub = delisting_df.loc[delisting_df["permno"].isin(permnos)]
    for _, row in sub.iterrows():
        if pd.isna(row["dlstdt"]):
            continue
        dlret = row.get("dlret")
        if pd.notna(dlret):
            ret = float(dlret)
        else:
            ret = _delisting_fallback_return(int(row["dlstcd"]))
        out[int(row["permno"])] = (pd.Timestamp(row["dlstdt"]), ret)
    return out


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
) -> tuple[list[Trade], pd.Series, float]:
    """Simulate ONE pair across ONE trading month.

    Parameters
    ----------
    panel : DataFrame
        Price panel spanning formation + trading window (so z-score lookback works).
    trading_dates : DatetimeIndex
        The subset of `panel.index` that constitutes the trading month.
    delisting_events : dict
        Permno -> (dlst_date, dlret) for any stock that delists.

    Returns
    -------
    (trades, daily_returns, days_in_position)
        daily_returns is indexed by trading_dates; 0.0 on flat days.
        days_in_position is used to weight the per-month average open-pair count.
    """
    prices_a = panel[permno_a]
    prices_b = panel[permno_b]
    spread = spread_series(prices_a, prices_b, gamma)
    z = rolling_zscore(spread, window=zscore_window)

    # find delisting events that fall on or before the end of the trading month
    dlst_a = delisting_events.get(permno_a)
    dlst_b = delisting_events.get(permno_b)
    last_trading_day = trading_dates[-1]
    if dlst_a and dlst_a[0] > last_trading_day:
        dlst_a = None
    if dlst_b and dlst_b[0] > last_trading_day:
        dlst_b = None

    position = 0
    entry_date: pd.Timestamp | None = None
    entry_z = 0.0
    entry_price_a = entry_price_b = 0.0

    trades: list[Trade] = []
    daily_pnl: list[float] = []
    days_in_position = 0

    # we need the previous close to compute today's return; for the first trading day
    # this is the last formation-window close
    first_trading_idx = panel.index.get_loc(trading_dates[0])
    prev_close_a = float(prices_a.iloc[first_trading_idx - 1])
    prev_close_b = float(prices_b.iloc[first_trading_idx - 1])

    for t in trading_dates:
        close_a = float(prices_a.loc[t])
        close_b = float(prices_b.loc[t])

        # ── 1. mark today's P&L based on position carried into today ──────
        delist_today = False
        if position != 0:
            # if either leg delists today, override its return with the
            # delisting return (Option B) and force-close at close[t]
            ret_a = close_a / prev_close_a - 1.0
            ret_b = close_b / prev_close_b - 1.0
            if dlst_a and dlst_a[0] == t:
                ret_a = dlst_a[1]
                delist_today = True
            if dlst_b and dlst_b[0] == t:
                ret_b = dlst_b[1]
                delist_today = True
            day_ret = _equal_dollar_daily_return(position, ret_a, ret_b)
            days_in_position += 1
        else:
            day_ret = 0.0
        daily_pnl.append(day_ret)
        prev_close_a = close_a
        prev_close_b = close_b

        # ── 2. compute signal at close (for execution from t+1) ───────────
        z_t = z.loc[t]
        if pd.isna(z_t):
            continue

        # forced exit: delisting today
        if position != 0 and delist_today:
            trades.append(Trade(
                permno_a=permno_a, permno_b=permno_b, direction=position,
                entry_date=entry_date, exit_date=t,
                entry_z=entry_z, exit_z=float(z_t),
                round_trip_return=float(np.sum(daily_pnl[-days_in_position:])
                                        if days_in_position else 0.0),
                exit_reason="delisting",
            ))
            position = 0
            continue

        # stop-loss (realism variant only): z moved further against us
        if (
            position != 0
            and stop_sigma is not None
            and position * float(z_t) <= -stop_sigma
        ):
            trades.append(Trade(
                permno_a=permno_a, permno_b=permno_b, direction=position,
                entry_date=entry_date, exit_date=t,
                entry_z=entry_z, exit_z=float(z_t),
                round_trip_return=float(np.sum(daily_pnl[-days_in_position:])
                                        if days_in_position else 0.0),
                exit_reason="stop_loss",
            ))
            position = 0
            continue

        # reversion exit: z crossed back through zero
        if position != 0 and position * float(z_t) >= exit_sigma:
            trades.append(Trade(
                permno_a=permno_a, permno_b=permno_b, direction=position,
                entry_date=entry_date, exit_date=t,
                entry_z=entry_z, exit_z=float(z_t),
                round_trip_return=float(np.sum(daily_pnl[-days_in_position:])
                                        if days_in_position else 0.0),
                exit_reason="reversion",
            ))
            position = 0
            continue

        # entry (only if flat)
        if position == 0:
            if float(z_t) >= entry_sigma:
                position = -1   # short spread
                entry_date = t
                entry_z = float(z_t)
                entry_price_a = close_a
                entry_price_b = close_b
            elif float(z_t) <= -entry_sigma:
                position = +1   # long spread
                entry_date = t
                entry_z = float(z_t)
                entry_price_a = close_a
                entry_price_b = close_b

    # ── 3. force-close at month end if still open ────────────────────────
    if position != 0:
        last_z = float(z.loc[trading_dates[-1]])
        trades.append(Trade(
            permno_a=permno_a, permno_b=permno_b, direction=position,
            entry_date=entry_date, exit_date=trading_dates[-1],
            entry_z=entry_z, exit_z=last_z if not np.isnan(last_z) else 0.0,
            round_trip_return=float(np.sum(daily_pnl[-days_in_position:])
                                    if days_in_position else 0.0),
            exit_reason="force_close",
        ))

    return trades, pd.Series(daily_pnl, index=trading_dates, name=f"{permno_a}_{permno_b}"), days_in_position


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
    metric: Literal["ssd", "pc"] = "ssd",
    cointegration_filter: bool = False,
    market_returns: pd.Series | None = None,
) -> MonthResult:
    """Run the full pipeline for one month.

    `formation_end` is the LAST formation-window day (= last trading day BEFORE
    the trading month starts).
    `trading_dates` is the index of trading days in the month being simulated.

    Phase 2 new args (all default to Phase 1 behavior):
      metric : "ssd" (Phase 1 default) or "pc" (Phase 2).
      cointegration_filter : if True, apply Engle-Granger + half-life filter
                             between clustering and γ-fit.
      market_returns : required when metric="pc". Series of daily market returns
                       (e.g. S&P 500). If None and metric="pc", loaded automatically.
    """
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
    else:
        raise ValueError(f"unknown metric {metric!r}; expected 'ssd' or 'pc'")

    labels = cluster_optics(
        dmat,
        min_samples=OPTICS_MIN_SAMPLES,
        xi=xi,
        min_cluster_size=OPTICS_MIN_CLUSTER_SIZE,
    )
    candidate_pairs = clusters_to_pairs(labels)
    n_candidate_pairs_pre_filter = len(candidate_pairs)

    # ── (optional Phase 2) cointegration filter ───────────────────────────
    if cointegration_filter and candidate_pairs:
        candidate_pairs, _coint_results = filter_cointegrated_pairs(
            candidate_pairs,
            formation_panel,
            p_threshold=COINTEGRATION_P_THRESHOLD,
            half_life_bounds=HALF_LIFE_BOUNDS,
        )

    if not candidate_pairs:
        return MonthResult(
            month_end=trading_dates[-1],
            n_candidate_pairs=n_candidate_pairs_pre_filter,
            n_pairs_traded=0, n_trades=0,
            avg_pairs_open=0.0, monthly_return=0.0,
            daily_returns=pd.Series(0.0, index=trading_dates),
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
    delisting_events = get_delisting_returns(universe, delisting_df)
    delisting_events = {
        p: (d, r) for p, (d, r) in delisting_events.items()
        if trading_dates[0] <= d <= trading_dates[-1]
    }

    # ── fit γ for each pair and simulate ─────────────────────────────────
    trades: list[Trade] = []
    pair_returns: dict[tuple[int, int], pd.Series] = {}
    pair_days_open: dict[tuple[int, int], int] = {}
    n_pairs_traded = 0

    for permno_a, permno_b in candidate_pairs:
        if permno_a not in formation_panel.columns or permno_b not in formation_panel.columns:
            continue
        # need full trading-month prices too (some pairs lose a leg mid-month)
        if pivot[permno_a].isna().any() or pivot[permno_b].isna().any():
            # one leg disappears during the trading month — could be a delisting
            # we'll let simulate_pair_in_month handle it via delisting_events; if
            # it's not in our events table, skip (data gap)
            if (permno_a not in delisting_events) and (permno_b not in delisting_events):
                continue

        try:
            fit = fit_hedge_ratio(
                formation_panel[permno_a], formation_panel[permno_b]
            )
        except ValueError:
            continue

        pair_trades, pair_ret, days_open = simulate_pair_in_month(
            permno_a, permno_b, fit.gamma,
            full_panel, trading_dates,
            entry_sigma=entry_sigma, exit_sigma=exit_sigma,
            stop_sigma=stop_sigma, zscore_window=zscore_window,
            delisting_events=delisting_events,
        )
        if pair_trades:
            n_pairs_traded += 1
        if days_open > 0:
            pair_returns[(permno_a, permno_b)] = pair_ret
            pair_days_open[(permno_a, permno_b)] = days_open
            trades.extend(pair_trades)

    # ── portfolio aggregation: equal-weight across currently-open pairs ──
    # daily return = mean of per-pair returns across pairs that had a position TODAY
    # (a pair that's flat contributes 0; we only count it as "open" if it had a nonzero pos.)
    if not pair_returns:
        daily_portfolio = pd.Series(0.0, index=trading_dates)
        avg_pairs_open = 0.0
    else:
        all_returns = pd.DataFrame(pair_returns)
        active = (all_returns != 0).astype(int)
        n_open_per_day = active.sum(axis=1)
        # mean of returns across active pairs, 0 on days no pair is open
        portfolio_ret = all_returns.where(all_returns != 0).mean(axis=1).fillna(0.0)
        daily_portfolio = portfolio_ret
        avg_pairs_open = float(n_open_per_day.mean())

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
    metric: Literal["ssd", "pc"] = "ssd",
    cointegration_filter: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[Trade]]:
    """Run the rolling monthly backtest.

    Phase 2 new args (defaults reproduce Phase 1 behavior bit-identically):
      metric : "ssd" (default — Phase 1) or "pc" (Phase 2)
      cointegration_filter : if True, Engle-Granger + half-life filter is
          applied each month between clustering and γ-fit
      market_returns : Series of daily market returns (required when
          metric="pc"; loaded automatically if None)

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
    if metric == "pc" and market_returns is None:
        market_returns = load_market_returns()

    month_ends = get_month_end_grid(crsp, start, end)
    if verbose:
        filt_str = " + cointegration filter" if cointegration_filter else ""
        print(f"Backtest [{metric.upper()}{filt_str}]: {len(month_ends)} months  "
              f"({month_ends[0].date()} → {month_ends[-1].date()})")

    rows: list[dict] = []
    all_trades: list[Trade] = []

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
            )
        except ValueError as e:
            if verbose:
                print(f"  {current_month_end.date()}: skipped ({e})")
            continue

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
