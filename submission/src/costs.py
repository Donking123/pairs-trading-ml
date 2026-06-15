"""
Realism cost model  [Phase 4a].

Turns the frictionless core backtest into a realistic one by charging, as reductions to
each pair's daily P&L:

  * Transaction costs — crossing HALF the bid/ask spread on each $0.50 leg, at entry and
    again at exit. Spreads are the ACTUAL per-name CRSP quotes (99% populated; realistically
    time-varying: ~27 bps median in 2000-04 → ~2.5 bps by 2015+), with a fixed fallback for
    the ~1% of missing/crossed quotes.
  * Borrow cost — an annual fee on the short leg (always $0.50 of a long-short pair),
    accrued daily while in position.

The 3.5σ stop-loss is the third realism lever; it already lives in the core engine via the
`stop_sigma` argument, so it is configured at call time, not here.

Everything defaults to OFF — `RealismConfig()` is frictionless and reproduces the core
backtest exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class RealismConfig:
    """Frictions for the realism backtest variant. Defaults = frictionless."""

    transaction_costs: bool = False     # charge half bid/ask spread at entry & exit
    borrow_bps_annual: float = 0.0      # annual borrow fee on the short leg
    default_spread_bps: float = 10.0    # fallback FULL spread (bps of mid) when quote bad
    spread_cost_multiplier: float = 1.0  # 1.0 = cross full half-spread (marketable);
                                         # 0.5 = passive limit orders; 0.0 = trade at mid

    @property
    def frictionless(self) -> bool:
        return (not self.transaction_costs) and self.borrow_bps_annual == 0.0


# Shared presets so notebooks stop re-declaring them inline.
#   REALISM_FULL    — marketable orders: cross the full half-spread (Phase 4a baseline).
#   REALISM_PASSIVE — limit orders: cross half the half-spread (Phase 4 cost-opt winner,
#                     +~0.20 net Sharpe). Phase 5's default realism preset.
# Both charge the 35 bps borrow on the short leg. The 3.5σ stop is a separate knob
# (run_backtest `stop_sigma`), so it is NOT baked into these presets.
REALISM_FULL = RealismConfig(
    transaction_costs=True, borrow_bps_annual=35.0,
    default_spread_bps=10.0, spread_cost_multiplier=1.0,
)
REALISM_PASSIVE = RealismConfig(
    transaction_costs=True, borrow_bps_annual=35.0,
    default_spread_bps=10.0, spread_cost_multiplier=0.5,
)


def build_spread_panel(crsp: pd.DataFrame) -> pd.DataFrame:
    """Full bid/ask spread fraction (ask-bid)/mid, as a date × permno panel.

    Invalid quotes (bid<=0, ask<=0, or ask<bid) become NaN — callers fall back to
    `RealismConfig.default_spread_bps` for those.
    """
    sub = crsp[["permno", "date", "bid", "ask"]].copy()
    bid = sub["bid"].to_numpy(dtype="float64")
    ask = sub["ask"].to_numpy(dtype="float64")
    valid = (bid > 0) & (ask > 0) & (ask >= bid)
    mid = (ask + bid) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(valid, (ask - bid) / mid, np.nan)
    sub["spread_frac"] = frac
    panel = sub.pivot_table(index="date", columns="permno", values="spread_frac",
                            aggfunc="first")
    return panel


def transaction_cost(
    spread_panel: pd.DataFrame | None,
    date: pd.Timestamp,
    permno_a: int,
    permno_b: int,
    default_spread_bps: float,
) -> float:
    """One-way transaction cost (entry OR exit) for the pair, as a fraction of $1 notional.

    Each $0.50 leg crosses half the spread: cost = 0.5·(½·spread_a) + 0.5·(½·spread_b).
    Missing/invalid quotes fall back to `default_spread_bps`.
    """
    default_frac = default_spread_bps / 1e4

    def leg_full_spread(permno: int) -> float:
        if spread_panel is None or permno not in spread_panel.columns:
            return default_frac
        try:
            v = spread_panel.at[date, permno]
        except KeyError:
            return default_frac
        return float(v) if (v is not None and np.isfinite(v)) else default_frac

    half_a = leg_full_spread(permno_a) / 2.0
    half_b = leg_full_spread(permno_b) / 2.0
    return 0.5 * half_a + 0.5 * half_b


def borrow_cost_daily(borrow_bps_annual: float) -> float:
    """Daily borrow fee as a fraction of $1 notional (short leg is $0.50)."""
    return (borrow_bps_annual / 1e4) / TRADING_DAYS_PER_YEAR * 0.5
