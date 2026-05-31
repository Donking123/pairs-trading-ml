"""ROCE / RUCE calculator (architecture §3.11, §2.4).

Subscribes to fills and assembles per-pair round-trips into
:class:`~asian_adr.core.RoceRuceResult` records:

    local_return = (P_local_close − P_local_open) / P_local_open      [USD]
    adr_return   = (P_ADR_short  − P_ADR_cover)  / P_ADR_short
    ROCE  = local_return + adr_return
    RUCE  = local_return + 2 × adr_return        (Reg-T 50% margin on the short)

An aborted trade (ADR short covered with no local leg ever placed) contributes
only the ADR return. Net figures subtract the pair's Roll (1984) round-trip
cost. The calculator also captures whether a close was a force-close by
listening to the signal stream.

Imports ``core``, ``event_bus``, and the strategy's liquidity-bucket helper
(the backtest layer orchestrates the strategy, so this dependency is expected).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from statistics import fmean, pstdev
from typing import Iterable

from asian_adr.core import (
    AsianADRApprovedPair,
    Clock,
    FillEvent,
    FOREIGN_EQUITY,
    HongSusmelSignalEvent,
    HSSignal,
    LiquidityBucket,
    RoceRuceResult,
    US_EQUITY,
)
from asian_adr.event_bus import AbstractEventBus
from asian_adr.strategy.hong_susmel.liquidity_bucket import assign_bucket

logger = logging.getLogger(__name__)

__all__ = ["RoceRuceCalculator", "distribution_stats"]


@dataclass
class _OpenTrip:
    adr_short: FillEvent | None = None
    local_buy: FillEvent | None = None
    local_sell: FillEvent | None = None


class RoceRuceCalculator:
    """Assembles round-trips from the fill stream and computes ROCE/RUCE."""

    def __init__(
        self,
        bus: AbstractEventBus,
        clock: Clock,
        pairs: Iterable[AsianADRApprovedPair],
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._pairs: dict[str, AsianADRApprovedPair] = {p.pair_id: p for p in pairs}
        self._open: dict[str, _OpenTrip] = {}
        self._force_close_pending: dict[str, bool] = {}
        self.results: list[RoceRuceResult] = []

    async def on_signal(self, signal: HongSusmelSignalEvent) -> None:
        """Record whether the next close for this pair is a force-close."""
        if signal.signal == HSSignal.FORCE_CLOSE:
            self._force_close_pending[signal.pair_id] = True
        elif signal.signal == HSSignal.EXIT:
            self._force_close_pending[signal.pair_id] = False

    async def on_fill(self, fill: FillEvent) -> None:
        pid = fill.pair_id
        if pid is None:
            return
        trip = self._open.setdefault(pid, _OpenTrip())
        if fill.venue == US_EQUITY and fill.side == "sell":
            trip.adr_short = fill
        elif fill.venue == FOREIGN_EQUITY and fill.side == "buy":
            trip.local_buy = fill
        elif fill.venue == FOREIGN_EQUITY and fill.side == "sell":
            trip.local_sell = fill
        elif fill.venue == US_EQUITY and fill.side == "buy":
            # ADR cover closes the round-trip (full or aborted)
            result = self._finalize(pid, trip, fill)
            self._open.pop(pid, None)
            if result is not None:
                self.results.append(result)

    def _finalize(
        self, pid: str, trip: _OpenTrip, cover: FillEvent
    ) -> RoceRuceResult | None:
        if trip.adr_short is None:
            return None  # cover with no open short — ignore
        pair = self._pairs.get(pid)
        roll = (
            (pair.roll_spread_adr + pair.roll_spread_local) if pair else Decimal("0")
        )
        bucket = (
            assign_bucket(pair.zero_return_pct_adr) if pair else LiquidityBucket.LOW
        )

        adr_open = trip.adr_short.fill_price
        adr_close = cover.fill_price
        adr_return = (adr_open - adr_close) / adr_open

        if trip.local_buy is None:
            # overnight abort — only the ADR leg traded
            local_return = Decimal("0")
            aborted = True
            open_date = trip.adr_short.timestamp_exchange.date()
            close_date = cover.timestamp_exchange.date()
        else:
            loc_open = trip.local_buy.fill_price_usd
            loc_close = (
                trip.local_sell.fill_price_usd if trip.local_sell else loc_open
            )
            local_return = (loc_close - loc_open) / loc_open
            aborted = False
            open_date = trip.local_buy.timestamp_exchange.date()
            close_date = (trip.local_sell or cover).timestamp_exchange.date()

        roce = local_return + adr_return
        ruce = local_return + Decimal("2") * adr_return
        was_force_closed = self._force_close_pending.pop(pid, False) and not aborted

        return RoceRuceResult(
            pair_id=pid,
            trade_open_date=open_date,
            trade_close_date=close_date,
            duration_days=(close_date - open_date).days,
            local_return=local_return,
            adr_return=adr_return,
            roce=roce,
            ruce=ruce,
            roll_cost_pct=roll,
            roce_net=roce - roll,
            ruce_net=ruce - roll,
            liquidity_bucket=bucket,
            was_force_closed=was_force_closed,
            was_aborted=aborted,
        )


# ---------------------------------------------------------------------------
# Distribution statistics (paper Table 7-B format)
# ---------------------------------------------------------------------------

_PCTILES = [
    ("mean", None), ("std", None), ("max", 1.0), ("p90", 0.90), ("p75", 0.75),
    ("median", 0.50), ("p25", 0.25), ("p10", 0.10), ("min", 0.0),
]


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < len(sorted_vals):
        return sorted_vals[lo] + frac * (sorted_vals[lo + 1] - sorted_vals[lo])
    return sorted_vals[lo]


def distribution_stats(values: Iterable[float]) -> dict[str, float]:
    """Compute the paper Table 7-B distribution for a metric series."""
    vals = sorted(float(v) for v in values)
    if not vals:
        return {label: float("nan") for label, _ in _PCTILES}
    out: dict[str, float] = {}
    for label, q in _PCTILES:
        if label == "mean":
            out[label] = fmean(vals)
        elif label == "std":
            out[label] = pstdev(vals) if len(vals) > 1 else 0.0
        elif label == "max":
            out[label] = vals[-1]
        elif label == "min":
            out[label] = vals[0]
        else:
            out[label] = _percentile(vals, q)  # type: ignore[arg-type]
    return out
