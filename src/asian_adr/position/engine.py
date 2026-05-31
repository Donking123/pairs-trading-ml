"""Position / PnL engine (architecture §3.8).

Maintains an average-cost :class:`LegPosition` for every ADR and local leg,
marks them to market on each daily bar, converts local-leg PnL to USD with the
current FX rate, and publishes a :class:`~asian_adr.core.PositionUpdateEvent` on
every change (fill, mark-to-market, or FX update). It also detects leg
imbalances (an ADR short open with no matching local long, or vice versa).

This engine is identical in backtest and live; only the bus, FX source, and
clock differ. It imports only ``core`` and ``event_bus``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from decimal import Decimal
from typing import Iterable, Literal

from asian_adr.core import (
    AsianADRApprovedPair,
    BarEvent,
    BaseEvent,
    Clock,
    FillEvent,
    FXRateEvent,
    PositionUpdateEvent,
)
from asian_adr.event_bus import AbstractEventBus, Topic

from .pnl_calculator import LegPosition, PnLCalculator

logger = logging.getLogger(__name__)

__all__ = ["PositionEngine", "LegImbalanceEvent"]


class LegImbalanceEvent(BaseEvent):
    """Alert: one leg of a pair is open while the other is flat.

    Transient during normal overnight sequencing (ADR short placed, local leg
    pending until the Asia open); a *persistent* imbalance indicates a failed
    leg and should be investigated.
    """

    event_type: Literal["leg_imbalance"] = "leg_imbalance"
    pair_id: str
    detail: str


class PositionEngine:
    """Tracks positions and publishes mark-to-market updates."""

    def __init__(
        self,
        bus: AbstractEventBus,
        clock: Clock,
        pairs: Iterable[AsianADRApprovedPair] = (),
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._pairs: dict[str, AsianADRApprovedPair] = {p.pair_id: p for p in pairs}
        self._legs: dict[tuple[str, str], LegPosition] = {}
        self._by_ticker: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._fx_rates: dict[str, FXRateEvent] = {}

    # -- registry / FX -------------------------------------------------------

    def set_pairs(self, pairs: Iterable[AsianADRApprovedPair]) -> None:
        self._pairs = {p.pair_id: p for p in pairs}

    async def on_fx_rate(self, event: FXRateEvent) -> None:
        self._fx_rates[event.base_currency] = event
        # Re-mark any foreign legs in this currency at their last price.
        for leg in self._legs.values():
            if leg.currency == event.base_currency and leg.last_mark is not None:
                await self._publish_update(leg, leg.last_mark, "fx_update")

    def get_usd_rate(self, currency: str) -> Decimal | None:
        if currency == "USD":
            return Decimal(1)
        event = self._fx_rates.get(currency)
        if event is None:
            return None
        return event.mid

    # -- leg bookkeeping -----------------------------------------------------

    def _currency_for(self, fill: FillEvent) -> str:
        if fill.venue == "us_equity":
            return "USD"
        pair = self._pairs.get(fill.pair_id) if fill.pair_id else None
        return pair.underlying_currency if pair else "USD"

    def _get_or_create_leg(self, fill: FillEvent) -> LegPosition:
        key = (fill.pair_id or "", fill.ticker)
        leg = self._legs.get(key)
        if leg is None:
            leg = LegPosition(
                pair_id=fill.pair_id or "",
                ticker=fill.ticker,
                venue=fill.venue,
                currency=self._currency_for(fill),
            )
            self._legs[key] = leg
            self._by_ticker[fill.ticker].append(key)
        return leg

    # -- handlers ------------------------------------------------------------

    async def on_fill(self, fill: FillEvent) -> None:
        if fill.pair_id is None:
            return
        leg = self._get_or_create_leg(fill)
        PnLCalculator.apply_fill(leg, fill)
        fx = leg.last_fx
        await self._publish_update(leg, fill.fill_price, "fill", fx=fx)
        self._check_imbalance(fill.pair_id)

    async def on_bar(self, event: BarEvent) -> None:
        keys = self._by_ticker.get(event.ticker)
        if not keys:
            return
        fx = self.get_usd_rate(event.currency)
        if fx is None:
            return  # cannot mark a foreign leg without FX → skip
        for key in keys:
            leg = self._legs[key]
            if leg.is_flat:
                continue
            await self._publish_update(leg, event.close, "mark_to_market", fx=fx)

    # -- publication ---------------------------------------------------------

    async def _publish_update(
        self,
        leg: LegPosition,
        mark_native: Decimal,
        trigger: Literal["fill", "mark_to_market", "fx_update", "reconciliation"],
        fx: Decimal | None = None,
    ) -> None:
        if fx is None:
            fx = self.get_usd_rate(leg.currency) or leg.last_fx
        leg.last_mark = mark_native
        leg.last_fx = fx
        unrealized = PnLCalculator.unrealized_pnl_usd(leg, mark_native, fx)
        notional = PnLCalculator.notional_usd(leg, mark_native, fx)
        await self._bus.publish(
            Topic.POSITIONS,
            PositionUpdateEvent(
                timestamp_exchange=self._clock.now(),
                timestamp_received=self._clock.now(),
                pair_id=leg.pair_id,
                ticker=leg.ticker,
                venue=leg.venue,  # type: ignore[arg-type]
                net_quantity=leg.net_quantity,
                average_entry_price=leg.avg_entry_price,
                average_entry_price_usd=leg.avg_entry_price_usd,
                unrealized_pnl_usd=unrealized,
                realized_pnl_usd=leg.realized_pnl_usd,
                mark_price=mark_native,
                mark_price_usd=mark_native * fx,
                notional_value_usd=notional,
                fx_rate=fx,
                is_short=leg.is_short,
                trigger=trigger,
            ),
        )

    # -- imbalance detection -------------------------------------------------

    def leg_imbalance(self, pair_id: str) -> str | None:
        """Return a description if the pair's legs are imbalanced, else ``None``."""
        pair = self._pairs.get(pair_id)
        if pair is None:
            return None
        adr = self._legs.get((pair_id, pair.adr_ticker))
        local = self._legs.get((pair_id, pair.underlying_ticker))
        adr_open = adr is not None and not adr.is_flat
        local_open = local is not None and not local.is_flat
        if adr_open and not local_open:
            return "naked_adr_short"
        if local_open and not adr_open:
            return "naked_local_long"
        return None

    def _check_imbalance(self, pair_id: str) -> None:
        detail = self.leg_imbalance(pair_id)
        if detail:
            logger.debug("leg_imbalance", extra={"pair_id": pair_id, "detail": detail})

    async def emit_imbalance_alert(self, pair_id: str) -> bool:
        """Publish a :class:`LegImbalanceEvent` if the pair is imbalanced."""
        detail = self.leg_imbalance(pair_id)
        if detail is None:
            return False
        await self._bus.publish(
            Topic.ALERTS,
            LegImbalanceEvent(
                timestamp_exchange=self._clock.now(),
                timestamp_received=self._clock.now(),
                pair_id=pair_id,
                detail=detail,
            ),
        )
        return True

    # -- introspection -------------------------------------------------------

    def leg(self, pair_id: str, ticker: str) -> LegPosition | None:
        return self._legs.get((pair_id, ticker))

    async def run(self) -> None:  # pragma: no cover - live entrypoint
        return None
