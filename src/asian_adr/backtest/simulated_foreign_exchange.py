"""Simulated Asian (foreign) equity exchange for backtest fills.

Mirror of :class:`SimulatedUSExchange` for the local leg. Fills
``foreign_equity`` orders at the most recent bar for the ticker, applies the
sqrt-impact slippage and the foreign cost model, and converts the native fill
price to USD via the cached FX rate (so ``fill_price_usd`` is what the
ROCE/RUCE calculator consumes).

Backtest only — there is no live foreign gateway; the local leg is executed by
the operator's own Asian brokerage in production.

Depends only on the standard library, ``asian_adr.core``, and ``event_bus``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from decimal import Decimal
from itertools import count
from uuid import uuid4

from asian_adr.core import (
    BarEvent,
    Clock,
    FillEvent,
    FOREIGN_EQUITY,
    FXRateEvent,
    OrderRequest,
)
from asian_adr.event_bus import AbstractEventBus, Topic

from .cost_model import EquitiesCostModel
from .slippage_models import SlippageModel, SqrtImpactSlippageModel

logger = logging.getLogger(__name__)

__all__ = ["SimulatedForeignExchange"]


class SimulatedForeignExchange:
    def __init__(
        self,
        bus: AbstractEventBus,
        clock: Clock,
        cost_model: EquitiesCostModel,
        slippage_model: SlippageModel | None = None,
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._cost = cost_model
        self._slip = slippage_model or SqrtImpactSlippageModel()
        self._last_bar: dict[str, BarEvent] = {}
        self._pending: dict[str, list[OrderRequest]] = defaultdict(list)
        self._fx: dict[str, Decimal] = {}
        self._fill_seq = count(1)

    async def on_fx_rate(self, event: FXRateEvent) -> None:
        self._fx[event.base_currency] = event.mid

    def _usd_rate(self, currency: str) -> Decimal | None:
        if currency == "USD":
            return Decimal("1")
        return self._fx.get(currency)

    async def on_bar(self, bar: BarEvent) -> None:
        self._last_bar[bar.ticker] = bar
        for order in self._pending.pop(bar.ticker, []):
            await self._fill(order, bar)

    async def on_order(self, order: OrderRequest) -> None:
        if order.venue != FOREIGN_EQUITY:
            return
        bar = self._last_bar.get(order.ticker)
        if bar is None:
            self._pending[order.ticker].append(order)
        else:
            await self._fill(order, bar)

    async def _fill(self, order: OrderRequest, bar: BarEvent) -> None:
        fx = self._usd_rate(bar.currency)
        if fx is None:
            logger.warning(
                "no FX rate for %s; skipping fill for %s", bar.currency, order.ticker
            )
            self._pending[order.ticker].append(order)
            return
        price = self._slip.calculate(bar, order.side, order.quantity)
        costs = self._cost.foreign_leg(bar.exchange, order.side, order.quantity, price)
        await self._bus.publish(
            Topic.FILLS,
            FillEvent(
                timestamp_exchange=bar.timestamp_exchange,
                timestamp_received=self._clock.now(),
                fill_id=f"fx-{next(self._fill_seq)}",
                client_order_id=order.event_id,
                broker_order_id=str(uuid4()),
                pair_id=order.pair_id,
                ticker=order.ticker,
                side=order.side,
                fill_price=price,
                fill_price_usd=price * fx,
                fill_quantity=order.quantity,
                remaining_quantity=Decimal("0"),
                commission=costs.commission,
                commission_currency=bar.currency,
                stamp_duty=costs.stamp_duty,
                fx_rate_used=fx,
                is_short_sale=False,
                venue=FOREIGN_EQUITY,
                exchange=bar.exchange,
            ),
        )
