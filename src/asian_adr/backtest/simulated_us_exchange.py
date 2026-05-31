"""Simulated U.S. equity exchange for backtest fills.

Fills ``us_equity`` orders at the most recent bar seen for the order's ticker —
i.e. the bar whose signal generated the order, applying slippage and the U.S.
cost model. Because an order is created *after* its triggering bar has already
been dispatched, it fills at that bar's price (the ADR short fills at the U.S.
close that produced the signal), exactly as the paper executes.

The overnight gap is **not** modelled here — it is modelled by the execution
sequencer, which only places the local leg on the next Asian bar. This keeps
fill timing faithful while preserving the sequenced two-leg discipline.

Depends only on the standard library, ``asian_adr.core``, and ``event_bus``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from decimal import Decimal
from itertools import count
from uuid import uuid4

from asian_adr.core import BarEvent, Clock, FillEvent, OrderRequest, US_EQUITY
from asian_adr.event_bus import AbstractEventBus, Topic

from .cost_model import EquitiesCostModel
from .slippage_models import HalfSpreadSlippageModel, SlippageModel

logger = logging.getLogger(__name__)

__all__ = ["SimulatedUSExchange"]


class SimulatedUSExchange:
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
        self._slip = slippage_model or HalfSpreadSlippageModel()
        self._last_bar: dict[str, BarEvent] = {}
        self._pending: dict[str, list[OrderRequest]] = defaultdict(list)
        self._fill_seq = count(1)

    async def on_bar(self, bar: BarEvent) -> None:
        self._last_bar[bar.ticker] = bar
        for order in self._pending.pop(bar.ticker, []):
            await self._fill(order, bar)

    async def on_order(self, order: OrderRequest) -> None:
        if order.venue != US_EQUITY:
            return
        bar = self._last_bar.get(order.ticker)
        if bar is None:
            self._pending[order.ticker].append(order)  # fill on first bar
        else:
            await self._fill(order, bar)

    async def _fill(self, order: OrderRequest, bar: BarEvent) -> None:
        price = self._slip.calculate(bar, order.side, order.quantity)
        costs = self._cost.us_leg(order.side, order.quantity, price, order.is_short_sale)
        await self._bus.publish(
            Topic.FILLS,
            FillEvent(
                timestamp_exchange=bar.timestamp_exchange,
                timestamp_received=self._clock.now(),
                fill_id=f"us-{next(self._fill_seq)}",
                client_order_id=order.event_id,
                broker_order_id=str(uuid4()),
                pair_id=order.pair_id,
                ticker=order.ticker,
                side=order.side,
                fill_price=price,
                fill_price_usd=price,  # ADR leg is USD
                fill_quantity=order.quantity,
                remaining_quantity=Decimal("0"),
                commission=costs.commission,
                commission_currency="USD",
                sec_fee=costs.sec_fee,
                short_borrow_fee=costs.short_borrow_fee,
                fx_rate_used=Decimal("1"),
                is_short_sale=order.is_short_sale,
                venue=US_EQUITY,
                exchange=bar.exchange,
            ),
        )
