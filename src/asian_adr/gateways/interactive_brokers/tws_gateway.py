"""Interactive Brokers TWS gateway — primary U.S. ADR broker (architecture §3.10).

Connects to the TWS / IB Gateway via ``ib_insync`` (imported lazily, so this
module is import-safe without it), submits market/limit orders for the ADR leg,
maps IB order IDs to internal client IDs, and republishes IB executions as
:class:`FillEvent`s. Short SELLs are gated by :class:`IBShortLocate` through the
:class:`AbstractGateway` flow.

This is a Phase-5 live component; the IB-specific paths are exercised only
against a running TWS/Gateway and are marked ``no cover``.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from asian_adr.core import Clock, OrderRequest, OrderType
from asian_adr.event_bus import AbstractEventBus

from ..base import AbstractGateway, reconnect_with_backoff
from .short_locate import IBShortLocate

logger = logging.getLogger(__name__)

__all__ = ["InteractiveBrokersGateway"]


class InteractiveBrokersGateway(AbstractGateway):
    """Primary U.S. ADR gateway over the IB TWS API."""

    def __init__(
        self,
        bus: AbstractEventBus,
        clock: Clock,
        *,
        host: str = "127.0.0.1",
        tws_port: int = 7497,
        client_id: int = 1,
    ) -> None:
        self._locate_provider = IBShortLocate()
        super().__init__(bus, clock, short_locate=self._locate_provider, name="ib")
        self._host = host
        self._port = tws_port
        self._client_id = client_id
        self._ib = None
        self._pending_orders: dict[str, OrderRequest] = {}

    # -- connection ----------------------------------------------------------

    async def _connect(self) -> None:  # pragma: no cover - requires TWS
        try:
            from ib_insync import IB
        except ImportError as exc:
            raise RuntimeError(
                "InteractiveBrokersGateway requires 'ib_insync' (uv add ib_insync)."
            ) from exc
        ib = IB()
        try:
            await ib.connectAsync(self._host, self._port, clientId=self._client_id)
        except Exception as exc:
            raise ConnectionError(f"IB connect failed: {exc}") from exc
        self._ib = ib
        self._locate_provider.set_ib(ib)
        ib.execDetailsEvent += self._on_exec_details

    async def run(self) -> None:  # pragma: no cover - requires TWS
        await reconnect_with_backoff(self._connect)
        # ib_insync drives callbacks on the asyncio loop; idle until cancelled.
        import asyncio

        while True:
            await asyncio.sleep(3600)

    # -- order submission ----------------------------------------------------

    async def submit(self, order: OrderRequest) -> None:  # pragma: no cover - requires TWS
        if self._ib is None:
            raise ConnectionError("IB gateway not connected")
        from ib_insync import LimitOrder, MarketOrder, Stock

        contract = Stock(order.ticker, "SMART", "USD")
        action = "BUY" if order.side == "buy" else "SELL"
        qty = float(order.quantity)
        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            ib_order = LimitOrder(action, qty, float(order.limit_price))
        else:
            ib_order = MarketOrder(action, qty)
        trade = self._ib.placeOrder(contract, ib_order)
        self._broker_to_client[str(trade.order.orderId)] = order.event_id
        self._pending_orders[str(trade.order.orderId)] = order

    # -- fill callback -------------------------------------------------------

    def _on_exec_details(self, trade, fill) -> None:  # pragma: no cover - requires TWS
        import asyncio

        order = self._pending_orders.get(str(trade.order.orderId))
        if order is None:
            return
        execution = fill.execution
        event = self.make_fill(
            order,
            Decimal(str(execution.price)),
            commission=Decimal(str(getattr(fill.commissionReport, "commission", 0) or 0)),
            broker_order_id=str(trade.order.orderId),
            fill_quantity=Decimal(str(execution.shares)),
            exchange=str(execution.exchange or "US"),
        )
        asyncio.create_task(self.publish_fill(event))
