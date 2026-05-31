"""Alpaca REST order gateway — commission-free fallback (architecture §3.10).

Submits U.S. ADR orders via the Alpaca trading REST API (``httpx`` imported
lazily). Orders carry ``client_order_id = str(order.event_id)`` so fills arriving
on the trade-updates stream (:class:`AlpacaWsGateway`) can be correlated back to
the originating :class:`OrderRequest` via :meth:`handle_trade_update`.

Phase-5 live component; network paths are marked ``no cover``.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from asian_adr.core import Clock, OrderRequest, OrderType
from asian_adr.event_bus import AbstractEventBus

from ..base import AbstractGateway, ShortLocateProvider

logger = logging.getLogger(__name__)

__all__ = ["AlpacaRestGateway"]

_HOSTS = {
    "paper": "https://paper-api.alpaca.markets",
    "live": "https://api.alpaca.markets",
}


class AlpacaRestGateway(AbstractGateway):
    """U.S. ADR gateway submitting orders over the Alpaca REST API."""

    def __init__(
        self,
        bus: AbstractEventBus,
        clock: Clock,
        *,
        api_key: str = "",
        api_secret: str = "",
        environment: str = "paper",
        short_locate: ShortLocateProvider | None = None,
        timeout: float = 10.0,
    ) -> None:
        super().__init__(bus, clock, short_locate=short_locate, name="alpaca")
        self._key = api_key
        self._secret = api_secret
        self._host = _HOSTS.get(environment, _HOSTS["paper"])
        self._timeout = timeout
        self._orders_by_client_id: dict[str, OrderRequest] = {}

    def _headers(self) -> dict:
        return {"APCA-API-KEY-ID": self._key, "APCA-API-SECRET-KEY": self._secret}

    async def submit(self, order: OrderRequest) -> None:  # pragma: no cover - network
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("AlpacaRestGateway requires 'httpx' (uv add httpx).") from exc

        client_order_id = str(order.event_id)
        self._orders_by_client_id[client_order_id] = order
        payload = {
            "symbol": order.ticker,
            "qty": str(order.quantity),
            "side": order.side,
            "type": "limit" if order.order_type == OrderType.LIMIT else "market",
            "time_in_force": "day",
            "client_order_id": client_order_id,
        }
        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            payload["limit_price"] = str(order.limit_price)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._host}/v2/orders", headers=self._headers(), json=payload
                )
                resp.raise_for_status()
        except Exception as exc:
            raise ConnectionError(f"alpaca order submit failed: {exc}") from exc

    async def handle_trade_update(self, update: dict) -> None:
        """Correlate an Alpaca trade-update fill back to its order and publish."""
        if update.get("event") not in ("fill", "partial_fill"):
            return
        o = update.get("order", {})
        order = self._orders_by_client_id.get(o.get("client_order_id", ""))
        if order is None:
            return
        price = o.get("filled_avg_price") or o.get("limit_price")
        if price is None:
            return
        fill = self.make_fill(
            order,
            Decimal(str(price)),
            broker_order_id=str(o.get("id", "")),
            fill_quantity=Decimal(str(o.get("filled_qty", order.quantity))),
        )
        await self.publish_fill(fill)
