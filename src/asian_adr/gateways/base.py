"""Broker gateway interface and shared types (architecture §3.10).

A gateway is U.S.-equity-only: it consumes :class:`~asian_adr.core.OrderRequest`
events for the ADR leg (short SELL / cover BUY), verifies a short-locate before
any short sale, submits to the broker, and publishes :class:`~asian_adr.core.FillEvent`s
back on ``fills``. Foreign-equity orders are ignored — the local Asian leg is
executed via the operator's own brokerage (instruction logged, execution
external).

``AbstractGateway`` centralises the venue filter, locate check, rejection
handling, and fill construction; concrete gateways implement :meth:`submit`.

Imports only ``core`` and ``event_bus``.
"""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from itertools import count
from typing import Awaitable, Callable, Literal, Protocol, runtime_checkable
from uuid import uuid4

from asian_adr.core import (
    BaseEvent,
    Clock,
    FillEvent,
    OrderRequest,
    US_EQUITY,
)
from asian_adr.event_bus import AbstractEventBus, Topic

logger = logging.getLogger(__name__)

__all__ = [
    "LocateResult",
    "ShortLocateProvider",
    "AlwaysAvailableLocate",
    "OrderRejectedEvent",
    "AbstractGateway",
    "reconnect_with_backoff",
]


@dataclass(frozen=True)
class LocateResult:
    """Outcome of a short-locate query."""

    available: bool
    borrow_rate: Decimal = Decimal("0")


@runtime_checkable
class ShortLocateProvider(Protocol):
    async def verify(self, ticker: str, quantity: Decimal) -> LocateResult: ...


class AlwaysAvailableLocate:
    """Default locate provider — always available at zero borrow (sim/paper)."""

    async def verify(self, ticker: str, quantity: Decimal) -> LocateResult:
        return LocateResult(available=True, borrow_rate=Decimal("0"))


class OrderRejectedEvent(BaseEvent):
    """Published to ``alerts`` when an order cannot be submitted/filled."""

    event_type: Literal["order_rejected"] = "order_rejected"
    pair_id: str | None
    ticker: str
    side: Literal["buy", "sell"]
    reason: str


async def reconnect_with_backoff(
    connect_fn: Callable[[], Awaitable[None]],
    *,
    max_retries: int = 10,
    max_delay: float = 60.0,
) -> None:
    """Exponential-backoff reconnect with jitter (architecture §10.4)."""
    for attempt in range(max_retries):
        try:
            await connect_fn()
            return
        except ConnectionError:
            if attempt == max_retries - 1:
                raise
            delay = min(2 ** attempt + random.uniform(0, 1), max_delay)
            logger.warning("gateway reconnect attempt=%d delay=%.1fs", attempt, delay)
            await asyncio.sleep(delay)


class AbstractGateway(ABC):
    """Base for U.S. ADR broker gateways."""

    venue = US_EQUITY

    def __init__(
        self,
        bus: AbstractEventBus,
        clock: Clock,
        *,
        short_locate: ShortLocateProvider | None = None,
        name: str = "gateway",
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._locate = short_locate or AlwaysAvailableLocate()
        self._name = name
        self._fill_seq = count(1)
        # internal_client_id (UUID) ↔ broker order id
        self._broker_to_client: dict[str, object] = {}

    # -- order intake --------------------------------------------------------

    async def on_order(self, order: OrderRequest) -> None:
        """Validate and route an order; ignores non-U.S.-equity venues."""
        if order.venue != self.venue:
            return  # foreign leg handled externally
        if order.side == "sell" and order.is_short_sale:
            locate = await self._locate.verify(order.ticker, order.quantity)
            if not locate.available:
                await self.reject(order, "short-locate unavailable")
                return
        try:
            await self.submit(order)
        except ConnectionError:
            raise
        except Exception as exc:  # pragma: no cover - broker-specific
            await self.reject(order, f"submit failed: {exc}")

    @abstractmethod
    async def submit(self, order: OrderRequest) -> None:
        """Broker-specific submission (may fill synchronously or via callback)."""

    async def run(self) -> None:  # pragma: no cover - overridden by live gateways
        """Connection / fill-stream lifecycle. No-op for synchronous gateways."""
        return None

    # -- helpers -------------------------------------------------------------

    async def reject(self, order: OrderRequest, reason: str) -> None:
        logger.warning("%s rejected %s %s: %s", self._name, order.side, order.ticker, reason)
        await self._bus.publish(
            Topic.ALERTS,
            OrderRejectedEvent(
                timestamp_exchange=order.timestamp_exchange,
                timestamp_received=self._clock.now(),
                pair_id=order.pair_id,
                ticker=order.ticker,
                side=order.side,
                reason=reason,
            ),
        )

    def make_fill(
        self,
        order: OrderRequest,
        fill_price: Decimal,
        *,
        commission: Decimal = Decimal("0"),
        sec_fee: Decimal = Decimal("0"),
        short_borrow_fee: Decimal = Decimal("0"),
        broker_order_id: str | None = None,
        fill_quantity: Decimal | None = None,
        exchange: str = "US",
    ) -> FillEvent:
        """Construct a U.S.-equity :class:`FillEvent` (USD; price_usd == price)."""
        qty = fill_quantity if fill_quantity is not None else order.quantity
        broker_id = broker_order_id or str(uuid4())
        self._broker_to_client[broker_id] = order.event_id
        return FillEvent(
            timestamp_exchange=order.timestamp_exchange,
            timestamp_received=self._clock.now(),
            fill_id=f"{self._name}-{next(self._fill_seq)}",
            client_order_id=order.event_id,
            broker_order_id=broker_id,
            pair_id=order.pair_id,
            ticker=order.ticker,
            side=order.side,
            fill_price=fill_price,
            fill_price_usd=fill_price,
            fill_quantity=qty,
            remaining_quantity=order.quantity - qty,
            commission=commission,
            commission_currency="USD",
            sec_fee=sec_fee,
            short_borrow_fee=short_borrow_fee,
            fx_rate_used=Decimal("1"),
            is_short_sale=order.is_short_sale,
            venue=US_EQUITY,
            exchange=exchange,
        )

    async def publish_fill(self, fill: FillEvent) -> None:
        await self._bus.publish(Topic.FILLS, fill)
