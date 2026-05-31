"""Event-bus interface, handler type, and canonical topic names.

Every component talks to the rest of the system **only** through a bus that
satisfies :class:`AbstractEventBus`. The same async surface is honoured by all
three implementations so that the Hong & Susmel engine, risk engine, position
engine, and sequencer are byte-for-byte identical across backtest and live
modes — only the concrete bus differs:

* :class:`~asian_adr.event_bus.memory_bus.MemoryBus` — deterministic, drained
  with :meth:`flush` after each replayed event (backtest + unit tests).
* :class:`~asian_adr.event_bus.asyncio_bus.AsyncioBus` — one ``asyncio.Queue``
  per topic with backpressure on critical topics (live single-process).
* :class:`~asian_adr.event_bus.kafka_bus.KafkaBus` — durable, replayable,
  out-of-process (production / Phase 4+).

This module depends only on the standard library and ``asian_adr.core``.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Protocol, Union, runtime_checkable

from asian_adr.core import BaseEvent

__all__ = ["EventHandler", "AbstractEventBus", "Topic", "CRITICAL_TOPICS"]


# A handler may be a plain callable or a coroutine function. Buses await the
# return value when it is awaitable, so both styles compose transparently.
EventHandler = Callable[[BaseEvent], Union[Awaitable[None], None]]


class Topic:
    """Canonical topic names (architecture §3.6 / data-contracts topic map).

    Using these constants instead of bare string literals keeps producers and
    consumers in sync and makes typos a static error at call sites that import
    from here.
    """

    MARKET_DATA = "market-data"
    FX_RATES = "fx-rates"
    SIGNALS = "signals"
    RISK_DECISIONS = "risk-decisions"
    ORDERS = "orders"
    FILLS = "fills"
    POSITIONS = "positions"
    PAIR_REGISTRY = "pair-registry"
    ALERTS = "alerts"

    ALL: frozenset[str] = frozenset(
        {
            MARKET_DATA,
            FX_RATES,
            SIGNALS,
            RISK_DECISIONS,
            ORDERS,
            FILLS,
            POSITIONS,
            PAIR_REGISTRY,
            ALERTS,
        }
    )


# Topics whose delivery must never be silently dropped under backpressure;
# publishers block rather than discard (see AsyncioBus).
CRITICAL_TOPICS: frozenset[str] = frozenset(
    {Topic.ORDERS, Topic.FILLS, Topic.RISK_DECISIONS}
)


@runtime_checkable
class AbstractEventBus(Protocol):
    """Pub/sub interface shared by every bus implementation.

    Contract guarantees expected of all implementations:

    * **Ordered delivery within a topic** — events are delivered to a topic's
      handlers in publish order.
    * **Handler-registration order** — when a topic has multiple handlers they
      are invoked in the order they subscribed.
    * **Sync or async handlers** — a handler returning an awaitable is awaited.
    """

    async def publish(self, topic: str, event: BaseEvent) -> None:
        """Publish ``event`` to ``topic`` for delivery to its subscribers."""
        ...

    async def subscribe(self, topic: str, handler: EventHandler) -> None:
        """Register ``handler`` to receive every event published to ``topic``."""
        ...

    async def subscribe_many(
        self, topics: list[str], handler: EventHandler
    ) -> None:
        """Register one ``handler`` across several ``topics`` at once."""
        ...

    async def flush(self) -> None:
        """Block until all currently-queued events have been delivered.

        A synchronisation barrier. For :class:`MemoryBus` this drains the
        pending queue (including any cascades handlers produce); for queue-based
        buses it waits for the per-topic queues to empty.
        """
        ...

    async def close(self) -> None:
        """Release resources (consumer tasks, broker connections)."""
        ...
