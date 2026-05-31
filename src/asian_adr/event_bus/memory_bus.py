"""Deterministic in-memory event bus for backtest and unit tests.

``MemoryBus`` is the bus used whenever determinism matters more than
concurrency. It holds no ``asyncio`` machinery of its own: :meth:`publish`
simply appends to an in-order FIFO queue, and :meth:`flush` drains that queue,
dispatching each event to its topic's handlers in registration order.

Cascades are handled naturally: a handler may itself publish more events while
being dispatched; those land at the tail of the queue and the drain loop keeps
going until the queue is empty. The backtest replay loop therefore publishes
each market/FX event then calls ``flush()`` to run the full signal → risk →
order → fill → position cascade to completion before advancing the clock — the
``flush()``-after-each-event discipline that makes runs reproducible.

The methods are ``async`` to match :class:`AbstractEventBus`, but they never
yield to an event loop; dispatch is inline and ordered. This keeps the engines
identical across backtest and live without paying for real concurrency here.

Depends only on the standard library and ``asian_adr.core``.
"""

from __future__ import annotations

import inspect
import logging
from collections import defaultdict, deque

from asian_adr.core import BaseEvent

from .base import EventHandler

logger = logging.getLogger(__name__)

__all__ = ["MemoryBus"]


class MemoryBus:
    """Synchronous, deterministic pub/sub bus.

    Parameters
    ----------
    raise_on_handler_error:
        When ``True`` (default) an exception raised by a handler propagates out
        of :meth:`flush`, failing the run fast — the right behaviour for a
        backtest, where a swallowed error would silently corrupt results.
    """

    def __init__(self, *, raise_on_handler_error: bool = True) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._queue: deque[tuple[str, BaseEvent]] = deque()
        self._raise_on_handler_error = raise_on_handler_error
        self._draining = False
        self._published_count = 0
        self._delivered_count = 0
        self._dropped_count = 0

    # -- subscription --------------------------------------------------------

    async def subscribe(self, topic: str, handler: EventHandler) -> None:
        handlers = self._handlers[topic]
        if handler not in handlers:
            handlers.append(handler)

    async def subscribe_many(
        self, topics: list[str], handler: EventHandler
    ) -> None:
        for topic in topics:
            await self.subscribe(topic, handler)

    def unsubscribe(self, topic: str, handler: EventHandler) -> None:
        """Remove a previously registered handler (no error if absent)."""
        if handler in self._handlers.get(topic, []):
            self._handlers[topic].remove(handler)

    # -- publish / drain -----------------------------------------------------

    async def publish(self, topic: str, event: BaseEvent) -> None:
        self._queue.append((topic, event))
        self._published_count += 1

    async def flush(self) -> None:
        """Drain the queue, dispatching every event (and any cascades).

        Re-entrant calls are a no-op: if a handler triggers ``flush`` while a
        drain is already in progress, the outer loop already owns the queue.
        """
        if self._draining:
            return
        self._draining = True
        try:
            while self._queue:
                topic, event = self._queue.popleft()
                handlers = self._handlers.get(topic)
                if not handlers:
                    self._dropped_count += 1
                    continue
                # Snapshot so a handler that (un)subscribes mid-dispatch does
                # not mutate the list we are iterating.
                for handler in tuple(handlers):
                    await self._dispatch(handler, event, topic)
        finally:
            self._draining = False

    async def _dispatch(
        self, handler: EventHandler, event: BaseEvent, topic: str
    ) -> None:
        try:
            result = handler(event)
            if inspect.isawaitable(result):
                await result
            self._delivered_count += 1
        except Exception:
            logger.exception(
                "memory_bus handler error", extra={"topic": topic}
            )
            if self._raise_on_handler_error:
                raise

    async def close(self) -> None:
        """Drop any undelivered events and clear handlers."""
        self._queue.clear()
        self._handlers.clear()

    # -- introspection (tests / metrics) -------------------------------------

    @property
    def pending(self) -> int:
        """Number of events queued but not yet delivered."""
        return len(self._queue)

    @property
    def published_count(self) -> int:
        return self._published_count

    @property
    def delivered_count(self) -> int:
        return self._delivered_count

    @property
    def dropped_count(self) -> int:
        """Events published to a topic with no subscribers."""
        return self._dropped_count

    def handler_count(self, topic: str) -> int:
        return len(self._handlers.get(topic, []))

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"MemoryBus(pending={self.pending}, "
            f"published={self._published_count}, delivered={self._delivered_count})"
        )
