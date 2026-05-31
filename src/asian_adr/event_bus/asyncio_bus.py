"""Single-process asyncio event bus for the live prototype (Phase 1–2).

``AsyncioBus`` gives each topic its own bounded :class:`asyncio.Queue` and runs
one background consumer task per topic. Publishers enqueue without blocking
when there is room; when a queue is full the bus applies the architecture's
backpressure policy (§8.2):

* **Critical topics** (``orders``, ``fills``, ``risk-decisions``) — the
  publisher *blocks* until space frees up, so these events are never dropped.
* **All other topics** — the event is dropped and counted, trading
  completeness for liveness on a saturated non-critical stream.

Within a topic, ordered delivery is preserved (single FIFO queue, single
consumer). Handlers registered on the same topic are invoked in subscription
order, and a handler returning an awaitable is awaited.

Depends only on the standard library and ``asian_adr.core``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict

from asian_adr.core import BaseEvent

from .base import CRITICAL_TOPICS, EventHandler

logger = logging.getLogger(__name__)

__all__ = ["AsyncioBus"]


class AsyncioBus:
    """Per-topic ``asyncio.Queue`` bus with backpressure on critical topics.

    Parameters
    ----------
    max_queue_size:
        Per-topic queue capacity before backpressure engages.
    """

    def __init__(self, max_queue_size: int = 10_000) -> None:
        self._max_queue_size = max_queue_size
        self._queues: dict[str, asyncio.Queue[BaseEvent]] = defaultdict(
            lambda: asyncio.Queue(maxsize=max_queue_size)
        )
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._consumers: dict[str, asyncio.Task[None]] = {}
        self._closed = False
        self._dropped_count = 0

    # -- subscription --------------------------------------------------------

    async def subscribe(self, topic: str, handler: EventHandler) -> None:
        handlers = self._handlers[topic]
        if handler not in handlers:
            handlers.append(handler)
        # Lazily start the consumer the first time a topic gains a subscriber.
        if topic not in self._consumers:
            self._consumers[topic] = asyncio.create_task(
                self._consume(topic), name=f"asyncio_bus-consumer-{topic}"
            )

    async def subscribe_many(
        self, topics: list[str], handler: EventHandler
    ) -> None:
        for topic in topics:
            await self.subscribe(topic, handler)

    # -- publish -------------------------------------------------------------

    async def publish(self, topic: str, event: BaseEvent) -> None:
        if self._closed:
            raise RuntimeError("cannot publish on a closed AsyncioBus")
        queue = self._queues[topic]
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("bus_queue_full", extra={"topic": topic})
            if topic in CRITICAL_TOPICS:
                await queue.put(event)  # block until space frees up
            else:
                self._dropped_count += 1

    # -- consumer loop -------------------------------------------------------

    async def _consume(self, topic: str) -> None:
        queue = self._queues[topic]
        while True:
            event = await queue.get()
            try:
                for handler in tuple(self._handlers.get(topic, ())):
                    try:
                        result = handler(event)
                        if inspect.isawaitable(result):
                            await result
                    except Exception:
                        logger.exception(
                            "asyncio_bus handler error", extra={"topic": topic}
                        )
            finally:
                queue.task_done()

    # -- lifecycle -----------------------------------------------------------

    async def flush(self) -> None:
        """Wait until every topic queue has been fully processed."""
        await asyncio.gather(*(q.join() for q in self._queues.values()))

    async def close(self) -> None:
        """Cancel all consumer tasks and mark the bus closed."""
        self._closed = True
        for task in self._consumers.values():
            task.cancel()
        if self._consumers:
            await asyncio.gather(
                *self._consumers.values(), return_exceptions=True
            )
        self._consumers.clear()

    # -- introspection -------------------------------------------------------

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    def queue_depth(self, topic: str) -> int:
        return self._queues[topic].qsize() if topic in self._queues else 0

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"AsyncioBus(topics={list(self._queues)}, "
            f"dropped={self._dropped_count}, closed={self._closed})"
        )
