"""Durable, replayable Kafka event bus for production (Phase 4+).

``KafkaBus`` swaps the in-process queue for Kafka topics so each component can
run as its own process / pod while retaining the exact same
:class:`AbstractEventBus` surface the engines already use. Events are
serialised as their pydantic JSON and reconstructed on the consumer side via a
small ``event_type`` → model registry.

``aiokafka`` is imported lazily inside :meth:`start`, so importing this module
(and the rest of ``event_bus``) never requires the dependency to be installed —
backtest and unit-test environments stay lean. A clear, actionable error is
raised only if you actually try to start a Kafka bus without the library.

Depends on the standard library and ``asian_adr.core``; ``aiokafka`` is an
optional runtime dependency.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Type

from asian_adr.core import (
    BarEvent,
    BaseEvent,
    FillEvent,
    FXRateEvent,
    HongSusmelSignalEvent,
    OrderRequest,
    PositionUpdateEvent,
    RiskDecision,
)

from .base import EventHandler

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aiokafka import AIOKafkaProducer

logger = logging.getLogger(__name__)

__all__ = ["KafkaBus", "EVENT_REGISTRY", "register_event"]


# Maps the ``event_type`` discriminator carried by every event to its model
# class, so a JSON payload off a Kafka topic can be reconstructed into the
# correct frozen event. Extend with :func:`register_event` for new event types.
EVENT_REGISTRY: dict[str, Type[BaseEvent]] = {
    cls.model_fields["event_type"].default: cls  # type: ignore[union-attr]
    for cls in (
        BarEvent,
        FXRateEvent,
        HongSusmelSignalEvent,
        RiskDecision,
        OrderRequest,
        FillEvent,
        PositionUpdateEvent,
    )
}


def register_event(cls: Type[BaseEvent]) -> Type[BaseEvent]:
    """Register an event class for deserialization (usable as a decorator)."""
    event_type = cls.model_fields["event_type"].default
    EVENT_REGISTRY[event_type] = cls
    return cls


def _deserialize(payload: bytes) -> BaseEvent:
    """Reconstruct an event from its JSON bytes using :data:`EVENT_REGISTRY`."""
    import json

    raw = json.loads(payload)
    event_type = raw.get("event_type")
    cls = EVENT_REGISTRY.get(event_type)
    if cls is None:
        raise ValueError(f"no event class registered for event_type={event_type!r}")
    return cls.model_validate(raw)


class KafkaBus:
    """Kafka-backed implementation of :class:`AbstractEventBus`.

    Parameters
    ----------
    bootstrap_servers:
        Kafka broker list, e.g. ``"localhost:9092"``.
    group_id:
        Consumer group; components sharing a group load-balance a topic, while
        distinct groups each receive the full stream.
    topic_prefix:
        Optional namespace prepended to every topic (e.g. ``"adr."``).
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        *,
        group_id: str = "asian-adr",
        topic_prefix: str = "",
    ) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._group_id = group_id
        self._topic_prefix = topic_prefix
        self._producer: "AIOKafkaProducer | None" = None
        self._handlers: dict[str, list[EventHandler]] = {}
        self._consumers: dict[str, asyncio.Task[None]] = {}
        self._started = False

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Create and start the Kafka producer (idempotent)."""
        if self._started:
            return
        try:
            from aiokafka import AIOKafkaProducer
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "KafkaBus requires the 'aiokafka' package. "
                "Install it with `uv add aiokafka` (production / Phase 4+)."
            ) from exc
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap_servers
        )
        await self._producer.start()
        self._started = True

    def _full_topic(self, topic: str) -> str:
        return f"{self._topic_prefix}{topic}"

    # -- publish -------------------------------------------------------------

    async def publish(self, topic: str, event: BaseEvent) -> None:
        if not self._started:
            await self.start()
        assert self._producer is not None
        payload = event.model_dump_json().encode("utf-8")
        await self._producer.send_and_wait(self._full_topic(topic), payload)

    # -- subscription --------------------------------------------------------

    async def subscribe(self, topic: str, handler: EventHandler) -> None:
        self._handlers.setdefault(topic, []).append(handler)
        if topic not in self._consumers:
            self._consumers[topic] = asyncio.create_task(
                self._consume(topic), name=f"kafka_bus-consumer-{topic}"
            )

    async def subscribe_many(
        self, topics: list[str], handler: EventHandler
    ) -> None:
        for topic in topics:
            await self.subscribe(topic, handler)

    async def _consume(self, topic: str) -> None:  # pragma: no cover - needs broker
        from aiokafka import AIOKafkaConsumer

        consumer = AIOKafkaConsumer(
            self._full_topic(topic),
            bootstrap_servers=self._bootstrap_servers,
            group_id=self._group_id,
            enable_auto_commit=True,
        )
        await consumer.start()
        try:
            async for message in consumer:
                try:
                    event = _deserialize(message.value)
                except Exception:
                    logger.exception(
                        "kafka_bus deserialize error", extra={"topic": topic}
                    )
                    continue
                for handler in tuple(self._handlers.get(topic, ())):
                    try:
                        result = handler(event)
                        if inspect.isawaitable(result):
                            await result
                    except Exception:
                        logger.exception(
                            "kafka_bus handler error", extra={"topic": topic}
                        )
        finally:
            await consumer.stop()

    # -- lifecycle barriers --------------------------------------------------

    async def flush(self) -> None:
        """Flush the producer's outstanding sends."""
        if self._producer is not None:
            await self._producer.flush()

    async def close(self) -> None:
        for task in self._consumers.values():
            task.cancel()
        if self._consumers:
            await asyncio.gather(
                *self._consumers.values(), return_exceptions=True
            )
        self._consumers.clear()
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None
        self._started = False

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"KafkaBus(servers={self._bootstrap_servers!r}, "
            f"group={self._group_id!r}, started={self._started})"
        )
