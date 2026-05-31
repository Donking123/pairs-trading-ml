"""Event-bus layer: the pub/sub backbone every component communicates through.

Exposes the interface (:class:`AbstractEventBus`), the canonical topic names
(:class:`Topic`), and the three implementations. ``KafkaBus`` is imported
lazily — referencing it does not import ``aiokafka`` until the bus is started.
"""

from __future__ import annotations

from .asyncio_bus import AsyncioBus
from .base import CRITICAL_TOPICS, AbstractEventBus, EventHandler, Topic
from .kafka_bus import KafkaBus
from .memory_bus import MemoryBus

__all__ = [
    "AbstractEventBus",
    "EventHandler",
    "Topic",
    "CRITICAL_TOPICS",
    "MemoryBus",
    "AsyncioBus",
    "KafkaBus",
]
