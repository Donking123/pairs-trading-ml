"""Record live bus events and replay them (architecture Phase 5).

:class:`DataRecorder` subscribes to chosen topics and appends every event to a
JSONL file (one ``{"topic", "event"}`` object per line) — capturing a live or
paper session for later deterministic replay or backtesting.

:class:`ReplayProducer` is the inverse: it reads such a recording and republishes
the events onto a bus, reconstructing each into its frozen event type via the
shared event registry. Together they let :class:`~asian_adr.runners.live_runner.LiveRunner`
run end-to-end against captured data without any live feed or broker.

Uses the standard library, ``asian_adr.core``, and ``asian_adr.event_bus``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Iterable

from asian_adr.core import BaseEvent
from asian_adr.event_bus import AbstractEventBus, Topic
from asian_adr.event_bus.kafka_bus import EVENT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["DataRecorder", "ReplayProducer", "DEFAULT_RECORD_TOPICS"]

DEFAULT_RECORD_TOPICS: tuple[str, ...] = (
    Topic.MARKET_DATA,
    Topic.FX_RATES,
    Topic.SIGNALS,
    Topic.ORDERS,
    Topic.FILLS,
    Topic.POSITIONS,
)


class DataRecorder:
    """Appends bus events to a JSONL recording."""

    def __init__(
        self,
        path: str | Path,
        topics: Iterable[str] = DEFAULT_RECORD_TOPICS,
    ) -> None:
        self._path = Path(path)
        self._topics = tuple(topics)
        self._fh = None
        self.count = 0

    async def attach(self, bus: AbstractEventBus) -> None:
        """Open the sink and subscribe to the recorded topics."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", encoding="utf-8")
        for topic in self._topics:
            await bus.subscribe(topic, self._make_handler(topic))
        logger.info("recording %s → %s", list(self._topics), self._path)

    def _make_handler(self, topic: str):
        def handler(event: BaseEvent) -> None:
            if self._fh is None:
                return
            line = json.dumps({"topic": topic, "event": event.model_dump(mode="json")})
            self._fh.write(line + "\n")
            self.count += 1

        return handler

    async def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
            logger.info("recorded %d events to %s", self.count, self._path)


class ReplayProducer:
    """Republishes a JSONL recording onto a bus (a drop-in 'feed' for replay)."""

    def __init__(
        self,
        bus: AbstractEventBus,
        path: str | Path,
        *,
        topics: Iterable[str] | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self._bus = bus
        self._path = Path(path)
        self._topics = set(topics) if topics is not None else None
        self._delay = delay_seconds

    async def run(self) -> None:
        if not self._path.exists():
            logger.warning("replay file %s not found; nothing to replay", self._path)
            return
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                topic = record["topic"]
                if self._topics is not None and topic not in self._topics:
                    continue
                event = self._deserialize(record["event"])
                if event is None:
                    continue
                await self._bus.publish(topic, event)
                await self._bus.flush()
                if self._delay:
                    await asyncio.sleep(self._delay)

    @staticmethod
    def _deserialize(payload: dict) -> BaseEvent | None:
        cls = EVENT_REGISTRY.get(payload.get("event_type"))
        if cls is None:
            logger.debug("no event class for %s; skipping", payload.get("event_type"))
            return None
        return cls.model_validate(payload)
