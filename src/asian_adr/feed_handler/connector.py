"""Connection lifecycle and the feed-handler orchestrator (architecture §3.2).

* :func:`reconnect_with_backoff` — the §10.4 exponential-backoff reconnect.
* :class:`AbstractConnector` — the provider-agnostic connection interface:
  connect → subscribe → stream raw messages → disconnect.
* :class:`IterableConnector` — an in-memory connector over a fixed message list,
  used for tests and offline replay.
* :class:`FeedHandler` — the ``Producer`` that ties a connector + normaliser +
  subscription manager + staleness monitor together: it connects, normalises
  each raw message into a ``BarEvent``, runs staleness checks, and publishes to
  ``market-data`` (and zero-return/gap alerts to ``alerts``), reconnecting on
  ``ConnectionError``.

Imports only ``core`` and ``event_bus`` — never ``strategy`` / ``risk`` /
``gateways`` (design rule).
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import AsyncIterator, Awaitable, Callable, Iterable, Protocol, runtime_checkable

from asian_adr.core import Clock
from asian_adr.event_bus import AbstractEventBus, Topic

from .normalizer import BarNormalizer
from .staleness_monitor import StalenessMonitor
from .subscription_manager import SubscriptionManager

logger = logging.getLogger(__name__)

__all__ = [
    "reconnect_with_backoff",
    "AbstractConnector",
    "IterableConnector",
    "FeedHandler",
]


async def reconnect_with_backoff(
    connect_fn: Callable[[], Awaitable[None]],
    *,
    max_retries: int = 10,
    max_delay: float = 60.0,
) -> None:
    """Call ``connect_fn`` with exponential backoff + jitter (architecture §10.4)."""
    for attempt in range(max_retries):
        try:
            await connect_fn()
            return
        except ConnectionError:
            if attempt == max_retries - 1:
                raise
            delay = min(2 ** attempt + random.uniform(0, 1), max_delay)
            logger.warning("reconnect_attempt attempt=%d delay=%.1fs", attempt, delay)
            await asyncio.sleep(delay)


@runtime_checkable
class AbstractConnector(Protocol):
    """Provider connection lifecycle."""

    async def connect(self) -> None: ...
    async def subscribe(self, tickers: Iterable[str]) -> None: ...
    def messages(self) -> AsyncIterator[dict]: ...
    async def disconnect(self) -> None: ...


class IterableConnector:
    """In-memory connector over a fixed message list (tests / offline replay)."""

    def __init__(self, messages: Iterable[dict]) -> None:
        self._messages = list(messages)
        self.connected = False
        self.subscribed: set[str] = set()

    async def connect(self) -> None:
        self.connected = True

    async def subscribe(self, tickers: Iterable[str]) -> None:
        self.subscribed |= set(tickers)

    async def messages(self) -> AsyncIterator[dict]:
        for msg in self._messages:
            yield msg

    async def disconnect(self) -> None:
        self.connected = False


class FeedHandler:
    """Orchestrates one provider feed into ``BarEvent``s on the bus."""

    def __init__(
        self,
        bus: AbstractEventBus,
        clock: Clock,
        connector: AbstractConnector,
        normalizer: BarNormalizer,
        subscription: SubscriptionManager,
        *,
        staleness: StalenessMonitor | None = None,
        name: str = "feed",
        max_retries: int = 10,
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._connector = connector
        self._normalizer = normalizer
        self._subscription = subscription
        self._staleness = staleness or StalenessMonitor(clock)
        self._name = name
        self._max_retries = max_retries
        self._running = False
        self._connected = False
        self.bars_published = 0

    # -- registry-driven subscription ---------------------------------------

    async def on_registry_update(self, event: object) -> None:
        added, removed = self._subscription.update_from_registry(event)
        if self._connected and (added or removed):
            await self._connector.subscribe(self._subscription.tickers)

    # -- run loop ------------------------------------------------------------

    async def run(self) -> None:
        """Connect, stream, normalise, and publish until stopped or exhausted."""
        self._running = True
        try:
            while self._running:
                try:
                    await reconnect_with_backoff(
                        self._connector.connect, max_retries=self._max_retries
                    )
                    self._connected = True
                    await self._connector.subscribe(self._subscription.tickers)
                    async for raw in self._connector.messages():
                        await self._handle_raw(raw)
                        if not self._running:
                            break
                    break  # message stream ended cleanly (finite source)
                except ConnectionError as exc:
                    logger.warning("%s disconnected, reconnecting: %s", self._name, exc)
                    self._connected = False
                    await self._safe_disconnect()
                    continue
        finally:
            self._connected = False
            await self._safe_disconnect()

    def stop(self) -> None:
        self._running = False

    async def _handle_raw(self, raw: dict) -> None:
        bar = self._normalizer.normalize(raw)
        if bar is None:
            return  # missing/invalid bar → skip (do not compute spread)
        for alert in self._staleness.observe(bar):
            await self._bus.publish(Topic.ALERTS, alert)
        await self._bus.publish(Topic.MARKET_DATA, bar)
        self.bars_published += 1

    async def _safe_disconnect(self) -> None:
        try:
            await self._connector.disconnect()
        except Exception:  # pragma: no cover - best effort
            logger.debug("%s disconnect error", self._name, exc_info=True)
