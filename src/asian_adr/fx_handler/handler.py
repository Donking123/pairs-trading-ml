"""FX connector interface and the polling handler/orchestrator.

(The fx_handler spec tree keeps connection lifecycle in ``connectors/oanda.py``;
this module holds the provider-agnostic pieces — the connector protocol, an
in-memory test connector, and the generic :class:`FXRateHandler` orchestration —
the same role ``feed_handler/connector.py`` plays for the market-data feed.)

:class:`FXRateHandler` is a ``Producer`` (``async run()``): each cycle it fetches
the day's rates, normalises and publishes them to ``fx-rates``, updates the
cache, and for any currency that did **not** report it republishes the prior
rate with ``is_stale=True`` (the fallback rule) and emits staleness alerts.

Imports only ``core`` and ``event_bus``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Iterable, Protocol, runtime_checkable

from asian_adr.core import Clock
from asian_adr.event_bus import AbstractEventBus, Topic

from .normalizer import FXNormalizer
from .rate_cache import FXRateCache
from .staleness_monitor import FXStalenessMonitor

logger = logging.getLogger(__name__)

__all__ = ["AbstractFXConnector", "IterableFXConnector", "FXRateHandler"]


@runtime_checkable
class AbstractFXConnector(Protocol):
    """Fetches the latest provider ticks for a set of instruments (REST)."""

    async def fetch_rates(self, instruments: Iterable[str]) -> list[dict]: ...


class IterableFXConnector:
    """In-memory connector returning canned ticks each fetch (tests / replay).

    ``ticks`` may be a fixed list, or a callable mapping the requested
    instruments to a list of ticks (to simulate a missing currency).
    """

    def __init__(self, ticks) -> None:
        self._ticks = ticks
        self.fetches = 0

    async def fetch_rates(self, instruments: Iterable[str]) -> list[dict]:
        self.fetches += 1
        if callable(self._ticks):
            return list(self._ticks(list(instruments)))
        return list(self._ticks)


class FXRateHandler:
    """Polls a connector for daily FX rates and publishes ``FXRateEvent``s."""

    def __init__(
        self,
        bus: AbstractEventBus,
        clock: Clock,
        connector: AbstractFXConnector,
        normalizer: FXNormalizer,
        *,
        currencies: Iterable[str],
        instrument_fn: Callable[[str], str] | None = None,
        cache: FXRateCache | None = None,
        staleness: FXStalenessMonitor | None = None,
        poll_interval: float = 86_400.0,
        max_cycles: int | None = None,
        name: str = "fx",
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._connector = connector
        self._normalizer = normalizer
        self._instrument_fn = instrument_fn or (lambda c: c)
        self._currencies = {c for c in currencies if c != "USD"}
        self.cache = cache or FXRateCache()
        self._staleness = staleness or FXStalenessMonitor(clock)
        self._poll_interval = poll_interval
        self._max_cycles = max_cycles
        self._name = name
        self._running = False

    @property
    def _instruments(self) -> list[str]:
        return [self._instrument_fn(c) for c in sorted(self._currencies)]

    async def on_registry_update(self, event: object) -> None:
        pairs = getattr(event, "registry", event)
        self._currencies = {
            p.underlying_currency for p in pairs if p.underlying_currency != "USD"
        }

    # -- run loop ------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        cycles = 0
        while self._running:
            await self.poll_once()
            cycles += 1
            if self._max_cycles is not None and cycles >= self._max_cycles:
                break
            await asyncio.sleep(self._poll_interval)

    def stop(self) -> None:
        self._running = False

    async def poll_once(self) -> None:
        """Fetch, normalise, publish, and run the stale-fallback for one cycle."""
        try:
            ticks = await self._connector.fetch_rates(self._instruments)
        except ConnectionError as exc:
            logger.warning("%s fetch failed: %s — using cached fallbacks", self._name, exc)
            ticks = []

        received: set[str] = set()
        for raw in ticks:
            event = self._normalizer.normalize(raw)
            if event is None:
                continue
            self.cache.update(event)
            self._staleness.observe(event)
            received.add(event.base_currency)
            await self._bus.publish(Topic.FX_RATES, event)

        today = self._clock.today()
        for ccy in sorted(self._currencies):
            if ccy not in received:
                prior = self.cache.get_event(ccy)
                if prior is not None:
                    stale = prior.model_copy(
                        update={"is_stale": True, "timestamp_received": self._clock.now()}
                    )
                    self.cache.update(stale)
                    await self._bus.publish(Topic.FX_RATES, stale)
            for alert in self._staleness.staleness_alerts(ccy, today):
                await self._bus.publish(Topic.ALERTS, alert)
