"""Polygon.io connector + feed handler for the U.S. ADR leg (Phase 3).

``websockets`` is imported lazily inside :meth:`PolygonConnector.connect`, so
importing this module never requires the dependency — only starting a live feed
does. Connection / stream failures surface as ``ConnectionError`` so the
:class:`~asian_adr.feed_handler.connector.FeedHandler` reconnect loop handles
them.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Iterable

from asian_adr.core import Clock
from asian_adr.event_bus import AbstractEventBus

from ..connector import FeedHandler
from ..normalizer import PolygonNormalizer
from ..subscription_manager import SubscriptionManager

logger = logging.getLogger(__name__)

__all__ = ["PolygonConnector", "PolygonFeedHandler"]


class PolygonConnector:
    """WebSocket connection to Polygon.io equities aggregates."""

    URL = "wss://socket.polygon.io/stocks"

    def __init__(self, api_key: str, *, channel_prefix: str = "A") -> None:
        self._api_key = api_key
        self._prefix = channel_prefix  # "A" per-second, "AM" per-minute aggregates
        self._ws = None

    async def connect(self) -> None:
        try:
            import json

            import websockets
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "PolygonConnector requires 'websockets' (uv add websockets)."
            ) from exc
        try:
            self._ws = await websockets.connect(self.URL)
            await self._ws.send(json.dumps({"action": "auth", "params": self._api_key}))
        except Exception as exc:  # pragma: no cover - network
            raise ConnectionError(f"polygon connect failed: {exc}") from exc

    async def subscribe(self, tickers: Iterable[str]) -> None:
        import json

        tickers = list(tickers)
        if self._ws is None or not tickers:
            return
        params = ",".join(f"{self._prefix}.{t}" for t in tickers)
        await self._ws.send(json.dumps({"action": "subscribe", "params": params}))

    async def messages(self) -> AsyncIterator[dict]:  # pragma: no cover - network
        import json

        if self._ws is None:
            return
        try:
            async for raw in self._ws:
                payload = json.loads(raw)
                for event in payload if isinstance(payload, list) else [payload]:
                    yield event
        except Exception as exc:
            raise ConnectionError(f"polygon stream error: {exc}") from exc

    async def disconnect(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


class PolygonFeedHandler(FeedHandler):
    """U.S. ADR feed handler backed by Polygon.io."""

    def __init__(
        self,
        bus: AbstractEventBus,
        clock: Clock,
        tickers: Iterable[str] = (),
        *,
        api_key: str = "",
        exchange: str = "US",
        channel_prefix: str = "A",
        **kwargs,
    ) -> None:
        super().__init__(
            bus,
            clock,
            PolygonConnector(api_key, channel_prefix=channel_prefix),
            PolygonNormalizer(clock, exchange=exchange),
            SubscriptionManager("adr", tickers),
            name="polygon",
            **kwargs,
        )
