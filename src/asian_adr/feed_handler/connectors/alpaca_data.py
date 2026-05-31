"""Alpaca connector + feed handler — alternative U.S. ADR feed (Phase 3).

Commission-free / paper-trading-friendly alternative to Polygon. ``websockets``
is imported lazily inside :meth:`connect`. Stream failures raise
``ConnectionError`` for the reconnect loop.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Iterable

from asian_adr.core import Clock
from asian_adr.event_bus import AbstractEventBus

from ..connector import FeedHandler
from ..normalizer import AlpacaNormalizer
from ..subscription_manager import SubscriptionManager

logger = logging.getLogger(__name__)

__all__ = ["AlpacaConnector", "AlpacaFeedHandler"]


class AlpacaConnector:
    """WebSocket connection to the Alpaca market-data stream."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        feed: str = "iex",  # "iex" (free) or "sip" (paid)
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._url = f"wss://stream.data.alpaca.markets/v2/{feed}"
        self._ws = None

    async def connect(self) -> None:
        try:
            import json

            import websockets
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "AlpacaConnector requires 'websockets' (uv add websockets)."
            ) from exc
        try:
            self._ws = await websockets.connect(self._url)
            await self._ws.send(
                json.dumps(
                    {"action": "auth", "key": self._api_key, "secret": self._api_secret}
                )
            )
        except Exception as exc:  # pragma: no cover - network
            raise ConnectionError(f"alpaca connect failed: {exc}") from exc

    async def subscribe(self, tickers: Iterable[str]) -> None:
        import json

        tickers = list(tickers)
        if self._ws is None or not tickers:
            return
        await self._ws.send(json.dumps({"action": "subscribe", "bars": tickers}))

    async def messages(self) -> AsyncIterator[dict]:  # pragma: no cover - network
        import json

        if self._ws is None:
            return
        try:
            async for raw in self._ws:
                payload = json.loads(raw)
                for event in payload if isinstance(payload, list) else [payload]:
                    if event.get("T") == "b":  # bar message
                        yield event
        except Exception as exc:
            raise ConnectionError(f"alpaca stream error: {exc}") from exc

    async def disconnect(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


class AlpacaFeedHandler(FeedHandler):
    """U.S. ADR feed handler backed by Alpaca."""

    def __init__(
        self,
        bus: AbstractEventBus,
        clock: Clock,
        tickers: Iterable[str] = (),
        *,
        api_key: str = "",
        api_secret: str = "",
        feed: str = "iex",
        exchange: str = "US",
        **kwargs,
    ) -> None:
        super().__init__(
            bus,
            clock,
            AlpacaConnector(api_key, api_secret, feed=feed),
            AlpacaNormalizer(clock, exchange=exchange),
            SubscriptionManager("adr", tickers),
            name="alpaca",
            **kwargs,
        )
