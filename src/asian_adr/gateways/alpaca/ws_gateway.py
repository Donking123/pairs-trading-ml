"""Alpaca trade-updates WebSocket — streams fills (architecture §3.10).

A ``Producer`` that connects to the Alpaca trading-events stream and forwards
each trade update to an :class:`AlpacaRestGateway`, which correlates it to the
originating order and publishes the :class:`FillEvent`. ``websockets`` is
imported lazily; the stream paths are Phase-5 live code (``no cover``).
"""

from __future__ import annotations

import logging

from .rest_gateway import AlpacaRestGateway
from ..base import reconnect_with_backoff

logger = logging.getLogger(__name__)

__all__ = ["AlpacaWsGateway"]

_STREAMS = {
    "paper": "wss://paper-api.alpaca.markets/stream",
    "live": "wss://api.alpaca.markets/stream",
}


class AlpacaWsGateway:
    """Streams Alpaca trade updates and routes fills to the REST gateway."""

    def __init__(
        self,
        rest_gateway: AlpacaRestGateway,
        *,
        api_key: str = "",
        api_secret: str = "",
        environment: str = "paper",
    ) -> None:
        self._rest = rest_gateway
        self._key = api_key
        self._secret = api_secret
        self._url = _STREAMS.get(environment, _STREAMS["paper"])
        self._ws = None
        self._running = False

    async def _connect(self) -> None:  # pragma: no cover - network
        try:
            import json

            import websockets
        except ImportError as exc:
            raise RuntimeError("AlpacaWsGateway requires 'websockets' (uv add websockets).") from exc
        try:
            self._ws = await websockets.connect(self._url)
            await self._ws.send(
                json.dumps(
                    {"action": "authenticate",
                     "data": {"key_id": self._key, "secret_key": self._secret}}
                )
            )
            await self._ws.send(
                json.dumps({"action": "listen", "data": {"streams": ["trade_updates"]}})
            )
        except Exception as exc:
            raise ConnectionError(f"alpaca ws connect failed: {exc}") from exc

    async def run(self) -> None:  # pragma: no cover - network
        import json

        self._running = True
        while self._running:
            try:
                await reconnect_with_backoff(self._connect)
                async for raw in self._ws:
                    msg = json.loads(raw)
                    if msg.get("stream") == "trade_updates":
                        await self._rest.handle_trade_update(msg.get("data", {}))
            except ConnectionError as exc:
                logger.warning("alpaca ws disconnected, reconnecting: %s", exc)
                continue

    def stop(self) -> None:
        self._running = False
