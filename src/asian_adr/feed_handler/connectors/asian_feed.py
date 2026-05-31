"""Asian underlying connector + feed handler (Phase 3).

The Asian feed spans several exchanges/currencies, so the normaliser resolves
each ticker's exchange and currency from a ``ticker_meta`` map built from the
pair registry. The concrete provider is TBD per exchange, so :class:`AsianConnector`
is a generic configurable WebSocket connector (``websockets`` imported lazily);
swap in an exchange-specific subclass when a provider is chosen.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Iterable, Mapping

from asian_adr.core import AsianADRApprovedPair, Clock
from asian_adr.event_bus import AbstractEventBus

from ..connector import FeedHandler
from ..normalizer import AsianNormalizer
from ..subscription_manager import SubscriptionManager

logger = logging.getLogger(__name__)

__all__ = ["AsianConnector", "AsianFeedHandler", "ticker_meta_from_pairs"]


def ticker_meta_from_pairs(
    pairs: Iterable[AsianADRApprovedPair],
) -> dict[str, tuple[str, str]]:
    """Build ``{underlying_ticker: (exchange, currency)}`` from the registry."""
    return {
        p.underlying_ticker: (p.underlying_exchange, p.underlying_currency)
        for p in pairs
    }


class AsianConnector:
    """Generic WebSocket connector for an Asian market-data provider.

    ``subscribe_template`` formats the subscription payload; ``auth_payload`` is
    sent on connect if provided. Adapt per provider once one is selected.
    """

    def __init__(
        self,
        url: str,
        *,
        auth_payload: dict | None = None,
    ) -> None:
        self._url = url
        self._auth = auth_payload
        self._ws = None

    async def connect(self) -> None:
        try:
            import json

            import websockets
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "AsianConnector requires 'websockets' (uv add websockets)."
            ) from exc
        try:
            self._ws = await websockets.connect(self._url)
            if self._auth is not None:
                await self._ws.send(json.dumps(self._auth))
        except Exception as exc:  # pragma: no cover - network
            raise ConnectionError(f"asian feed connect failed: {exc}") from exc

    async def subscribe(self, tickers: Iterable[str]) -> None:
        import json

        tickers = list(tickers)
        if self._ws is None or not tickers:
            return
        await self._ws.send(json.dumps({"action": "subscribe", "symbols": tickers}))

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
            raise ConnectionError(f"asian feed stream error: {exc}") from exc

    async def disconnect(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


class AsianFeedHandler(FeedHandler):
    """Asian underlying feed handler.

    Provide ``pairs`` (the registry) so the handler can build both the
    subscription list and the per-ticker exchange/currency metadata the
    normaliser needs; or pass ``ticker_meta`` and ``tickers`` explicitly.
    """

    def __init__(
        self,
        bus: AbstractEventBus,
        clock: Clock,
        *,
        pairs: Iterable[AsianADRApprovedPair] | None = None,
        ticker_meta: Mapping[str, tuple[str, str]] | None = None,
        tickers: Iterable[str] | None = None,
        connector: AsianConnector | None = None,
        url: str = "",
        **kwargs,
    ) -> None:
        if pairs is not None:
            pairs = list(pairs)
            meta = ticker_meta_from_pairs(pairs)
            tick = {p.underlying_ticker for p in pairs}
        else:
            meta = dict(ticker_meta or {})
            tick = set(tickers or meta.keys())
        super().__init__(
            bus,
            clock,
            connector or AsianConnector(url),
            AsianNormalizer(clock, meta),
            SubscriptionManager("underlying", tick),
            name="asian",
            **kwargs,
        )
