"""OANDA v20 REST connector + FX handler (Phase 3).

Fetches daily FX close rates from OANDA's pricing endpoint. ``httpx`` is
imported lazily inside :meth:`fetch_rates`, so importing this module never
requires the dependency. HTTP failures surface as ``ConnectionError`` so the
handler falls back to cached rates for that cycle.

``OANDAFXHandler`` matches the §8.1 wiring name and maps each registry currency
to its OANDA instrument (majors like AUD quote USD; most Asian currencies are
quoted ``USD_XXX``).
"""

from __future__ import annotations

import logging
from typing import Iterable

from asian_adr.core import Clock
from asian_adr.event_bus import AbstractEventBus

from ..handler import FXRateHandler
from ..normalizer import OANDANormalizer

logger = logging.getLogger(__name__)

__all__ = ["OANDAConnector", "OANDAFXHandler", "oanda_instrument", "USD_QUOTE_CCYS"]

# Currencies conventionally quoted as XXX/USD (USD is the quote); everything
# else (most Asian currencies) is quoted USD/XXX.
USD_QUOTE_CCYS: frozenset[str] = frozenset({"AUD", "NZD", "EUR", "GBP"})

_HOSTS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}


def oanda_instrument(currency: str) -> str:
    """Map a currency to its OANDA instrument vs USD."""
    if currency in USD_QUOTE_CCYS:
        return f"{currency}_USD"
    return f"USD_{currency}"


class OANDAConnector:
    """OANDA v20 REST pricing connector."""

    def __init__(
        self,
        api_token: str,
        account_id: str,
        *,
        environment: str = "practice",
        timeout: float = 10.0,
    ) -> None:
        self._token = api_token
        self._account_id = account_id
        self._host = _HOSTS.get(environment, _HOSTS["practice"])
        self._timeout = timeout

    async def fetch_rates(self, instruments: Iterable[str]) -> list[dict]:
        instruments = list(instruments)
        if not instruments:
            return []
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "OANDAConnector requires 'httpx' (uv add httpx)."
            ) from exc
        url = f"{self._host}/v3/accounts/{self._account_id}/pricing"
        headers = {"Authorization": f"Bearer {self._token}"}
        params = {"instruments": ",".join(instruments)}
        try:  # pragma: no cover - network
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=headers, params=params)
                resp.raise_for_status()
                return resp.json().get("prices", [])
        except Exception as exc:  # pragma: no cover - network
            raise ConnectionError(f"oanda pricing fetch failed: {exc}") from exc


class OANDAFXHandler(FXRateHandler):
    """Daily FX rate handler backed by OANDA v20 REST."""

    def __init__(
        self,
        bus: AbstractEventBus,
        clock: Clock,
        currency_pairs: Iterable[str] = (),
        *,
        api_token: str = "",
        account_id: str = "",
        environment: str = "practice",
        **kwargs,
    ) -> None:
        super().__init__(
            bus,
            clock,
            OANDAConnector(api_token, account_id, environment=environment),
            OANDANormalizer(clock),
            currencies=currency_pairs,
            instrument_fn=oanda_instrument,
            name="oanda",
            **kwargs,
        )
