"""Provider FX ticks → canonical ``FXRateEvent`` (architecture §3.3).

FX rates are always normalised to **USD per 1 unit of the base currency**,
inverting the quote when the provider quotes USD as the base (e.g. ``USD_JPY``
gives JPY per USD, so ``mid = 1 / price``). Cross pairs with no USD leg are
skipped.

Depends only on the standard library and ``asian_adr.core``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Protocol, runtime_checkable

from asian_adr.core import Clock, FXRateEvent

logger = logging.getLogger(__name__)

__all__ = ["FXNormalizer", "OANDANormalizer"]


@runtime_checkable
class FXNormalizer(Protocol):
    def normalize(self, raw: dict) -> FXRateEvent | None: ...


def _dec(value) -> Decimal | None:
    if value is None:
        return None
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return d if d > 0 else None


def _mid_price(raw: dict) -> Decimal | None:
    """Extract a mid price from a candle (`mid.c`) or pricing (`bids`/`asks`) tick."""
    if "mid" in raw:
        mid = raw["mid"]
        return _dec(mid.get("c") if isinstance(mid, dict) else mid)
    bid = raw.get("bids")
    ask = raw.get("asks")
    if bid and ask:
        b = _dec(bid[0].get("price"))
        a = _dec(ask[0].get("price"))
        if b is not None and a is not None:
            return (b + a) / 2
    return _dec(raw.get("price") or raw.get("c"))


class OANDANormalizer:
    """OANDA v20 pricing/candle tick → FXRateEvent (USD per 1 unit of base)."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    def normalize(self, raw: dict) -> FXRateEvent | None:
        instrument = raw.get("instrument") or raw.get("symbol")
        if not instrument or "_" not in str(instrument):
            return None
        base, quote = str(instrument).split("_", 1)
        price = _mid_price(raw)
        if price is None:
            return None

        if quote == "USD":
            base_ccy, mid_usd = base, price            # already USD per 1 base
        elif base == "USD":
            base_ccy, mid_usd = quote, Decimal(1) / price  # invert USD_XXX
        else:
            logger.debug("cross pair %s has no USD leg; skipping", instrument)
            return None

        ts = self._parse_time(raw.get("time"))
        return FXRateEvent(
            timestamp_exchange=ts,
            timestamp_received=self._clock.now(),
            currency_pair=f"{base_ccy}_USD",
            base_currency=base_ccy,
            quote_currency="USD",
            mid=mid_usd,
            provider="oanda",
        )

    def _parse_time(self, t) -> datetime:
        if not t:
            return self._clock.now()
        try:
            return datetime.fromisoformat(str(t).replace("Z", "+00:00"))
        except ValueError:
            return self._clock.now()
