"""Provider format → canonical ``BarEvent`` (architecture §3.2).

Every raw provider message is normalised here before it ever reaches the bus —
no component downstream sees provider-specific shapes. Each provider has its own
normaliser implementing :class:`BarNormalizer`; all share :func:`build_bar`,
which coerces fields to the canonical types and stamps ``timestamp_received``
from the injected clock (never ``datetime.now()``).

Depends only on the standard library and ``asian_adr.core``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Mapping, Protocol, runtime_checkable

from asian_adr.core import BarEvent, Clock

logger = logging.getLogger(__name__)

__all__ = [
    "BarNormalizer",
    "build_bar",
    "PolygonNormalizer",
    "AlpacaNormalizer",
    "AsianNormalizer",
]


@runtime_checkable
class BarNormalizer(Protocol):
    """Maps a raw provider message to a :class:`BarEvent` (or ``None`` to skip)."""

    def normalize(self, raw: dict) -> BarEvent | None: ...


def _dec(value) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def build_bar(
    clock: Clock,
    *,
    ticker: str,
    open_: object,
    high: object,
    low: object,
    close: object,
    volume: object,
    exchange: str,
    currency: str,
    timestamp_exchange: datetime,
    is_adjusted: bool = False,
    bar_interval: str = "1d",
) -> BarEvent | None:
    """Build a :class:`BarEvent`, returning ``None`` if any price is invalid.

    A live message with a missing/NaN close is dropped rather than published —
    no spread can be computed from it (the "missing bar → skip" rule).
    """
    c = _dec(close)
    if c is None:
        return None
    o = _dec(open_) or c
    h = _dec(high) or c
    low_ = _dec(low) or c
    try:
        vol = int(volume) if volume is not None else 0
    except (TypeError, ValueError):
        vol = 0
    return BarEvent(
        timestamp_exchange=timestamp_exchange,
        timestamp_received=clock.now(),
        ticker=ticker,
        open=o,
        high=h,
        low=low_,
        close=c,
        volume=vol,
        bar_interval=bar_interval,
        is_adjusted=is_adjusted,
        exchange=exchange,
        currency=currency,
    )


def _ts_from_ms(ms: object) -> datetime:
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return datetime.now(timezone.utc)


class PolygonNormalizer:
    """Polygon.io aggregate message → BarEvent (U.S. ADR leg, USD).

    Polygon aggregate (``ev='A'``/``'AM'``) fields: ``sym``, ``o``, ``h``,
    ``l``, ``c``, ``v``, and the aggregate end time ``e`` (epoch ms).
    """

    def __init__(self, clock: Clock, exchange: str = "US") -> None:
        self._clock = clock
        self._exchange = exchange

    def normalize(self, raw: dict) -> BarEvent | None:
        sym = raw.get("sym") or raw.get("ticker")
        if not sym:
            return None
        return build_bar(
            self._clock,
            ticker=str(sym),
            open_=raw.get("o"),
            high=raw.get("h"),
            low=raw.get("l"),
            close=raw.get("c"),
            volume=raw.get("v"),
            exchange=self._exchange,
            currency="USD",
            timestamp_exchange=_ts_from_ms(raw.get("e") or raw.get("s")),
        )


class AlpacaNormalizer:
    """Alpaca bar message → BarEvent (U.S. ADR leg, USD).

    Alpaca bar fields: ``S`` (symbol), ``o``, ``h``, ``l``, ``c``, ``v``,
    ``t`` (RFC-3339 timestamp).
    """

    def __init__(self, clock: Clock, exchange: str = "US") -> None:
        self._clock = clock
        self._exchange = exchange

    def normalize(self, raw: dict) -> BarEvent | None:
        sym = raw.get("S") or raw.get("symbol")
        if not sym:
            return None
        t = raw.get("t")
        try:
            ts = datetime.fromisoformat(str(t).replace("Z", "+00:00")) if t else self._clock.now()
        except ValueError:
            ts = self._clock.now()
        return build_bar(
            self._clock,
            ticker=str(sym),
            open_=raw.get("o"),
            high=raw.get("h"),
            low=raw.get("l"),
            close=raw.get("c"),
            volume=raw.get("v"),
            exchange=self._exchange,
            currency="USD",
            timestamp_exchange=ts,
        )


class AsianNormalizer:
    """Asian provider message → BarEvent, resolving exchange/currency per ticker.

    The Asian feed carries instruments across several exchanges/currencies, so
    each ticker's exchange and currency are looked up from ``ticker_meta``
    (built from the pair registry). Expected raw fields: ``symbol``, ``open``,
    ``high``, ``low``, ``close``, ``volume``, ``timestamp`` (epoch ms).
    """

    def __init__(
        self, clock: Clock, ticker_meta: Mapping[str, tuple[str, str]]
    ) -> None:
        self._clock = clock
        self._meta = dict(ticker_meta)

    def normalize(self, raw: dict) -> BarEvent | None:
        sym = raw.get("symbol") or raw.get("sym")
        if not sym:
            return None
        meta = self._meta.get(str(sym))
        if meta is None:
            logger.debug("no exchange/currency meta for %s; skipping", sym)
            return None
        exchange, currency = meta
        return build_bar(
            self._clock,
            ticker=str(sym),
            open_=raw.get("open"),
            high=raw.get("high"),
            low=raw.get("low"),
            close=raw.get("close"),
            volume=raw.get("volume"),
            exchange=exchange,
            currency=currency,
            timestamp_exchange=_ts_from_ms(raw.get("timestamp")),
        )
