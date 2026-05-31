"""In-memory FX rate cache (architecture §3.3).

A small write-only-from-the-handler cache: the handler calls :meth:`update`;
all consumers read via :meth:`get_usd_rate`, which returns **USD per 1 unit of
the base currency** (or ``None`` if not cached). ``USD`` resolves to ``1``.

Depends only on the standard library and ``asian_adr.core``.
"""

from __future__ import annotations

from decimal import Decimal

from asian_adr.core import FXRateEvent

__all__ = ["FXRateCache"]


class FXRateCache:
    """Latest :class:`FXRateEvent` per base currency."""

    def __init__(self) -> None:
        self._by_base: dict[str, FXRateEvent] = {}

    def update(self, event: FXRateEvent) -> None:
        self._by_base[event.base_currency] = event

    def get_usd_rate(self, base_currency: str) -> Decimal | None:
        """USD per 1 unit of ``base_currency``; ``None`` if not cached."""
        if base_currency == "USD":
            return Decimal(1)
        event = self._by_base.get(base_currency)
        return event.mid if event is not None else None

    def get_event(self, base_currency: str) -> FXRateEvent | None:
        return self._by_base.get(base_currency)

    @property
    def currencies(self) -> frozenset[str]:
        return frozenset(self._by_base)

    def __contains__(self, base_currency: str) -> bool:
        return base_currency in self._by_base
