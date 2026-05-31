"""Interactive Brokers short-locate verification (architecture §3.10).

Queries IB for shortable shares before every short SELL; the Risk Engine's
``ShortLocateRule`` and the gateway both gate on the result. ``ib_insync`` is
imported lazily, so this module is import-safe without it.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from ..base import LocateResult

logger = logging.getLogger(__name__)

__all__ = ["IBShortLocate"]


class IBShortLocate:
    """Short-locate provider backed by an IB connection.

    Parameters
    ----------
    ib:
        A connected ``ib_insync.IB`` instance (injected by the gateway). If
        ``None``, :meth:`verify` returns unavailable so shorts are blocked until
        a connection exists.
    fallback_available:
        Returned when the IB query cannot be completed (e.g. data permission
        gaps) — defaults to ``False`` (fail-safe: block the short).
    """

    def __init__(self, ib=None, *, fallback_available: bool = False) -> None:
        self._ib = ib
        self._fallback = fallback_available

    def set_ib(self, ib) -> None:
        self._ib = ib

    async def verify(self, ticker: str, quantity: Decimal) -> LocateResult:
        if self._ib is None:
            return LocateResult(available=False)
        try:  # pragma: no cover - requires live IB
            from ib_insync import Stock

            contract = Stock(ticker, "SMART", "USD")
            tickers = await self._ib.reqTickersAsync(contract)
            if not tickers:
                return LocateResult(available=self._fallback)
            shortable = getattr(tickers[0], "shortableShares", None)
            borrow = getattr(tickers[0], "feeRate", None)
            available = shortable is None or shortable >= float(quantity)
            return LocateResult(
                available=available,
                borrow_rate=Decimal(str(borrow)) if borrow is not None else Decimal("0"),
            )
        except Exception as exc:  # pragma: no cover - network
            logger.warning("IB locate query failed for %s: %s", ticker, exc)
            return LocateResult(available=self._fallback)
