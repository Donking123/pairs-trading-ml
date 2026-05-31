"""FX staleness monitoring (architecture §3.3).

Tracks the last date a fresh rate was received per currency and raises an
:class:`FXStaleEvent` when a currency's rate ages past the current bar:

* age ≥ 1 business day → stale (the prior day's rate is used as fallback with
  ``is_stale=True``; the H&S engine then skips spread computation for affected
  pairs);
* age > ``suspend_after_business_days`` (default 2) → ``suspended=True`` — the
  pair should be marked SUSPENDED and the operator alerted.

Depends only on the standard library and ``asian_adr.core``.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Literal

from asian_adr.core import Clock, FXRateEvent

from asian_adr.core import BaseEvent

logger = logging.getLogger(__name__)

__all__ = ["FXStaleEvent", "FXStalenessMonitor", "business_days_between"]


class FXStaleEvent(BaseEvent):
    """A currency's FX rate is older than the current bar."""

    event_type: Literal["fx_stale"] = "fx_stale"
    base_currency: str
    age_business_days: int
    suspended: bool


def business_days_between(d0: date, d1: date) -> int:
    """Number of weekdays strictly after ``d0`` up to and including ``d1``."""
    if d1 <= d0:
        return 0
    days = 0
    cur = d0
    while cur < d1:
        cur += timedelta(days=1)
        if cur.weekday() < 5:  # Mon-Fri
            days += 1
    return days


class FXStalenessMonitor:
    """Per-currency freshness tracker."""

    def __init__(self, clock: Clock, *, suspend_after_business_days: int = 2) -> None:
        self._clock = clock
        self._suspend_after = suspend_after_business_days
        self._last_date: dict[str, date] = {}

    def observe(self, event: FXRateEvent) -> None:
        """Record a freshly received (non-stale) rate."""
        if not event.is_stale:
            self._last_date[event.base_currency] = event.timestamp_exchange.date()

    def staleness_alerts(self, currency: str, today: date) -> list[BaseEvent]:
        """Return an :class:`FXStaleEvent` if ``currency`` is behind ``today``."""
        last = self._last_date.get(currency)
        if last is None:
            return []
        age = business_days_between(last, today)
        if age <= 0:
            return []
        suspended = age > self._suspend_after
        if suspended:
            logger.warning("FX %s stale %d business days → SUSPEND", currency, age)
        return [
            FXStaleEvent(
                timestamp_exchange=self._clock.now(),
                timestamp_received=self._clock.now(),
                base_currency=currency,
                age_business_days=age,
                suspended=suspended,
            )
        ]
