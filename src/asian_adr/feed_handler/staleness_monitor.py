"""Staleness / zero-return-day detection (architecture §3.2, §10.3).

As bars arrive the monitor watches for two conditions and emits the matching
alert events:

* **Zero-return day** — a close identical to the prior close for a ticker
  (the Bekaert et al. illiquidity proxy). Emits :class:`ZeroReturnEvent`, the
  same signal the ``ZeroReturnDayFilter`` risk rule consumes, and keeps a
  rolling zero-return-day percentage per ticker.
* **Feed gap** — a date jump greater than ``max_gap_bars`` since the ticker's
  last bar. Emits :class:`FeedGapEvent` so affected pairs can be marked
  data-unavailable.

Alerts go to the ``alerts`` topic; the handler publishes whatever the monitor
returns. Depends only on the standard library and ``asian_adr.core``.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import date
from decimal import Decimal
from typing import Literal

from asian_adr.core import BarEvent, BaseEvent, Clock

logger = logging.getLogger(__name__)

__all__ = ["ZeroReturnEvent", "FeedGapEvent", "StalenessMonitor"]


class ZeroReturnEvent(BaseEvent):
    """A ticker printed the same close as the prior day (zero-return day)."""

    event_type: Literal["zero_return"] = "zero_return"
    ticker: str
    exchange: str
    close: Decimal
    consecutive_zero_days: int
    zero_return_pct: Decimal


class FeedGapEvent(BaseEvent):
    """A ticker's bar stream skipped more than the allowed number of bars."""

    event_type: Literal["feed_gap"] = "feed_gap"
    ticker: str
    exchange: str
    gap_days: int


class _TickerState:
    __slots__ = ("last_close", "last_date", "consecutive", "window")

    def __init__(self, window_size: int) -> None:
        self.last_close: Decimal | None = None
        self.last_date: date | None = None
        self.consecutive = 0
        self.window: deque[int] = deque(maxlen=window_size)


class StalenessMonitor:
    """Tracks per-ticker freshness and emits zero-return / gap alerts."""

    def __init__(
        self, clock: Clock, *, window: int = 60, max_gap_bars: int = 3
    ) -> None:
        self._clock = clock
        self._window = window
        self._max_gap_bars = max_gap_bars
        self._states: dict[str, _TickerState] = {}

    def observe(self, bar: BarEvent) -> list[BaseEvent]:
        """Update state for ``bar`` and return any alert events to publish."""
        state = self._states.setdefault(bar.ticker, _TickerState(self._window))
        alerts: list[BaseEvent] = []
        bar_date = bar.timestamp_exchange.date()

        # -- feed gap --------------------------------------------------------
        if state.last_date is not None:
            gap = (bar_date - state.last_date).days
            if gap > self._max_gap_bars:
                alerts.append(
                    FeedGapEvent(
                        timestamp_exchange=bar.timestamp_exchange,
                        timestamp_received=self._clock.now(),
                        ticker=bar.ticker,
                        exchange=bar.exchange,
                        gap_days=gap,
                    )
                )

        # -- zero-return day -------------------------------------------------
        is_zero = state.last_close is not None and bar.close == state.last_close
        state.window.append(1 if is_zero else 0)
        if is_zero:
            state.consecutive += 1
            alerts.append(
                ZeroReturnEvent(
                    timestamp_exchange=bar.timestamp_exchange,
                    timestamp_received=self._clock.now(),
                    ticker=bar.ticker,
                    exchange=bar.exchange,
                    close=bar.close,
                    consecutive_zero_days=state.consecutive,
                    zero_return_pct=self.zero_return_pct(bar.ticker),
                )
            )
        else:
            state.consecutive = 0

        state.last_close = bar.close
        state.last_date = bar_date
        return alerts

    def zero_return_pct(self, ticker: str) -> Decimal:
        """Rolling fraction of zero-return days for ``ticker``."""
        state = self._states.get(ticker)
        if state is None or not state.window:
            return Decimal(0)
        return Decimal(sum(state.window)) / Decimal(len(state.window))
