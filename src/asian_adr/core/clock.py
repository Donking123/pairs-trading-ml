"""Clock abstractions for live and backtest execution.

Every component reads time through a :class:`Clock` rather than calling
``datetime.now`` directly. This keeps the Hong & Susmel engine, risk engine,
and position engine **identical** across live and backtest modes (the
research/production-parity principle): only the concrete clock differs.

* :class:`LiveClock` — wall-clock UTC, used in live / paper trading.
* :class:`SimulatedClock` — advanced explicitly by the backtest replay loop as
  events stream off the merged min-heap. It exposes ``date_changed`` so the
  engine can reload the point-in-time pair registry exactly on day rollover.

Only the standard library is used here.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "LiveClock", "SimulatedClock"]


@runtime_checkable
class Clock(Protocol):
    """Read-only time source shared by every component.

    Implementations must return timezone-aware UTC datetimes from
    :meth:`now`. ``today`` is the UTC calendar date used for holding-period
    arithmetic.
    """

    def now(self) -> datetime:
        """Current instant as a timezone-aware UTC ``datetime``."""
        ...

    def today(self) -> date:
        """Current UTC calendar date."""
        ...


class LiveClock:
    """Wall-clock implementation backed by the system clock (UTC)."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def today(self) -> date:
        return self.now().date()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "LiveClock()"


class SimulatedClock:
    """Deterministic clock driven by the backtest replay loop.

    The replay engine calls :meth:`advance_to` with each event's exchange
    timestamp. The clock never moves backwards; attempting to do so raises
    ``ValueError`` to catch out-of-order replay streams early. After each
    advance, :attr:`date_changed` reports whether the calendar date rolled
    over, which the engine uses to trigger a point-in-time registry reload.

    Parameters
    ----------
    start:
        Initial instant. A naive ``datetime`` or a ``date`` is interpreted as
        midnight UTC.
    """

    def __init__(self, start: datetime | date) -> None:
        self._current = self._coerce(start)
        self._date_changed = False

    # -- construction helpers ------------------------------------------------

    @staticmethod
    def _coerce(value: datetime | date) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        # bare ``date`` → midnight UTC
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)

    # -- Clock protocol ------------------------------------------------------

    def now(self) -> datetime:
        return self._current

    def today(self) -> date:
        return self._current.date()

    # -- backtest control ----------------------------------------------------

    def advance_to(self, timestamp: datetime | date) -> None:
        """Move the clock forward to ``timestamp``.

        Sets :attr:`date_changed` to ``True`` iff the UTC calendar date is
        different from the date before the advance. Raises ``ValueError`` if
        ``timestamp`` precedes the current time (out-of-order replay).
        """
        target = self._coerce(timestamp)
        if target < self._current:
            raise ValueError(
                f"SimulatedClock cannot move backwards: "
                f"{target.isoformat()} < {self._current.isoformat()}"
            )
        self._date_changed = target.date() != self._current.date()
        self._current = target

    @property
    def date_changed(self) -> bool:
        """Whether the most recent :meth:`advance_to` crossed a date boundary."""
        return self._date_changed

    @property
    def current_date(self) -> date:
        """Alias for :meth:`today` as a property, for ergonomic call sites."""
        return self._current.date()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"SimulatedClock(current={self._current.isoformat()})"
