"""Per-pair mutable state for the Hong & Susmel engine.

Two pieces of mutable state live here (events stay immutable; only engines
mutate):

* :class:`WelfordRollingStats` — rolling µ/σ over the estimation window ``T``,
  computed in ``Decimal`` to keep spread arithmetic exact. ``ddof=1`` matches
  the sample standard deviation used throughout the paper.
* :class:`HSPairState` — the prices, freshness dates, position flag, holding
  clock, and rolling ADR zero-return tracker for one pair.

Depends only on the standard library and ``asian_adr.core``.
"""

from __future__ import annotations

from collections import deque
from datetime import date
from decimal import Decimal

from asian_adr.core import AsianADRApprovedPair, HSPosition

__all__ = ["WelfordRollingStats", "HSPairState"]


class WelfordRollingStats:
    """Rolling mean / standard deviation over a fixed window of ``window`` samples.

    Maintained incrementally with a running sum and sum-of-squares over a
    bounded ``deque`` (O(1) per update). ``Decimal`` arithmetic at the default
    28-digit precision keeps the variance free of the catastrophic cancellation
    that float sum-of-squares would suffer; tiny negative residuals are clamped
    to zero before the square root.

    The engine treats the window as warm once ``count`` reaches ``window``
    (``estimation_days``), so no signal is emitted until a full window of
    spreads has been observed.

    Parameters
    ----------
    window:
        Number of most-recent samples retained (``T``).
    ddof:
        Delta degrees of freedom for the variance denominator (``1`` → sample).
    """

    def __init__(self, window: int, ddof: int = 1) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = window
        self.ddof = ddof
        self._values: deque[Decimal] = deque()
        self._sum = Decimal(0)
        self._sumsq = Decimal(0)

    def update(self, x: Decimal) -> None:
        """Add a new sample, evicting the oldest when the window is full."""
        x = Decimal(x)
        self._values.append(x)
        self._sum += x
        self._sumsq += x * x
        if len(self._values) > self.window:
            old = self._values.popleft()
            self._sum -= old
            self._sumsq -= old * old

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def is_full(self) -> bool:
        return len(self._values) >= self.window

    @property
    def mean(self) -> Decimal:
        n = len(self._values)
        if n == 0:
            return Decimal(0)
        return self._sum / n

    @property
    def variance(self) -> Decimal:
        n = len(self._values)
        if n - self.ddof <= 0:
            return Decimal(0)
        mean = self._sum / n
        var = (self._sumsq - n * mean * mean) / (n - self.ddof)
        return var if var > 0 else Decimal(0)  # clamp float-free residuals

    @property
    def std(self) -> Decimal:
        return self.variance.sqrt()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"WelfordRollingStats(count={self.count}, window={self.window})"


class HSPairState:
    """Mutable per-pair state machine inputs for one approved pair.

    Tracks the latest close and bar date of each leg (for the stale-leg guard),
    the rolling spread statistics, the logical position and its entry date (for
    holding-period arithmetic), and a rolling window of ADR zero-return flags
    that the :class:`ZeroReturnDayFilter` risk rule consumes.
    """

    def __init__(self, pair: AsianADRApprovedPair) -> None:
        self.pair = pair
        self.adr_ticker = pair.adr_ticker
        self.underlying_ticker = pair.underlying_ticker

        self.adr_price: Decimal | None = None
        self.underlying_price: Decimal | None = None
        self.adr_date: date | None = None
        self.underlying_date: date | None = None

        self.rolling_stats = WelfordRollingStats(pair.estimation_days, ddof=1)
        self.position = HSPosition.FLAT
        self.entry_date: date | None = None

        self._prev_adr_price: Decimal | None = None
        self._zero_returns: deque[int] = deque()

    # -- price ingestion -----------------------------------------------------

    def update_price(self, ticker: str, close: Decimal, bar_date: date) -> None:
        """Record a new close for whichever leg ``ticker`` identifies.

        For the ADR leg this also updates the rolling zero-return-day window:
        a close identical to the prior ADR close counts as a zero-return day
        (the Bekaert et al. illiquidity proxy).
        """
        if ticker == self.adr_ticker:
            if self._prev_adr_price is not None:
                self._zero_returns.append(1 if close == self._prev_adr_price else 0)
                if len(self._zero_returns) > self.pair.estimation_days:
                    self._zero_returns.popleft()
            self._prev_adr_price = close
            self.adr_price = close
            self.adr_date = bar_date
        elif ticker == self.underlying_ticker:
            self.underlying_price = close
            self.underlying_date = bar_date

    def both_legs_fresh(self) -> bool:
        """Stale-leg guard: True only when both legs carry the same bar date."""
        return self.adr_date is not None and self.adr_date == self.underlying_date

    def rolling_zero_return_pct(self) -> Decimal:
        """Fraction of zero-return ADR days over the current window."""
        if not self._zero_returns:
            return Decimal(0)
        return Decimal(sum(self._zero_returns)) / Decimal(len(self._zero_returns))

    # -- position transitions ------------------------------------------------

    def open_position(self, entry_date: date) -> None:
        self.position = HSPosition.OPEN
        self.entry_date = entry_date

    def close_position(self) -> None:
        self.position = HSPosition.FLAT
        self.entry_date = None

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"HSPairState(pair_id={self.pair.pair_id!r}, position={self.position.value}, "
            f"count={self.rolling_stats.count})"
        )
