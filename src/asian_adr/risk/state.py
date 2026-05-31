"""Portfolio risk state aggregated from the position feed.

``RiskState`` consumes :class:`~asian_adr.core.PositionUpdateEvent` to maintain
the running figures the rules need — open-pair count, gross and per-country
notional, total PnL, peak-to-trough drawdown, day-loss, and the rolling
rate-of-loss window — and assembles them into an immutable
:class:`~asian_adr.risk.rules.base.RiskContext` for each signal.

Depends only on the standard library and ``asian_adr.core`` (+ the rule
framework for the context/config types).
"""

from __future__ import annotations

from collections import deque
from decimal import Decimal

from asian_adr.core import AsianADRApprovedPair, HongSusmelSignalEvent, PositionUpdateEvent

from .rules.base import RiskConfig, RiskContext
from .rules.country_concentration import country_of

__all__ = ["RiskState"]


class RiskState:
    """Mutable aggregate of portfolio risk metrics."""

    def __init__(
        self, config: RiskConfig, pairs: list[AsianADRApprovedPair] | None = None
    ) -> None:
        self._config = config
        self._pairs: dict[str, AsianADRApprovedPair] = {
            p.pair_id: p for p in (pairs or [])
        }
        # latest position per (pair_id, ticker)
        self._positions: dict[tuple[str, str], PositionUpdateEvent] = {}
        self._short_locate: dict[str, bool] = {}

        self._day_start_pnl = Decimal(0)
        self._peak_equity = config.aum_usd
        # rolling PnL snapshots for the rate-of-loss window
        self._pnl_window: deque[Decimal] = deque(maxlen=config.rate_of_loss_bars + 1)
        self._pnl_window.append(Decimal(0))

    # -- registry ------------------------------------------------------------

    def set_pairs(self, pairs: list[AsianADRApprovedPair]) -> None:
        self._pairs = {p.pair_id: p for p in pairs}

    def set_short_locate(self, adr_ticker: str, available: bool) -> None:
        self._short_locate[adr_ticker] = available

    # -- ingestion -----------------------------------------------------------

    def update_position(self, event: PositionUpdateEvent) -> None:
        self._positions[(event.pair_id, event.ticker)] = event
        equity = self._config.aum_usd + self.total_pnl_usd
        if equity > self._peak_equity:
            self._peak_equity = equity

    def record_bar(self) -> None:
        """Snapshot current PnL for the rate-of-loss rolling window."""
        self._pnl_window.append(self.total_pnl_usd)

    def start_new_day(self) -> None:
        """Reset the daily-loss baseline at a date rollover."""
        self._day_start_pnl = self.total_pnl_usd

    # -- aggregates ----------------------------------------------------------

    @property
    def total_pnl_usd(self) -> Decimal:
        return sum(
            (p.unrealized_pnl_usd + p.realized_pnl_usd for p in self._positions.values()),
            Decimal(0),
        )

    @property
    def open_pairs(self) -> int:
        return len(
            {
                pair_id
                for (pair_id, _), p in self._positions.items()
                if p.net_quantity != 0
            }
        )

    @property
    def gross_notional_usd(self) -> Decimal:
        return sum(
            (abs(p.notional_value_usd) for p in self._positions.values()), Decimal(0)
        )

    def country_notional_usd(self, country: str) -> Decimal:
        total = Decimal(0)
        for (pair_id, _), p in self._positions.items():
            pair = self._pairs.get(pair_id)
            if pair and country_of(pair.underlying_exchange) == country:
                total += abs(p.notional_value_usd)
        return total

    @property
    def daily_pnl_usd(self) -> Decimal:
        return self.total_pnl_usd - self._day_start_pnl

    @property
    def drawdown_pct(self) -> Decimal:
        equity = self._config.aum_usd + self.total_pnl_usd
        if self._config.aum_usd <= 0:
            return Decimal(0)
        return (self._peak_equity - equity) / self._config.aum_usd

    @property
    def loss_window_usd(self) -> Decimal:
        """Signed PnL change over the rolling window (negative = loss)."""
        if len(self._pnl_window) < 2:
            return Decimal(0)
        return self._pnl_window[-1] - self._pnl_window[0]

    def short_locate_available(self, adr_ticker: str) -> bool:
        return self._short_locate.get(adr_ticker, True)

    # -- context assembly ----------------------------------------------------

    def build_context(
        self,
        signal: HongSusmelSignalEvent | None,
        pair: AsianADRApprovedPair | None,
        zero_return_pct: Decimal = Decimal("0"),
        kill_switch_active: bool = False,
    ) -> RiskContext:
        country = country_of(pair.underlying_exchange) if pair else None
        return RiskContext(
            config=self._config,
            pair=pair,
            proposed_notional=self._config.entry_notional_usd,
            zero_return_pct=zero_return_pct,
            open_pairs=self.open_pairs,
            gross_notional_usd=self.gross_notional_usd,
            country=country,
            country_notional_usd=(
                self.country_notional_usd(country) if country else Decimal(0)
            ),
            daily_pnl_usd=self.daily_pnl_usd,
            drawdown_pct=self.drawdown_pct,
            loss_window_usd=self.loss_window_usd,
            short_locate_available=(
                self.short_locate_available(signal.adr_ticker) if signal else True
            ),
            kill_switch_active=kill_switch_active,
            days_held=signal.days_held if signal else 0,
        )
