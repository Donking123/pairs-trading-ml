"""Hong & Susmel signal-generation engine (architecture §3.5).

The engine is a collection of per-pair state machines. On every daily bar it:

1. updates the relevant leg's price and bar date,
2. enforces the **stale-leg guard** (both legs must share a bar date),
3. converts the local close to USD via the cached FX rate,
4. reconstructs the dollar spread ``P_ADR − (P_local × FX) / ratio`` —
   ``ratio`` is the structural ``β``, **never OLS-estimated** (rule ADR-002),
5. updates the rolling ``µ/σ`` and, once the window is warm, emits at most one
   signal per pair: ``SHORT_ADR`` on entry, ``EXIT`` on convergence, or
   ``FORCE_CLOSE`` at the holding-period limit.

Direction is permanently one-sided (``SHORT_ADR`` only) — Asian short-selling
restrictions eliminate the symmetric trade.

This engine is **identical** in backtest and live; only the bus, FX source, and
clock differ. It imports only from ``core`` and ``event_bus`` — never from
``risk``, ``position``, or ``gateways`` (layering rule).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from decimal import Decimal
from typing import Iterable

from asian_adr.core import (
    AsianADRApprovedPair,
    BarEvent,
    Clock,
    FXRateEvent,
    HongSusmelSignalEvent,
    HSPosition,
    HSSignal,
)
from asian_adr.event_bus import AbstractEventBus, Topic

from .signal_factory import SignalFactory
from .state import HSPairState

logger = logging.getLogger(__name__)

__all__ = ["HongSusmelEngine"]


class HongSusmelEngine:
    """Per-pair spread monitor and signal emitter.

    Parameters
    ----------
    bus:
        Event bus; signals are published to :attr:`Topic.SIGNALS`.
    clock:
        Time source (live or simulated).
    pairs:
        Initial active pair registry.
    """

    def __init__(
        self,
        bus: AbstractEventBus,
        clock: Clock,
        pairs: Iterable[AsianADRApprovedPair],
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._factory = SignalFactory(clock)
        self._states: dict[str, HSPairState] = {}
        self._ticker_index: dict[str, list[AsianADRApprovedPair]] = defaultdict(list)
        self._fx_rates: dict[str, FXRateEvent] = {}
        self._load_registry(pairs)

    # -- registry ------------------------------------------------------------

    def _load_registry(self, pairs: Iterable[AsianADRApprovedPair]) -> None:
        """(Re)build the ticker index and per-pair states.

        Existing state for a still-active pair is preserved so the rolling
        window and open position survive a registry refresh.
        """
        new_states: dict[str, HSPairState] = {}
        index: dict[str, list[AsianADRApprovedPair]] = defaultdict(list)
        for pair in pairs:
            if not pair.is_active:
                continue
            new_states[pair.pair_id] = self._states.get(pair.pair_id) or HSPairState(pair)
            index[pair.adr_ticker].append(pair)
            index[pair.underlying_ticker].append(pair)
        self._states = new_states
        self._ticker_index = index

    def _pairs_for_ticker(self, ticker: str) -> list[AsianADRApprovedPair]:
        return self._ticker_index.get(ticker, [])

    async def on_registry_update(self, event: object) -> None:
        """Reload the registry from a pair-registry event (or raw iterable)."""
        pairs = getattr(event, "registry", event)
        self._load_registry(pairs)  # type: ignore[arg-type]

    # -- FX cache ------------------------------------------------------------

    async def on_fx_rate(self, event: FXRateEvent) -> None:
        """Cache the latest FX rate keyed by base currency."""
        self._fx_rates[event.base_currency] = event

    def get_usd_rate(self, currency: str) -> Decimal | None:
        """USD per 1 unit of ``currency``; ``None`` if missing or stale."""
        if currency == "USD":
            return Decimal(1)
        event = self._fx_rates.get(currency)
        if event is None or event.is_stale:
            return None
        return event.mid

    # -- bar handling --------------------------------------------------------

    async def on_daily_bar(self, event: BarEvent) -> None:
        """Bus handler: evaluate the bar and publish any resulting signals."""
        for signal in self.evaluate(event):
            await self._bus.publish(Topic.SIGNALS, signal)

    def evaluate(self, event: BarEvent) -> list[HongSusmelSignalEvent]:
        """Pure, synchronous core: returns the signals this bar produces.

        Separated from :meth:`on_daily_bar` so unit tests can assert on the
        emitted signals without a bus.
        """
        signals: list[HongSusmelSignalEvent] = []
        bar_date = event.timestamp_exchange.date()

        for pair in self._pairs_for_ticker(event.ticker):
            state = self._states[pair.pair_id]
            state.update_price(event.ticker, event.close, bar_date)

            if not state.both_legs_fresh():
                continue  # stale-leg guard

            fx_rate = self.get_usd_rate(pair.underlying_currency)
            if fx_rate is None:
                continue  # FX stale/missing → skip (graceful degradation)

            assert state.adr_price is not None and state.underlying_price is not None
            local_usd = state.underlying_price * fx_rate
            spread = state.adr_price - (local_usd / pair.adr_ratio)

            state.rolling_stats.update(spread)
            if state.rolling_stats.count < pair.estimation_days:
                continue  # warm-up

            mu = state.rolling_stats.mean
            sigma = state.rolling_stats.std
            kappa_open = mu + pair.k0 * sigma
            kappa_close = mu + pair.kc * sigma

            if state.position == HSPosition.FLAT:
                if spread > kappa_open:
                    signals.append(
                        self._factory.build(
                            pair, HSSignal.SHORT_ADR, spread, mu, sigma,
                            timestamp_exchange=event.timestamp_exchange,
                        )
                    )
                    state.open_position(entry_date=bar_date)

            elif state.position == HSPosition.OPEN:
                days_held = (bar_date - state.entry_date).days
                if spread < kappa_close:
                    signals.append(
                        self._factory.build(
                            pair, HSSignal.EXIT, spread, mu, sigma,
                            timestamp_exchange=event.timestamp_exchange,
                            days_held=days_held,
                        )
                    )
                    state.close_position()
                elif days_held >= pair.holding_days:
                    signals.append(
                        self._factory.build(
                            pair, HSSignal.FORCE_CLOSE, spread, mu, sigma,
                            timestamp_exchange=event.timestamp_exchange,
                            days_held=days_held,
                        )
                    )
                    state.close_position()

        return signals

    # -- introspection -------------------------------------------------------

    def state_for(self, pair_id: str) -> HSPairState | None:
        return self._states.get(pair_id)

    async def run(self) -> None:  # pragma: no cover - live entrypoint
        """No-op run loop; the engine is purely event-driven via handlers."""
        return None
