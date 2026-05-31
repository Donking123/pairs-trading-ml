"""Asian execution sequencer / OMS (architecture §3.9).

Asian markets and the NYSE have **zero trading-hour overlap**, so a pair is
always established across the overnight gap — never with a simultaneous two-leg
submission (which would produce systematically optimistic backtest fills). This
sequencer encodes that discipline as a per-pair state machine:

    IDLE ──SHORT_ADR──▶ AWAITING_LOCAL ──ADR fill──▶ AWAITING_ASIA_OPEN
        ──spread holds──▶ OPEN (buy local)
        ──spread reversed──▶ IDLE (cover naked short, emit abort alert)
    OPEN ──EXIT/FORCE_CLOSE──▶ IDLE (sell local + cover ADR)

Orders are emitted as :class:`OrderRequest` events on the ``orders`` topic; a
broker gateway (live) or simulated exchange (backtest) consumes them and
publishes fills back on ``fills``. The sequencer therefore never imports the
``gateways`` package — it only depends on ``core`` and ``event_bus`` (layering
rule). Local-leg quantities hedge the ADR leg exactly: ``adr_qty × ratio``.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from itertools import count
from uuid import UUID, uuid4

from asian_adr.core import (
    AsianADRApprovedPair,
    BarEvent,
    Clock,
    FillEvent,
    FXRateEvent,
    HongSusmelSignalEvent,
    HSSignal,
    OrderRequest,
    OrderType,
    FOREIGN_EQUITY,
    US_EQUITY,
)
from asian_adr.event_bus import AbstractEventBus, Topic

from .sequencer_state import AdrOvernightAbortEvent, SeqPhase, SequencerState

logger = logging.getLogger(__name__)

__all__ = ["AsianExecutionSequencer"]


class AsianExecutionSequencer:
    """Sequences overnight two-leg execution for every active pair.

    Parameters
    ----------
    bus:
        Event bus; orders go to :attr:`Topic.ORDERS`, alerts to ``alerts``.
    clock:
        Time source for order timestamps.
    pairs:
        Active pair registry (indexed by ``pair_id`` and by ticker).
    notional_per_leg:
        Target USD notional for the ADR leg; share quantity is
        ``floor(notional / last_adr_price)``.
    """

    def __init__(
        self,
        bus: AbstractEventBus,
        clock: Clock,
        pairs: list[AsianADRApprovedPair],
        notional_per_leg: Decimal = Decimal("100000"),
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._notional_per_leg = notional_per_leg
        self._pairs: dict[str, AsianADRApprovedPair] = {p.pair_id: p for p in pairs}
        self._states: dict[str, SequencerState] = {
            p.pair_id: SequencerState(p.pair_id, p.adr_ticker, p.underlying_ticker)
            for p in pairs
        }
        self._ticker_to_pair: dict[str, str] = {}
        for p in pairs:
            self._ticker_to_pair[p.adr_ticker] = p.pair_id
            self._ticker_to_pair[p.underlying_ticker] = p.pair_id

        self._fx_rates: dict[str, FXRateEvent] = {}
        self._last_price: dict[str, Decimal] = {}
        self._gateways: dict[str, object] = {}
        self._order_seq = count(1)

    # -- wiring helpers ------------------------------------------------------

    def register_gateway(self, venue: str, gateway: object) -> None:
        """Record a venue's gateway (optional; canonical routing is via the bus)."""
        self._gateways[venue] = gateway

    async def on_fx_rate(self, event: FXRateEvent) -> None:
        self._fx_rates[event.base_currency] = event

    def _usd_rate(self, currency: str) -> Decimal | None:
        if currency == "USD":
            return Decimal(1)
        event = self._fx_rates.get(currency)
        if event is None or event.is_stale:
            return None
        return event.mid

    # -- signal entry --------------------------------------------------------

    async def on_signal(self, event: HongSusmelSignalEvent) -> None:
        """Dispatch a signal to the entry or exit path by its type."""
        if event.signal == HSSignal.SHORT_ADR:
            await self._on_entry_signal(event)
        elif event.signal in (HSSignal.EXIT, HSSignal.FORCE_CLOSE):
            await self.on_exit_signal(event)

    async def _on_entry_signal(self, event: HongSusmelSignalEvent) -> None:
        state = self._states.get(event.pair_id)
        if state is None or state.phase != SeqPhase.IDLE:
            return
        submitted = await self._submit_sell_adr(event)
        if submitted:
            state.transition(SeqPhase.AWAITING_LOCAL, signal_event=event)

    # -- fill handling -------------------------------------------------------

    async def on_fill(self, fill: FillEvent) -> None:
        """Record the ADR short fill and arm the Asia-open recheck."""
        state = self._states.get(fill.pair_id) if fill.pair_id else None
        if state is None or state.phase != SeqPhase.AWAITING_LOCAL:
            return
        if fill.ticker != state.adr_ticker or fill.side != "sell":
            return
        state.adr_fill = fill
        state.transition(SeqPhase.AWAITING_ASIA_OPEN)

    # -- bar handling (price cache + Asia-open recheck) ----------------------

    async def on_bar(self, event: BarEvent) -> None:
        # Cache the latest close for order sizing across all bars.
        self._last_price[event.ticker] = event.close

        pair_id = self._ticker_to_pair.get(event.ticker)
        if pair_id is None:
            return
        state = self._states[pair_id]
        if state.phase != SeqPhase.AWAITING_ASIA_OPEN:
            return
        pair = self._pairs[pair_id]
        # Only the underlying (Asia) bar drives the recheck at its open.
        if event.ticker != pair.underlying_ticker:
            return

        fx_rate = self._usd_rate(pair.underlying_currency)
        if fx_rate is None:
            return  # cannot recheck without FX → wait for next underlying bar

        assert state.adr_fill is not None and state.signal_event is not None
        local_usd_open = event.open * fx_rate
        spread = state.adr_fill.fill_price - (local_usd_open / pair.adr_ratio)

        if spread > state.signal_event.kappa_close:
            await self._submit_buy_local(pair, state)
            state.transition(SeqPhase.OPEN)
        else:
            await self._submit_adr_cover(pair, state, reason="overnight_abort")
            await self._bus.publish(
                Topic.ALERTS,
                AdrOvernightAbortEvent(
                    timestamp_exchange=event.timestamp_exchange,
                    timestamp_received=self._clock.now(),
                    pair_id=pair_id,
                ),
            )
            state.reset()

    # -- exit ----------------------------------------------------------------

    async def on_exit_signal(self, event: HongSusmelSignalEvent) -> None:
        if event.signal not in (HSSignal.EXIT, HSSignal.FORCE_CLOSE):
            return
        state = self._states.get(event.pair_id)
        if state is None or state.phase != SeqPhase.OPEN:
            return
        pair = self._pairs[event.pair_id]
        reason = "force_close" if event.signal == HSSignal.FORCE_CLOSE else "exit"
        await self._submit_sell_local(pair, state, reason=reason)
        await self._submit_adr_cover(pair, state, reason=reason)
        state.reset()

    # -- order construction --------------------------------------------------

    def _next_order_id(self) -> str:
        return f"seq-{next(self._order_seq)}"

    def _risk_decision_id(self, state: SequencerState) -> UUID:
        """Stand-in linkage to the originating signal (real RiskDecision in live)."""
        if state.signal_event is not None:
            return state.signal_event.event_id
        return uuid4()

    async def _publish_order(self, order: OrderRequest) -> None:
        await self._bus.publish(Topic.ORDERS, order)

    async def _submit_sell_adr(self, event: HongSusmelSignalEvent) -> bool:
        pair = self._pairs[event.pair_id]
        price = self._last_price.get(pair.adr_ticker)
        if price is None or price <= 0:
            logger.warning(
                "no ADR price to size short; skipping entry",
                extra={"pair_id": pair.pair_id},
            )
            return False
        qty = (self._notional_per_leg / price).to_integral_value(rounding="ROUND_DOWN")
        if qty <= 0:
            return False
        await self._publish_order(
            OrderRequest(
                timestamp_exchange=event.timestamp_exchange,
                timestamp_received=self._clock.now(),
                pair_id=pair.pair_id,
                risk_decision_id=event.event_id,
                ticker=pair.adr_ticker,
                side="sell",
                quantity=qty,
                order_type=OrderType.MARKET,
                venue=US_EQUITY,
                currency="USD",
                is_short_sale=True,
            )
        )
        return True

    async def _submit_buy_local(
        self, pair: AsianADRApprovedPair, state: SequencerState
    ) -> None:
        assert state.adr_fill is not None
        qty = state.adr_fill.fill_quantity * pair.adr_ratio  # exact hedge
        await self._publish_order(
            OrderRequest(
                timestamp_exchange=self._clock.now(),
                timestamp_received=self._clock.now(),
                pair_id=pair.pair_id,
                risk_decision_id=self._risk_decision_id(state),
                ticker=pair.underlying_ticker,
                side="buy",
                quantity=qty,
                order_type=OrderType.MARKET,
                venue=FOREIGN_EQUITY,
                currency=pair.underlying_currency,
            )
        )

    async def _submit_sell_local(
        self, pair: AsianADRApprovedPair, state: SequencerState, *, reason: str
    ) -> None:
        assert state.adr_fill is not None
        qty = state.adr_fill.fill_quantity * pair.adr_ratio
        await self._publish_order(
            OrderRequest(
                timestamp_exchange=self._clock.now(),
                timestamp_received=self._clock.now(),
                pair_id=pair.pair_id,
                risk_decision_id=self._risk_decision_id(state),
                ticker=pair.underlying_ticker,
                side="sell",
                quantity=qty,
                order_type=OrderType.MARKET,
                venue=FOREIGN_EQUITY,
                currency=pair.underlying_currency,
            )
        )

    async def _submit_adr_cover(
        self, pair: AsianADRApprovedPair, state: SequencerState, *, reason: str
    ) -> None:
        assert state.adr_fill is not None
        qty = state.adr_fill.fill_quantity
        await self._publish_order(
            OrderRequest(
                timestamp_exchange=self._clock.now(),
                timestamp_received=self._clock.now(),
                pair_id=pair.pair_id,
                risk_decision_id=self._risk_decision_id(state),
                ticker=pair.adr_ticker,
                side="buy",
                quantity=qty,
                order_type=OrderType.MARKET,
                venue=US_EQUITY,
                currency="USD",
                is_short_sale=False,
            )
        )

    # -- introspection -------------------------------------------------------

    def phase_of(self, pair_id: str) -> SeqPhase | None:
        state = self._states.get(pair_id)
        return state.phase if state else None

    async def run(self) -> None:  # pragma: no cover - live entrypoint
        return None
