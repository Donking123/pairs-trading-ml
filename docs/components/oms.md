# Asian Execution Sequencer

**Package**: `asian_adr.strategy.hong_susmel`

## Responsibilities

- Bridge the overnight gap between U.S. close and Asian market open
- Run a per-pair state machine: short ADR first, then conditionally buy local next bar
- Recheck spread at Asia open before committing the local leg
- Abort and cover the naked ADR short if spread reversed overnight
- Handle simultaneous close (sell local + cover ADR) on EXIT / FORCE_CLOSE signals
- Route orders to the U.S. Broker Gateway via standard `OrderRequest`

**Inputs**: `RiskDecision` (topic: `risk-decisions`), `FillEvent` (topic: `fills`), `BarEvent` (topic: `market-data`)
**Outputs**: `OrderRequest` (to Broker Gateway)

## Module Structure

```
strategy/hong_susmel/
├── execution_sequencer.py    # AsianExecutionSequencer state machine
└── sequencer_state.py        # SeqPhase enum, per-pair mutable state
```

## State Machine

```
IDLE
  ──▶ on SHORT_ADR signal:
        submit SELL_ADR (U.S. close bar)
        transition → AWAITING_LOCAL

AWAITING_LOCAL
  ──▶ on ADR fill confirmed + next Asia bar arrives:
        recompute spread vs κ_close
        if spread > κ_close    →  submit BUY_LOCAL  →  transition OPEN
        if spread reversed     →  submit BUY_ADR cover (abort) → IDLE
                                  publish AdrOvernightAbortEvent

OPEN
  ──▶ on EXIT or FORCE_CLOSE signal:
        submit SELL_LOCAL (Asia bar)
        submit BUY_ADR cover (same or next U.S. bar)
        transition → IDLE
```

## Implementation

```python
class AsianExecutionSequencer:
    async def on_signal(self, event: HongSusmelSignalEvent) -> None:
        state = self._states[event.pair_id]
        if event.signal == HSSignal.SHORT_ADR and state.phase == SeqPhase.IDLE:
            await self._submit_sell_adr(event)
            state.transition(SeqPhase.AWAITING_LOCAL, signal_event=event)

    async def on_fill(self, fill: FillEvent) -> None:
        state = self._states.get(fill.pair_id)
        if state is None or state.phase != SeqPhase.AWAITING_LOCAL:
            return
        if fill.ticker != state.adr_ticker or fill.side != "sell":
            return
        state.adr_fill = fill
        state.transition(SeqPhase.AWAITING_ASIA_OPEN)

    async def on_bar(self, event: BarEvent) -> None:
        for pair_id, state in self._states.items():
            if state.phase != SeqPhase.AWAITING_ASIA_OPEN:
                continue
            if event.ticker != self._pairs[pair_id].underlying_ticker:
                continue
            fx_rate = self._fx_cache.get_usd_rate(self._pairs[pair_id].underlying_currency)
            spread  = (state.adr_fill.fill_price
                       - event.open * fx_rate / self._pairs[pair_id].adr_ratio)

            if spread > state.signal_event.kappa_close:
                await self._submit_buy_local(pair_id, event)
                state.transition(SeqPhase.OPEN)
            else:
                await self._submit_adr_cover(pair_id)
                state.transition(SeqPhase.IDLE)
                await self._bus.publish("alerts", AdrOvernightAbortEvent(pair_id=pair_id))

    async def on_exit_signal(self, event: HongSusmelSignalEvent) -> None:
        if event.signal not in (HSSignal.EXIT, HSSignal.FORCE_CLOSE):
            return
        state = self._states[event.pair_id]
        if state.phase != SeqPhase.OPEN:
            return
        await self._submit_sell_local(event.pair_id)
        await self._submit_adr_cover(event.pair_id)
        state.transition(SeqPhase.IDLE)
```

## Why Not Simultaneous Submission

Asian markets (TSE, HKEX, KRX, etc.) and NYSE have zero trading-hour overlap. Submitting both legs simultaneously via `asyncio.gather` would produce systematically optimistic backtest fills by ignoring the overnight gap. The sequencer models the two-step reality: ADR short at U.S. close, local buy conditionally at next Asian open after re-checking the spread.

## Backtest Mode

In backtest mode the sequencer uses `SimulatedForeignExchange` for local fills and `SimulatedUSExchange` for ADR fills. Both use next-bar fill prices (ADR-005), enforcing the same two-step overnight sequence.
