# Position & PnL Engine

**Package**: `asian_adr.position`

## Responsibilities

- Track open quantity, average-cost basis, and unrealised P&L for every ADR and local leg
- Mark positions to market on each new daily bar
- Convert local-leg P&L to USD using the current FX rate
- Publish `PositionUpdateEvent` on every change
- Detect and alert on leg imbalances (ADR short open but no matching local long)

**Inputs**: `FillEvent` (topic: `fills`), `BarEvent` (topic: `market-data`), `FXRateEvent` (topic: `fx-rates`)
**Outputs**: `PositionUpdateEvent` (topic: `positions`)

## Module Structure

```
position/
├── engine.py
└── pnl_calculator.py
```

## Mark-to-Market (Multi-Currency)

```python
class PositionEngine:
    async def on_bar(self, event: BarEvent) -> None:
        for pair_id, pos in self._positions_for_ticker(event.ticker):
            fx = self._fx_cache.get_usd_rate(pos.local_currency)
            pnl = self._calculate_pnl(pos, event.close, fx)
            await self._bus.publish("positions", PositionUpdateEvent(
                pair_id=pair_id,
                unrealized_pnl_usd=pnl.total_usd,
                trigger="mark_to_market",
            ))
```

## Position State Per Pair

Each pair maintains two position records — one per leg:

| Leg | Venue | Side | Currency |
|-----|-------|------|----------|
| ADR | `us_equity` | Short | USD |
| Local | `foreign_equity` | Long | Local currency (JPY, HKD, KRW, etc.) |

Unrealised P&L for both legs is always expressed in USD. The local leg P&L is converted using the latest cached FX rate; if the rate is stale (`is_stale=True`), the P&L is flagged but still published.

## Leg Imbalance Detection

If an ADR short fill is received but no matching local fill arrives within a configurable window, the engine publishes a `LegImbalanceAlert`. This is the primary signal for the OvernightAbortCoverRule to act.
