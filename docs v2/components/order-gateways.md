# Broker Gateway

**Package**: `asian_adr.gateways`

## Responsibilities

- Translate internal `OrderRequest` into broker-specific API calls for U.S. ADR trades only
- Handle ADR short-leg (SELL) and cover-leg (BUY) via Alpaca or Interactive Brokers
- Map broker order IDs to internal order IDs
- Receive fills; publish `FillEvent` to the event bus
- Verify short-locate availability before submitting any SELL (short)

**Inputs**: `OrderRequest` from the Asian Execution Sequencer
**Outputs**: `FillEvent` (topic: `fills`)

## Module Structure

```
gateways/
├── base.py
├── interactive_brokers/
│   ├── tws_gateway.py          # U.S. ADR equity only; primary gateway
│   └── short_locate.py         # Query IB locate availability
├── alpaca/
│   ├── rest_gateway.py
│   └── ws_gateway.py
└── simulation/
    ├── simulated_us_gateway.py
    └── simulated_foreign_gateway.py   # Backtest only
```

## Supported Gateways

| Gateway | Protocol | Venue | Notes |
|---------|----------|-------|-------|
| Interactive Brokers | TWS API / FIX | U.S. ADR equities | Primary broker; full institutional feature set |
| Alpaca | REST / WebSocket | U.S. ADR equities | Commission-free; paper trading fallback |
| Simulation | In-process | U.S. equities | Backtest and paper trading |

## No Foreign Equity Gateway

The local (Asian) leg in live trading is executed via the operator's own Asian brokerage account. The platform generates the `BUY_LOCAL` order instruction and logs it; actual local execution is external. In backtest mode `SimulatedForeignExchange` handles local fills using next-bar open prices.

## Short-Locate Verification

```python
class IBShortLocate:
    async def verify(self, ticker: str, quantity: Decimal) -> LocateResult:
        """
        Queries IB TWS for short availability before every SELL order.
        Returns LocateResult(available=bool, borrow_rate=Decimal).
        The Risk Engine's ShortLocateRule blocks the signal if available=False.
        """
```

## Failure Handling

| Failure | Response |
|---------|----------|
| Short-locate unavailable | Risk Engine blocks SELL; pair skipped for this signal |
| Order rejected by broker | Publish `OrderRejectedEvent`; sequencer transitions to abort path |
| Fill timeout | Alert operator; treat as partial fill; reconcile on next poll |
| Connection loss | Exponential backoff reconnect; replay any unconfirmed orders on reconnect |
