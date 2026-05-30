# Market Data Handler

**Package**: `asian_adr.feed_handler`

## Responsibilities

- Connect to Polygon.io or Alpaca for real-time U.S. ADR end-of-day bars
- Connect to Asian market data provider for foreign underlying daily bars
- Subscribe only to tickers in the active pair registry
- Normalise incoming data into canonical `BarEvent`
- Reconnect with exponential backoff on disconnection
- Detect and flag stale bars (repeated closing price = zero-return day)

**Inputs**: WebSocket / REST streams from Polygon.io, Alpaca, or Asian feed provider
**Outputs**: `BarEvent` (topic: `market-data`), `ZeroReturnEvent` (topic: `alerts`)

## Module Structure

```
feed_handler/
├── connector.py              # WebSocket / REST connection lifecycle
├── normalizer.py             # Provider format → BarEvent
├── subscription_manager.py   # Dynamic subscribe/unsubscribe based on pair registry
├── staleness_monitor.py      # Flags zero-return days; feeds ZeroReturnDayFilter
└── connectors/
    ├── polygon.py            # U.S. ADR equities (Polygon.io)
    ├── alpaca_data.py        # U.S. ADR equities (Alpaca alternative)
    └── asian_feed.py         # Asian underlying equities
```

## Design Rules

- Feed handler never imports from `strategy/`, `risk/`, or `gateways/`
- All raw provider data is normalised to `BarEvent` before entering the bus
- Subscription list is rebuilt whenever a `PairRegistryUpdateEvent` is received
- All time references use the injected `Clock`, never `datetime.now()`

## Failure Handling

| Failure | Response |
|---------|----------|
| WebSocket disconnect | Exponential backoff reconnect |
| Missing bar | Skip; flag gap in metrics; do not compute spread |
| Stale bar (same price as prior day) | Increment zero-return counter; publish `ZeroReturnEvent` |
| Feed gap > 3 bars | Mark affected pairs data-unavailable; skip signals; alert operator |
