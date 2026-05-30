# FX Rate Feed

**Package**: `asian_adr.fx_handler`

## Responsibilities

- Fetch daily FX close rates for all currencies in the active pair registry
- Publish `FXRateEvent` to the event bus at end-of-day (or on-demand in backtest)
- Maintain an in-memory FX rate cache for low-latency lookups by the Hong & Susmel Engine
- FX is used **only for spread computation** — no FX hedging occurs

**Inputs**: OANDA REST v20 (live); WRDS Datastream FX Parquet cache (backtest)
**Outputs**: `FXRateEvent` (topic: `fx-rates`)

## Module Structure

```
fx_handler/
├── rate_cache.py             # In-memory cache: {currency_pair → FXRateEvent}
├── normalizer.py             # Provider ticks → FXRateEvent
├── staleness_monitor.py      # Alert if daily rate not received by bar close
└── connectors/
    └── oanda.py              # OANDA REST v20: daily FX close rates
```

## Rate Cache Interface

```python
class FXRateCache:
    def update(self, event: FXRateEvent) -> None: ...
    def get_usd_rate(self, base_currency: str) -> Decimal | None:
        """Returns USD per 1 unit of base_currency, or None if not cached."""
```

## Design Rules

- FX rate cache is write-only from the handler; all consumers read via `get_usd_rate()`
- `HongSusmelEngine` checks `fx_rate is None` before computing spread; skips affected pairs
- FX rates are always expressed as **USD per 1 unit of base currency** (`mid` field)
- `is_stale=True` is set when the prior day's rate is used as fallback

## Failure Handling

| Failure | Response |
|---------|----------|
| FX rate missing for today | Use prior day's rate with `is_stale=True` |
| Stale FX | H&S Engine skips spread computation for affected pairs; logs warning |
| FX age > 2 business days | Pair marked SUSPENDED; alert operator |
