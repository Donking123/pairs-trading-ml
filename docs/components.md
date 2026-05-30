# Components

Each core component has its own document:

| Component | Package | Document |
|-----------|---------|----------|
| Historical Data Ingestion | `datastream/` scripts | [components/wrds-data-fetcher.md](components/wrds-data-fetcher.md) |
| Market Data Handler | `asian_adr.feed_handler` | [components/feed-handler.md](components/feed-handler.md) |
| FX Rate Feed | `asian_adr.fx_handler` | [components/fx-handler.md](components/fx-handler.md) |
| Asian ADR Pair Selection | `datastream/` scripts | [components/research.md](components/research.md) |
| Hong & Susmel Engine | `asian_adr.strategy.hong_susmel` | [components/strategy.md](components/strategy.md) |
| Event Bus | `asian_adr.event_bus` | [components/event-bus.md](components/event-bus.md) |
| Risk Management Engine | `asian_adr.risk` | [components/risk.md](components/risk.md) |
| Position & PnL Engine | `asian_adr.position` | [components/position.md](components/position.md) |
| Asian Execution Sequencer | `asian_adr.strategy.hong_susmel` | [components/oms.md](components/oms.md) |
| Broker Gateway | `asian_adr.gateways` | [components/order-gateways.md](components/order-gateways.md) |
| ROCE / RUCE Calculator | `asian_adr.backtest` | [components/roce-ruce.md](components/roce-ruce.md) |
| Backtest Engine | `asian_adr.backtest` | [components/backtest.md](components/backtest.md) |

## Stateful vs Stateless

| Component | Stateful? | State Location |
|-----------|-----------|----------------|
| Historical Data Ingestion | No | Writes Parquet cache; stateless at runtime |
| Market Data Handler | Yes | In-process latest bar per ticker |
| FX Rate Feed | Yes | In-process rate cache |
| Pair Selection | No | Offline pipeline; writes to JSON registry |
| Hong & Susmel Engine | Yes | Per-pair rolling stats + position phase |
| Event Bus | Yes | MemoryBus (backtest) / asyncio.Queue (live) / Kafka (production) |
| Risk Engine | Yes | In-process exposure tracker |
| Position Engine | Yes | In-process + PostgreSQL |
| Asian Execution Sequencer | Yes | Per-pair sequencer state machine |
| Broker Gateway | No | Stateless adapter; order state lives in OMS |
| ROCE / RUCE Calculator | Yes | Accumulates per-pair and aggregate distributions |
| Backtest Engine | No | Stateless orchestrator; drives all other components |

Stateful components must handle restart gracefully by replaying recent events or loading their last snapshot from storage.

## Operation Modes

| Component | Research (offline) | Backtest | Live |
|-----------|-------------------|----------|------|
| Historical Data Ingestion | ✅ primary (`datastream/`) | reads cache | reads cache |
| Market Data Handler | — | — | ✅ |
| FX Rate Feed | — | reads cache | ✅ |
| Pair Selection | ✅ primary (`datastream/`) | — | — |
| Hong & Susmel Engine | — | ✅ | ✅ |
| Event Bus | — | MemoryBus | asyncio.Queue / Kafka |
| Risk Engine | — | ✅ | ✅ |
| Position Engine | — | ✅ | ✅ |
| Asian Execution Sequencer | — | ✅ (next-bar fills) | ✅ |
| Broker Gateway | — | Simulated | IB / Alpaca |
| ROCE / RUCE Calculator | — | ✅ | ✅ |
| Backtest Engine | — | ✅ primary | — |
