# Asian ADR Pairs Strategy — System Architecture

> **Version**: 1.0.0 | **Status**: Living Document | **Language**: Python 3.12+ | **Paradigm**: Event-Driven, Event-Sourced
>
> **Strategy Reference**: Hong & Susmel (2013), *Pairs-Trading in the Asian ADR Market*

## Documents

### General

| File | Contents |
|------|----------|
| [overview.md](overview.md) | High-level architecture diagram, event flows (live + backtest + research), concurrency model, architecture principles |
| [strategy-specification.md](strategy-specification.md) | Dollar spread formula, entry/exit thresholds, execution model, ROCE/RUCE metrics, liquidity bucketing |
| [data-contracts.md](data-contracts.md) | All Pydantic event schemas, topic map, Parquet cache schemas |
| [technology-stack.md](technology-stack.md) | Language, messaging, databases, quantitative libraries, Python dependencies |
| [project-structure.md](project-structure.md) | Full directory layout, `datastream/` script inventory, structural rules, live runner wiring |
| [roadmap.md](roadmap.md) | Phased implementation plan (Phase 1–7) with deliverables per phase |
| [engineering-practices.md](engineering-practices.md) | Typing, linting, testing hierarchy, config management, ADR table, reproducibility checklist |
| [backtesting.md](backtesting.md) | Replay engine design, cost model, slippage models, ROCE/RUCE validation targets |

### Components

| File | Contents |
|------|----------|
| [components.md](components.md) | Component index, stateful/stateless summary, operation modes by phase |
| [components/wrds-data-fetcher.md](components/wrds-data-fetcher.md) | Historical data ingestion — `datastream/` scripts, Parquet schemas, WRDS SQL queries |
| [components/feed-handler.md](components/feed-handler.md) | Market Data Handler — live daily bars for U.S. ADRs and Asian underlyings |
| [components/fx-handler.md](components/fx-handler.md) | FX Rate Feed — OANDA daily rates for USD spread conversion |
| [components/research.md](components/research.md) | Pair selection — `datastream/run_asian_adr_screening.py` and `rescreen.py` |
| [components/strategy.md](components/strategy.md) | Hong & Susmel Engine — spread computation, rolling stats, signal generation |
| [components/event-bus.md](components/event-bus.md) | Event Bus — pub/sub backbone, topic map, backpressure |
| [components/risk.md](components/risk.md) | Risk Management Engine — pre-trade rules, kill switch, circuit breakers, audit logging |
| [components/position.md](components/position.md) | Position & PnL Engine — inventory, cost basis, multi-currency mark-to-market |
| [components/oms.md](components/oms.md) | Asian Execution Sequencer — overnight gap state machine, abort cover logic |
| [components/order-gateways.md](components/order-gateways.md) | Broker Gateway — U.S. ADR short/cover via Interactive Brokers or Alpaca |
| [components/roce-ruce.md](components/roce-ruce.md) | ROCE / RUCE Calculator — per-trade metrics, liquidity bucket attribution, Table 7-B distributions |
| [components/backtest.md](components/backtest.md) | Backtest Engine — multi-feed replay, simulated exchanges, cost model, tearsheet |

### Infrastructure

| File | Contents |
|------|----------|
| [infrastructure/observability.md](infrastructure/observability.md) | Structured logging, Prometheus metrics, Grafana dashboards, latency SLOs, alert rules |
| [infrastructure/storage.md](infrastructure/storage.md) | Persistence layer, storage decision matrix, Parquet layout, data retention |
| [infrastructure/deployment.md](infrastructure/deployment.md) | Local dev setup, Docker Compose, CI/CD pipeline, environment variables reference |

## Core Principles

- **Event immutability**: all events are frozen Pydantic models; mutable state lives only inside engines
- **Research / production parity**: H&S Engine, Risk Engine, and Position Engine are identical in backtest and live modes — only the bus and exchange gateway differ
- **β = ADR ratio, never OLS-estimated**: the cointegrating vector is the legally fixed conversion ratio (ADR-002)
- **One-sided entry by design**: Asian short-selling restrictions eliminate the symmetric trade; `HSSignal` has no `LONG_ADR` variant (ADR-004)
- **Next-bar fill discipline**: simulated broker always fills on the bar after submission, correctly modelling the overnight gap (ADR-005)
- **No FX hedge**: the overnight gap makes simultaneous FX spot submission impossible; FX is used only for spread computation (ADR-003)
- **Graceful degradation**: stale FX → skip spread; bar gap → skip signal; zero-return excess → block entry; never crash
- **`datastream/` is standalone**: all historical data ingestion and pair selection scripts run independently — they never import from `src/asian_adr/`

## Quick Start

```bash
cd datastream/

# 1. Fetch historical data (one-time)
python fetch_datastream_adr_data.py
python fetch_datastream_global_data.py
python fetch_fx_history.py

# 2. Run pair selection
python run_asian_adr_screening.py

# 3. Run backtest
python run_backtest.py

# 4. Generate tearsheet
python backtest_report.py

# 5. Re-screen with new data (weekly, Phase 7)
python rescreen.py
```

See [infrastructure/deployment.md](infrastructure/deployment.md) for the full local setup and [roadmap.md](roadmap.md) for the phased implementation plan.
