# Implementation Roadmap

## Phase 1 — Research Foundation: WRDS Data + Pair Selection

**Objectives**: Validate Asian ADR pair selection end-to-end; no live trading

**Architecture decisions**
- Standalone scripts in `datastream/`; no event bus, no `src/asian_adr/` package required
- WRDS Datastream (`tr_ds_equities.wrds_ds2dsf`) for historical OHLCV prices
- WRDS Datastream (`trdstrm.ds2fxrate`) for historical daily FX rates

**Deliverables**
- `datastream/fetch_datastream_adr_data.py` — WRDS query for U.S. ADR OHLCV + reference → Parquet
- `datastream/fetch_datastream_global_data.py` — WRDS query for Asian underlying OHLCV → Parquet
- `datastream/fetch_fx_history.py` — WRDS Datastream SPOT FX rates (inverted to USD/CCY) → Parquet
- `datastream/run_asian_adr_screening.py` — full pipeline: exchange filter, ratio estimation, dollar spread, ADF/PP, liquidity filter, Roll spread → `asian_adr_pairs.json`
- `datastream/rescreen.py` — incremental re-screening orchestrator: gap-fetch + re-run + changelog (Phase 7)

**Testing**: Unit tests on synthetic ADR/FX series with known spread properties

**Accepted technical debt**: No event bus; no persistence beyond Parquet and JSON

---

## Phase 2 — Backtesting Engine

**Objectives**: Validate strategy on historical WRDS Datastream data; reproduce paper benchmarks

**Architecture decisions**
- `MemoryBus` (synchronous) for speed and determinism
- `SimulatedClock` driven by merged U.S. / Asian / FX daily streams
- Point-in-time pair registry (no look-ahead on pair selection dates)
- Next-bar fill model correctly simulates overnight gap
- Full cost model: commission, SEC fee, short borrow, stamp duty, local levy

**Deliverables**
- `core/events.py` — all event dataclasses
- `strategy/hong_susmel/engine.py` — `HongSusmelEngine`
- `strategy/hong_susmel/execution_sequencer.py` — `AsianExecutionSequencer`
- `strategy/hong_susmel/state.py` — `HSPairState`, `WelfordRollingStats(ddof=1)`
- `risk/` — `HoldingPeriodForceCloseRule`, `ZeroReturnDayFilter`, `OvernightAbortCoverRule`
- `backtest/` — complete module with multi-feed replay
- `backtest/roce_ruce_calculator.py` — ROCE/RUCE per trade; aggregate distributions
- `backtest/report.py` — HTML tearsheet matching paper Table 7-B format

**Validation targets**: median ROCE ≈ 2.8%, median RUCE ≈ 5.3%, median duration ≈ 3 days

---

## Phase 3 — Live Data Feed Integration

**Objectives**: Connect to Polygon.io (U.S. ADR) and Asian feeds; run in paper mode

**Architecture decisions**
- `AsyncioBus` replaces `MemoryBus`
- Feed handlers subscribe only to tickers in the active pair registry
- Exponential backoff reconnect on all external connections

**Deliverables**
- `feed_handler/connectors/polygon.py` — U.S. ADR end-of-day WebSocket
- `feed_handler/connectors/asian_feed.py` — Asian underlying daily bars
- `fx_handler/connectors/oanda.py` — OANDA daily FX close rates
- `runners/live_runner.py` — wires all components (paper mode)
- Integration tests replaying captured feed sessions

---

## Phase 4 — Persistence + Event Bus

**Objectives**: Survive restarts; persist fills, positions, and pair state

**Architecture decisions**
- PostgreSQL for orders/fills/audit log (SQLAlchemy + asyncpg)
- TimescaleDB for OHLCV history
- Kafka replaces asyncio.Queue as event bus
- Redis for hot spread/position cache

**Deliverables**
- `persistence/` — complete module
- `event_bus/kafka_bus.py`
- Alembic migrations (pair registry, orders, fills, positions, audit log)
- State recovery on startup (reload open positions; rebuild rolling stats from bar history)
- Docker Compose with all infrastructure services

---

## Phase 5 — Live Broker Connectivity

**Objectives**: Execute real U.S. ADR short/cover trades

**Deliverables**
- `gateways/interactive_brokers/tws_gateway.py` — primary U.S. ADR gateway
- `gateways/interactive_brokers/short_locate.py` — verify locate before every SELL
- `runners/data_recorder.py` — record live bars for subsequent backtesting
- Integration tests against Interactive Brokers paper trading account

---

## Phase 6 — Full Risk Controls + Observability

**Objectives**: Production-grade risk management and full observability stack

**Deliverables**
- `risk/rules/` — all rule implementations (notional, drawdown, rate-of-loss, country concentration, kill switch)
- Kill switch REST endpoint
- `monitoring/` — complete module with spread and ROCE/RUCE metrics
- Grafana dashboards: spread monitor, ROCE/RUCE tracker, liquidity monitor
- Alertmanager rules: overnight abort cluster, force-close cluster, zero-return-day spike
- Runbooks for all alert types

---

## Phase 7 — Dynamic Pair Rotation + Kubernetes

**Objectives**: Automate weekly re-screening; production Kubernetes deployment

**Deliverables**
- Weekly cron jobs: re-run Asian ADR screener and registry update
- Helm chart for Kubernetes deployment
- CI/CD pipeline (GitHub Actions → ArgoCD)
- Full type coverage; 80%+ test coverage

**Technical debt resolution**: Address all TODO items from Phases 1–6
