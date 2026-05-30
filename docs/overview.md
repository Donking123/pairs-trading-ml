# Architecture Overview

## High-Level Diagram

```
┌───────────────────────────────────────────────────────────────────────────┐
│                             EXTERNAL WORLD                                 │
│  WRDS Datastream · Polygon.io · Asian Market Feeds · OANDA FX REST        │
└──────────┬──────────────────────────┬──────────────────┬───────────────────┘
           │ historical OHLCV + FX    │ live daily bars  │ fills
           ▼                          ▼                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        INGESTION / GATEWAY LAYER                           │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │ WRDS Fetcher │  │ Market Data      │  │  FX Rate    │  │  Broker    │  │
│  │ (Datastream) │  │ Handler          │  │  Feed       │  │  Gateway   │  │
│  │              │  │ (US ADR + Asian) │  │  (OANDA)    │  │  (US only) │  │
│  └──────┬───────┘  └────────┬─────────┘  └──────┬──────┘  └─────┬──────┘  │
└─────────┼───────────────────┼───────────────────┼───────────────┼──────────┘
          │ HistoricalBarEvent│ BarEvent           │ FXRateEvent   │ FillEvent
          ▼                   ▼                    ▼               ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                           EVENT BUS                                        │
│         (MemoryBus in backtest / asyncio.Queue in live prototype)          │
│  Topics: market-data | fx-rates | signals | risk-decisions                 │
│          orders | fills | positions | pair-registry | alerts               │
└──┬───────────────────┬──────────────────┬─────────────────────────────────┘
   │                   │                  │
   ▼                   ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  ┌────────────────┐
│  Hong &      │  │    Risk      │  │  Asian           │  │  Position /    │
│  Susmel      │  │  Management  │  │  Execution       │  │  PnL Engine    │
│  Engine      │  │  Engine      │  │  Sequencer       │  │                │
└──────┬───────┘  └──────┬───────┘  └──────────────────┘  └────────────────┘
       │ SignalEvent      │ RiskDecision
       └──────────────────┴──────────────────────────────────▶ back to bus
┌───────────────────────────────────────────────────────────────────────────┐
│                        RESEARCH LAYER (offline)                            │
│  Asian ADR Screener → Cointegration Test → Liquidity Filter → Pair Registry│
└───────────────────────────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────────────────────────┐
│                        INFRASTRUCTURE LAYER                                │
│  TimescaleDB │ PostgreSQL │ Redis │ S3/MinIO │ Prometheus │ Grafana        │
└───────────────────────────────────────────────────────────────────────────┘
```

## Event Flow — Live Trading

```
1.  Market Data Handler (U.S. ADR, daily bar)    ──▶  EVENT BUS (topic: market-data)
2.  Market Data Handler (Asian underlying, daily) ──▶  EVENT BUS (topic: market-data)
3.  FX Rate Feed (OANDA daily close)              ──▶  EVENT BUS (topic: fx-rates)
4.  Hong & Susmel Engine  ◀── EVENT BUS (subscribe: market-data + fx-rates)
5.  Hong & Susmel Engine   ──▶ EVENT BUS (topic: signals)
6.  Risk Engine            ◀── EVENT BUS (subscribe: signals + positions)
7.  Risk Engine            ──▶ EVENT BUS (topic: risk-decisions)
8.  Asian Execution Sequencer ◀── EVENT BUS (subscribe: risk-decisions + market-data + fills)
9.  Asian Execution Sequencer ──▶ U.S. Broker Gateway (ADR short, then cover)
10. U.S. Broker Gateway    ──▶ EVENT BUS (topic: fills)
11. Position Engine        ◀── EVENT BUS (subscribe: fills + market-data + fx-rates)
12. Position Engine        ──▶ EVENT BUS (topic: positions)
13. ROCE/RUCE Calculator   ◀── EVENT BUS (subscribe: fills) [closed round-trips]
```

## Event Flow — Backtesting

```
1.  Datastream Parquet Cache (Asian underlying prices)  ──▶  Replay Engine
2.  Datastream Parquet Cache (U.S. ADR prices)          ──▶  Replay Engine
3.  OANDA FX Parquet Cache (daily FX close rates)       ──▶  Replay Engine
4.  Replay Engine                                       ──▶  MemoryBus (synchronous)
5.  Hong & Susmel Engine (unchanged)    ◀── MemoryBus
6.  Risk Engine (unchanged)             ◀── MemoryBus
7.  Asian Execution Sequencer (backtest mode: next-bar fills) ◀── MemoryBus
8.  Simulated U.S. Exchange             ──▶ MemoryBus (fills)
9.  Position Engine (unchanged)         ◀── MemoryBus
10. ROCE/RUCE Calculator (unchanged)    ◀── MemoryBus
```

> **Key invariant**: Hong & Susmel Engine, Risk Engine, Position Engine, and ROCE/RUCE Calculator are **identical** in live and backtest modes. Only the bus implementation, data sources, and broker gateway differ.

## Research Flow — Pair Selection (Offline)

All research steps are implemented as standalone scripts in `datastream/` — no `src/asian_adr/` package is involved.

```
1.  datastream/fetch_datastream_adr_data.py   — pulls U.S. ADR OHLCV from WRDS Datastream
                                                → data/parquet/adr/adr_prices.parquet
                                                → data/parquet/adr/adr_reference.parquet
2.  datastream/fetch_datastream_global_data.py — pulls Asian underlying OHLCV from WRDS
                                                → data/parquet/global/global_prices.parquet
3.  datastream/fetch_fx_history.py            — pulls SPOT FX rates from WRDS Datastream
                                                (CCY/USD inverted to USD/CCY on output)
                                                → data/parquet/fx/fx_rates.parquet
4.  datastream/run_asian_adr_screening.py     — reads the three Parquet caches and runs:
                                                  · Asian exchange filter
                                                  · ADR ratio estimation (price-median + snap to standard)
                                                  · Dollar spread: P_ADR − (P_local × FX) / ratio
                                                  · ADF + Phillips-Perron cointegration test (5%)
                                                  · Liquidity filter: ≥ 50% non-zero return days, ≥ 2 years
                                                  · Roll (1984) effective spread estimation
                                                → config/pairs/asian_adr_pairs.json
                                                → config/pairs/screening_diagnostics.json
5.  datastream/rescreen.py                    — incremental re-screening orchestrator (Phase 7):
                                                  · auto-detects Parquet tail dates
                                                  · fetches only the missing date gap from WRDS
                                                  · re-runs the full screening pipeline
                                                  · writes changelog (added / dropped / changed pairs)
                                                  · backs up the previous registry
```

## Concurrency Model

| Phase | Model | Notes |
|-------|-------|-------|
| Phase 1–2 | Single process, `asyncio` event loop, coroutine-per-component | |
| Phase 3+ | Multi-process; each component is a separate process; Kafka as bus | |
| Phase 5+ | Kubernetes pods; each component independently scaled | |

## Architecture Principles

| Principle | Application |
|-----------|-------------|
| **Point-in-time correctness** | Pair selection never uses future data; ADR ratio history enforced as-of selection date |
| **Research / production parity** | Same H&S Engine, Risk Engine, and Position Engine in research, backtest, and live |
| **β = ratio, never estimated** | ADR conversion ratio is legally fixed; OLS estimation introduces spurious time-varying hedges |
| **Dollar spread, not log-price** | Hong & Susmel use raw dollar spread; log-spread distorts threshold geometry for high-ratio pairs |
| **One-sided entry by design** | Asian short-selling restrictions eliminate the symmetric trade; `HSSignal` has no `LONG_ADR` variant |
| **No FX hedge** | Overnight gap makes simultaneous FX spot submission impossible; FX is conversion-only |
| **Event immutability** | All events are frozen Pydantic models; mutable state lives only in engines |
| **Graceful degradation** | FX stale → skip spread; bar gap → skip signal; zero-return excess → block entry; never crash |
| **Next-bar fill discipline** | Simulated broker always fills on the bar after order submission; models the overnight gap correctly |
| **RUCE as live tracking metric** | RUCE (0.5× denominator on ADR leg) is the economically realistic measure; ROCE reported for comparison |

## System State Recovery Sequence (Live)

```
1. Load pair registry from PostgreSQL (active pairs only)
2. Rebuild rolling spread stats from last T bars of Parquet cache
3. Reload open position state from PostgreSQL
4. Reconcile open orders with broker REST API
5. Resume market data subscriptions (U.S. ADR + Asian feeds + FX)
6. Mark system READY; begin processing daily bars
```
