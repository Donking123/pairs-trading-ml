# Asian ADR Pairs Trading — System Architecture & Implementation Plan

> **Version**: 1.0.0
> **Status**: Living Document
> **Strategy Reference**: Hong & Susmel (2013), *Pairs-Trading in the Asian ADR Market*
> **Language**: Python 3.12+
> **Paradigm**: Event-Driven, Event-Sourced, Modular
> **Universe**: U.S.-listed ADRs with Asian underlying shares (non-overlapping market hours)
> **Data Source**: WRDS Datastream (historical) · Polygon.io / Alpaca (live U.S.) · Asian Feeds (live foreign) · OANDA REST (FX conversion)

---

## Table of Contents

1.  [High-Level Architecture](#1-high-level-architecture)
2.  [Strategy Specification](#2-strategy-specification)
3.  [Core Components](#3-core-components)
4.  [Architecture Principles](#4-architecture-principles)
5.  [Technology Stack](#5-technology-stack)
6.  [Project Structure](#6-project-structure)
7.  [Internal APIs and Data Contracts](#7-internal-apis-and-data-contracts)
8.  [Concurrency and Runtime Model](#8-concurrency-and-runtime-model)
9.  [Backtesting Design](#9-backtesting-design)
10. [Risk and Reliability](#10-risk-and-reliability)
11. [Observability](#11-observability)
12. [Agile Implementation Roadmap](#12-agile-implementation-roadmap)
13. [Deployment Strategy](#13-deployment-strategy)
14. [Engineering Best Practices](#14-engineering-best-practices)

---

## 1. High-Level Architecture

### 1.1 Overview

The platform is a **layered event-driven system** built around a single strategy: Asian ADR Statistical Arbitrage, as formalised by Hong & Susmel (2013). The strategy exploits mean-reversion in the dollar-denominated spread between U.S.-listed ADRs and their Asian underlying shares. Because Asian markets and NYSE have zero trading-hour overlap, execution is always sequenced overnight: short the ADR at U.S. close, then conditionally buy the local share at the next Asian open.

**Strategy at a glance**

| Property | Value |
|----------|-------|
| Universe | U.S. ADRs with Asian underlyings (JP, KR, HK/CN, IN, AU, TW, SG, TH, ID, PH, MY) |
| Signal | Dollar spread vs rolling µ ± k·σ thresholds |
| Entry direction | Short ADR / long local only (Asian short-selling restrictions) |
| Execution | Overnight sequenced — no simultaneous two-leg submission |
| FX | Conversion to USD for spread computation only; no FX hedge |
| Return metrics | ROCE and RUCE (paper-defined) |
| Holding period | Configurable; default 90 days; force-close at expiry |

The system operates in two modes:

- **Research / Pair Selection** (offline): Queries WRDS Datastream for historical prices, validates cointegration on the dollar spread, applies liquidity filters, and writes approved pairs to the registry.
- **Live / Backtest Trading** (online): Replays or streams daily bars, computes rolling spread thresholds, sequences overnight orders, and tracks ROCE/RUCE performance.

Every state transition is triggered by an immutable event. Components never call each other directly — they publish and subscribe to a central event bus.

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

### 1.2 Event Flow — Live Trading

```
1.  Market Data Handler (U.S. ADR, daily bar)   ──▶  EVENT BUS (topic: market-data)
2.  Market Data Handler (Asian underlying, daily)──▶  EVENT BUS (topic: market-data)
3.  FX Rate Feed (OANDA daily close)             ──▶  EVENT BUS (topic: fx-rates)
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

### 1.3 Event Flow — Backtesting

```
1.  Datastream Parquet Cache (Asian underlying prices)        ──▶  Replay Engine
2.  Datastream Parquet Cache (U.S. ADR prices)                ──▶  Replay Engine
3.  OANDA FX Parquet Cache (daily FX close rates)           ──▶  Replay Engine
4.  Replay Engine                                           ──▶  MemoryBus (synchronous)
5.  Hong & Susmel Engine (unchanged)    ◀── MemoryBus
6.  Risk Engine (unchanged)             ◀── MemoryBus
7.  Asian Execution Sequencer (backtest mode: next-bar fills) ◀── MemoryBus
8.  Simulated U.S. Exchange             ──▶ MemoryBus (fills)
9.  Position Engine (unchanged)         ◀── MemoryBus
10. ROCE/RUCE Calculator (unchanged)    ◀── MemoryBus
```

> **Key invariant**: Hong & Susmel Engine, Risk Engine, Position Engine, and ROCE/RUCE Calculator are **identical** in live and backtest modes. Only the bus implementation, data sources, and broker gateway differ.

### 1.4 Research Flow — Asian ADR Pair Selection (Offline)

All research steps are implemented as standalone scripts in `datastream/` — no `src/asian_adr/` package is involved.

```
1.  datastream/fetch_datastream_adr_data.py   — pulls U.S. ADR OHLCV from WRDS Datastream
                                                (tr_ds_equities.wrds_ds2dsf, typecode='ADR')
                                                → data/parquet/adr/adr_prices.parquet
                                                → data/parquet/adr/adr_reference.parquet
2.  datastream/fetch_datastream_global_data.py — pulls Asian underlying OHLCV from WRDS
                                                (tr_ds_equities.wrds_ds2dsf, Asian exchanges)
                                                → data/parquet/global/global_prices.parquet
3.  datastream/fetch_fx_history.py            — pulls daily FX SPOT rates from WRDS Datastream
                                                (trdstrm.ds2fxrate / trdstrm.ds2fxcode)
                                                Rates stored CCY/USD; inverted to USD/CCY on output
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

### 1.5 Concurrency Model

- **Single process (Phase 1–2)**: `asyncio` event loop, coroutine-per-component
- **Multi-process (Phase 3+)**: Each component is a separate process; Kafka as bus
- **Multi-machine (Phase 5+)**: Kubernetes pods, each component independently scaled

---

## 2. Strategy Specification

### 2.1 Core Formulae

**Dollar spread (USD-denominated)**

```
spread_t  =  P_ADR,t  −  (P_local,t × FX_t) / ratio
```

- `P_ADR,t` — U.S. ADR closing price in USD
- `P_local,t` — Asian underlying closing price in local currency
- `FX_t` — FX spot rate (USD per 1 unit of local currency)
- `ratio` — ADR conversion ratio (local shares per 1 ADR); **fixed structural constant, never OLS-estimated**

**Rolling estimation window (T days)**

```
µ_t  =  mean(spread[t-T : t])
σ_t  =  std(spread[t-T : t],  ddof=1)     # WelfordRollingStats
```

**Entry / exit thresholds**

```
κ_open  =  µ_t + k0 × σ_t      # entry: emit SHORT_ADR when spread_t > κ_open
κ_close =  µ_t + kc × σ_t      # exit:  emit EXIT when spread_t < κ_close
```

Z-score equivalence: `z_t = (spread_t − µ_t) / σ_t`; entry at `z > k0`, exit at `z < kc`.

`kc = 0` (default) closes when spread returns to its rolling mean — law-of-one-price convergence.

### 2.2 Parameter Grid

| Parameter | Values tested (paper) | Platform default |
|-----------|----------------------|-----------------|
| k0 (entry multiplier) | 1.65, 2.0, 2.33 | **2.0** |
| kc (exit multiplier) | 0.0, 0.5, 1.0 | **0.0** |
| T (estimation window, days) | 30, 60, 90, 120 | **60** |
| H (max holding period, days) | 30, 90, 120 | **90** |

### 2.3 Execution Model

Asian markets (TSE, HKEX, KRX, etc.) and NYSE have **zero trading-hour overlap**. Establishing both legs always crosses the overnight gap.

**Open sequence**

```
Day D,  U.S. close (16:00 ET)       →  spread_D > κ_open  →  SELL ADR (short)
Day D+1, Asia open (≈09:00 local)   →  recheck spread vs κ_close:
    if spread still > κ_close       →  BUY local  (pair fully open)
    if spread reversed overnight    →  BUY ADR cover (abort — no local leg ever placed)
```

**Close sequence**

```
Any day K: spread_K < κ_close  OR  days_held ≥ H (force-close)
    →  SELL local (Asia close bar)
    →  BUY ADR cover (same or next U.S. bar)
```

**Why no FX hedge**: The overnight gap makes simultaneous FX spot submission impossible. The paper accepts residual FX exposure. `fx_hedge_required = False` on all registry pairs.

### 2.4 Return Metrics

```
local_return  =  (P_local_close − P_local_open)  /  P_local_open
adr_return    =  (P_ADR_short   − P_ADR_cover)   /  P_ADR_short

ROCE  =  local_return + adr_return
RUCE  =  local_return + 2 × adr_return          # Reg-T 50% margin → 2× ADR component
```

RUCE is the economically realistic measure (investor posts 50% margin on short ADR leg).
ROCE is the conservative bound (full notional as denominator on both legs).

**Paper benchmarks (k0=2, kc=0, T=60, H=90):**

| Metric | Paper value |
|--------|-------------|
| Median ROCE | ~2.8% |
| Median RUCE | ~5.3% |
| Median duration | 3 days |
| IQ range of duration | 1–6 days |
| Trades per firm per year (median) | 11.6 |
| ADR leg contribution to returns | ~90% |
| Median net RUCE (after Roll costs) | ~2.7% |

### 2.5 Liquidity Bucketing

| Bucket | ADV criterion | Zero-return-day criterion | Typical median ROCE |
|--------|--------------|--------------------------|---------------------|
| High | > 91,332 shares | < 6.21% | ~2.0% |
| High-Medium | 17,222–91,332 | 6.21–14.57% | ~2.8% |
| Medium-Low | 3,531–17,222 | 14.57–29.75% | ~3.0% |
| Low | < 3,531 | > 29.75% | ~3.7% |

Higher illiquidity → higher median profit (limits-to-arbitrage premium). The `ZeroReturnDayFilter` risk rule blocks entries when the ADR zero-return-day percentage exceeds a configurable ceiling.

### 2.6 Transaction Cost Benchmark

Roll (1984) effective spread: `2 × √|autocov(returns)|`

Used as a post-hoc round-trip cost estimate per pair. Not used as an entry signal. The paper's median total Roll cost is ~2.67%, making net RUCE (~2.7%) modestly positive under the realistic RUCE measure.

---

## 3. Core Components

### 3.1 Historical Data Ingestion (datastream/)

Historical data ingestion and pair selection are handled entirely by standalone scripts in `datastream/`. There is no `src/asian_adr/data/` package — the scripts serve as both the initial data pipeline and the ongoing re-screening pipeline.

**Scripts**

| Script | Responsibility | Output |
|--------|---------------|--------|
| `fetch_datastream_adr_data.py` | WRDS query for U.S. ADR OHLCV + reference mapping | `data/parquet/adr/` |
| `fetch_datastream_global_data.py` | WRDS query for Asian underlying OHLCV | `data/parquet/global/` |
| `fetch_fx_history.py` | WRDS Datastream SPOT FX rates (inverted to USD/CCY) | `data/parquet/fx/` |
| `run_asian_adr_screening.py` | Full pair selection pipeline (cointegration, liquidity, Roll) | `config/pairs/asian_adr_pairs.json` |
| `rescreen.py` | Incremental re-screening orchestrator: gap-fetch + re-screen + changelog | `config/pairs/` |

**Parquet Cache Schemas**

| File | Columns | Notes |
|------|---------|-------|
| `adr/adr_prices.parquet` | `infocode`, `marketdate`, `close`, `high`, `low`, `open`, `volume`, `adj_factor`, `ticker`, `isin` | 4M+ rows |
| `adr/adr_reference.parquet` | `adr_ticker`, `adr_isin`, `adr_infocode`, `underlying_ticker`, `underlying_exchange`, `underlying_currency`, `adr_ratio` (NULL) | `adr_ratio` always NULL from WRDS; estimated from price median in screening |
| `global/global_prices.parquet` | `infocode`, `marketdate`, `close`, `high`, `low`, `open`, `volume`, `adj_factor`, `ticker`, `exchange`, `currency` | Filtered to Asian exchange mnemonics |
| `fx/fx_rates.parquet` | `date`, `base_currency`, `quote_currency`, `currency_pair`, `mid`, `provider` | `mid` = USD per 1 unit of base; inverted from Datastream CCY/USD |

**Key WRDS Queries**

```python
# U.S. ADR prices — fetch_datastream_adr_data.py
"""SELECT n.infocode, d.marketdate, d.close, d.high, d.low, d.open, d.volume,
          d.cumadjfactor AS adj_factor, n.dscode AS ticker, n.isin
   FROM tr_ds_equities.wrds_ds2dsf AS d
   JOIN tr_ds_equities.wrds_ds_names AS n ON d.infocode = n.infocode
   WHERE d.marketdate BETWEEN %(start_date)s AND %(end_date)s
     AND n.region = 'US' AND n.typecode = 'ADR'"""

# FX rates — fetch_fx_history.py (CCY/USD → inverted to USD/CCY on output)
"""SELECT c.fromcurrcode AS iso_currency, r.exratedate AS marketdate, r.midrate AS rate
   FROM trdstrm.ds2fxrate AS r
   JOIN trdstrm.ds2fxcode AS c ON r.exrateintcode = c.exrateintcode
   WHERE c.ratetypecode = 'SPOT' AND c.tocurrcode = 'USD'
     AND r.exratedate BETWEEN %(start_date)s AND %(end_date)s"""
```

**Failure Handling**
- WRDS timeout → exponential backoff reconnect (1s, 2s, 4s, max 60s)
- ADR ratio unavailable from reference → estimated from `median(P_local × FX / P_ADR)`, snapped to nearest standard ratio (0.01, 0.1, 0.5, 1, 2, 5, 10 …); pairs with unresolvable ratio are rejected
- Missing adjusted price → fallback to raw price with `adj_factor = 1.0`

---

### 3.2 Market Data Handler (Live Daily Feeds)

**Responsibilities**
- Connect to Polygon.io or Alpaca for real-time U.S. ADR end-of-day bars
- Connect to Asian market data provider for foreign underlying daily bars
- Subscribe only to tickers in the active pair registry
- Normalise incoming data into canonical `BarEvent`
- Reconnect with exponential backoff on disconnection
- Detect and flag stale bars (e.g., repeated closing price = zero-return day)

**Internal Modules**

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

**Failure Handling**
- WebSocket disconnect → exponential backoff reconnect
- Missing bar → skip; flag gap in metrics; do not compute spread
- Stale bar (same price as prior day) → increment zero-return counter; publish `ZeroReturnEvent`

---

### 3.3 FX Rate Feed (USD Conversion)

**Responsibilities**
- Fetch daily FX close rates for all currencies in the active pair registry
- Publish `FXRateEvent` to the event bus at end-of-day (or on-demand in backtest)
- Maintain an in-memory FX rate cache for low-latency lookups by the Hong & Susmel Engine
- Note: FX is used **only for spread computation** — no FX hedging occurs

**Internal Modules**

```
fx_handler/
├── rate_cache.py             # In-memory cache: {currency_pair → FXRateEvent}
├── normalizer.py             # Provider ticks → FXRateEvent
├── staleness_monitor.py      # Alert if daily rate not received by bar close
└── connectors/
    └── oanda.py              # OANDA REST v20: daily FX close rates
```

**Failure Handling**
- FX rate missing for today → use prior day's rate with `is_stale=True` flag
- Stale FX → `HongSusmelEngine` skips spread computation for affected pairs; logs warning
- FX age > 2 business days → pair marked SUSPENDED; alert operator

---

### 3.4 Asian ADR Pair Selection (datastream/run_asian_adr_screening.py)

Pair selection is implemented in `datastream/run_asian_adr_screening.py` — a single self-contained script. There is no `src/asian_adr/research/` package. For weekly automated re-screening, `datastream/rescreen.py` orchestrates the full pipeline (incremental WRDS fetch → screening → changelog → registry backup).

**Eligible Asian Exchanges**

```python
ASIAN_EXCHANGES: set[str] = {
    "TKS", "TSE",   # Tokyo Stock Exchange (Japan)
    "HKG",          # Hong Kong Exchange (HKEX)
    "KRX",          # Korea Exchange
    "BOM", "BSE",   # Bombay Stock Exchange (India)
    "NSE",          # National Stock Exchange (India)
    "ASX",          # Australian Securities Exchange
    "SES",          # Singapore Exchange (SGX)
    "TAI",          # Taiwan Stock Exchange (TWSE)
    "IDX",          # Indonesia Stock Exchange
    "PHS",          # Philippine Stock Exchange (PSE)
    "KLS",          # Bursa Malaysia
}
```

**Pair Eligibility Criteria**

| Criterion | Threshold |
|-----------|-----------|
| Cointegration (ADF + PP on dollar spread) | Reject unit root at 5% |
| Min continuous trading | 2 years (~504 trading days) |
| Min non-zero return days (both legs) | ≥ 50% of trading days |
| Zero-return-day % (ADR, entry ceiling) | ≤ 50% |
| ADR ratio resolvable | Required; ratio > 0 |

**Screening Steps (run_pipeline)**

```
1. Filter adr_reference to ASIAN_EXCHANGES
2. For each candidate pair:
   a. Resolve adr_ratio — estimate from median(P_local × FX / P_ADR),
      snap to nearest standard ratio; reject if unresolvable
   b. Reconstruct dollar spread: P_ADR − (P_local × FX) / ratio
   c. Trim to as_of date (no look-ahead)
   d. Apply liquidity filter: non_zero_return_pct ≥ 0.50 on both legs
   e. ADF + Phillips-Perron cointegration test (p < 0.05 on both)
   f. Compute Roll (1984) effective spread: 2 × √|autocov(returns)|
3. Write approved pairs to config/pairs/asian_adr_pairs.json
4. Write per-candidate diagnostics to config/pairs/screening_diagnostics.json
```

**Roll (1984) Effective Spread**

```python
def roll_effective_spread(prices: pd.Series) -> float:
    rets = prices.pct_change().dropna().values
    autocov = float(np.cov(rets[:-1], rets[1:], ddof=1)[0, 1])
    return float(2.0 * np.sqrt(abs(autocov)))
```

---

### 3.5 Hong & Susmel Engine (Signal Generation)

**Responsibilities**
- Subscribe to `BarEvent` for each ADR and its Asian underlying (daily bars)
- Subscribe to `FXRateEvent` for USD conversion of local price
- Maintain per-pair `WelfordRollingStats(ddof=1)` over estimation window `T`
- Compute dollar spread on every daily bar update when both legs are fresh
- Emit `HongSusmelSignalEvent(signal=SHORT_ADR)` when `spread_t > κ_open`
- Emit `HongSusmelSignalEvent(signal=EXIT)` when `spread_t < κ_close`
- Emit `HongSusmelSignalEvent(signal=FORCE_CLOSE)` when holding days ≥ H
- Enforce stale-leg guard: both legs must carry the same bar date before computing spread
- Direction is permanently one-sided: `SHORT_ADR` only

**Engine Implementation**

```python
class HongSusmelEngine:
    """
    Per-pair state machine.
    β = pair.adr_ratio (structural constant; never OLS-estimated).
    """

    def on_daily_bar(self, event: BarEvent) -> list[HongSusmelSignalEvent]:
        signals = []
        for pair in self._pairs_for_ticker(event.ticker):
            state = self._states[pair.pair_id]
            state.update_price(event.ticker, event.close, event.timestamp_exchange.date())

            if not state.both_legs_fresh():        # stale-leg guard
                continue

            fx_rate = self._fx_cache.get_usd_rate(pair.underlying_currency)
            if fx_rate is None:
                continue

            local_usd = state.underlying_price * fx_rate
            spread    = state.adr_price - (local_usd / pair.adr_ratio)

            state.rolling_stats.update(spread)
            if state.rolling_stats.count < pair.estimation_days:
                continue                           # warm-up period

            mu, sigma   = state.rolling_stats.mean, state.rolling_stats.std
            kappa_open  = mu + pair.k0 * sigma
            kappa_close = mu + pair.kc * sigma

            if state.position == HSPosition.FLAT:
                if spread > kappa_open:
                    signals.append(
                        self._build_signal(pair, HSSignal.SHORT_ADR, spread, mu, sigma)
                    )
                    state.open_position(entry_date=event.timestamp_exchange.date())

            elif state.position == HSPosition.OPEN:
                days_held = (event.timestamp_exchange.date() - state.entry_date).days
                if spread < kappa_close:
                    signals.append(self._build_signal(pair, HSSignal.EXIT, spread, mu, sigma))
                    state.close_position()
                elif days_held >= pair.holding_days:
                    signals.append(
                        self._build_signal(pair, HSSignal.FORCE_CLOSE, spread, mu, sigma)
                    )
                    state.close_position()

        return signals
```

**Internal Modules**

```
strategy/hong_susmel/
├── engine.py               # HongSusmelEngine: spread computation, signal emission
├── state.py                # HSPairState: prices, rolling stats, holding-period counter
├── signal_factory.py       # Constructs HongSusmelSignalEvent
└── liquidity_bucket.py     # Assigns High/High-Med/Med-Low/Low bucket for attribution
```

---

### 3.6 Event Bus / Messaging Layer

**Responsibilities**
- Decouple all components via pub/sub
- Guarantee ordered delivery within a topic
- Support both in-process (`MemoryBus`) and out-of-process (Kafka) modes
- Provide backpressure signalling to publishers

**Interface**

```python
class AbstractEventBus(Protocol):
    async def publish(self, topic: str, event: BaseEvent) -> None: ...
    async def subscribe(self, topic: str, handler: Callable) -> None: ...
    async def subscribe_many(self, topics: list[str], handler: Callable) -> None: ...
```

**Topic Design**

| Topic | Producers | Consumers |
|-------|-----------|-----------|
| `market-data` | Market Data Handler | H&S Engine, Position Engine |
| `fx-rates` | FX Rate Feed | H&S Engine, Position Engine |
| `signals` | H&S Engine | Risk Engine |
| `risk-decisions` | Risk Engine | Asian Execution Sequencer |
| `orders` | Asian Execution Sequencer | Broker Gateway, Monitoring |
| `fills` | Broker Gateway | Sequencer, Position Engine, ROCE/RUCE Calculator |
| `positions` | Position Engine | Risk Engine, Dashboard |
| `pair-registry` | Research Engine | H&S Engine, Risk Engine |
| `alerts` | Risk, Feed, Sequencer | Monitoring, Dashboard |

**Implementations**

```
event_bus/
├── base.py             # Protocol / interface definition
├── memory_bus.py       # Synchronous in-memory bus (backtest + unit tests)
├── asyncio_bus.py      # Single-process asyncio.Queue (live prototype)
└── kafka_bus.py        # Production Kafka implementation
```

---

### 3.7 Risk Management Engine

**Responsibilities**
- Validate every `HongSusmelSignalEvent` against pre-trade risk rules
- Enforce position limits, drawdown limits, and holding-period force-close
- Block entries when ADR zero-return-day percentage exceeds threshold
- Detect FX conversion data staleness; suspend affected pairs
- Trigger kill switch on breach of critical thresholds

**Risk Rules**

```
risk/rules/
├── base.py
├── notional_limits.py           # Max per-leg and portfolio notional
├── drawdown_limits.py           # Daily and peak-to-trough drawdown
├── rate_of_loss.py              # Max dollar loss per N bars
├── holding_period_force_close.py# Force-close at H days
├── zero_return_day_filter.py    # Block entry if ADR zero-return pct > threshold
├── overnight_abort_cover.py     # Cover naked ADR short if local leg never filled
├── short_locate.py              # Verify ADR short-locate before SELL
├── country_concentration.py     # Max exposure per country
└── kill_switch.py
```

**Holding Period Force-Close Rule**

```python
class HoldingPeriodForceCloseRule(AbstractRiskRule):
    """
    Triggers FORCE_CLOSE if a pair has been held longer than holding_days.
    Hard stop independent of spread level.
    """
    def evaluate(self, state: HSPairState, pair: AsianADRApprovedPair) -> bool:
        days_held = (self._clock.date() - state.entry_date).days
        return days_held >= pair.holding_days
```

**Zero Return Day Filter**

```python
class ZeroReturnDayFilter(AbstractRiskRule):
    """
    Blocks new entries for pairs whose ADR zero-return-day percentage over
    the estimation window exceeds the configured ceiling.
    Prevents stale-price false signals from illiquid ADRs.
    """
    def evaluate(
        self, signal: HongSusmelSignalEvent, state: HSPairState, config: RiskConfig
    ) -> RiskRuleResult:
        zero_pct = state.rolling_zero_return_pct()
        if zero_pct > config.max_zero_return_pct:
            return RiskRuleResult(
                passed=False,
                reason=f"ADR zero-return {zero_pct:.1%} exceeds limit {config.max_zero_return_pct:.1%}",
                severity=Severity.BLOCK,
            )
        return RiskRuleResult(passed=True)
```

**Overnight Abort Cover Rule**

```python
class OvernightAbortCoverRule(AbstractRiskRule):
    """
    If the ADR leg filled (short) but the spread reversed overnight and no
    local leg was placed, automatically covers the naked ADR short.
    Prevents one-sided inventory from overnight gap reversals.
    """
```

**Pre-Trade Limits**

| Limit | Default |
|-------|---------|
| Max simultaneous open pairs | 20 |
| Max notional per leg | $100,000 |
| Max zero-return-day % (entry block) | 50% |
| Max holding period (force-close) | 90 days |
| Max daily loss | 2% of AUM |
| Max drawdown kill switch | 5% of AUM |
| ADR short-locate required | Yes |
| Rate of loss | $25,000 / 30 bars |
| Max country concentration | 30% of gross notional |

---

### 3.8 Position / PnL Engine

**Responsibilities**
- Track open quantity, average-cost basis, and unrealised P&L for every ADR and local leg
- Mark positions to market on each new daily bar
- Convert local-leg P&L to USD using current FX rate
- Publish `PositionUpdateEvent` on every change
- Detect and alert on leg imbalances (ADR short open but no matching local long)

**Mark-to-Market (multi-currency)**

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

---

### 3.9 Asian Execution Sequencer (OMS)

**Responsibilities**
- Bridges the overnight gap between U.S. close and Asian market open
- Per-pair state machine: short ADR first, then conditionally buy local next bar
- Rechecks spread at Asia open before committing the local leg
- Aborts and covers the naked ADR short if spread reversed overnight
- Handles simultaneous close (sell local + cover ADR) on EXIT / FORCE_CLOSE signals
- Routes orders to the U.S. Broker Gateway via standard `OrderRequest`

**Why not simultaneous submission**: The overnight gap makes `asyncio.gather`-style simultaneous leg submission impossible for Asian pairs. Forcing both legs into one bar produces systematically optimistic backtest fills.

**State Machine**

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

**Implementation**

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

**Internal Modules**

```
strategy/hong_susmel/
├── execution_sequencer.py    # AsianExecutionSequencer state machine
└── sequencer_state.py        # SeqPhase enum, per-pair mutable state
```

---

### 3.10 Broker Gateway (U.S. Equity Only)

**Responsibilities**
- Translate internal `OrderRequest` into broker-specific API calls for U.S. ADR trades only
- Handle ADR short-leg (SELL) and cover-leg (BUY) via Alpaca or Interactive Brokers
- Map broker order IDs to internal order IDs
- Receive fills; publish `FillEvent` to the event bus
- Verify short-locate availability before submitting any SELL (short)

**Supported Gateways**

| Gateway | Protocol | Venue | Notes |
|---------|----------|-------|-------|
| Interactive Brokers (US) | TWS API / FIX | U.S. ADR equities | Full institutional feature set; primary broker |
| Alpaca | REST / WebSocket | U.S. ADR equities | Commission-free; paper trading fallback |
| Simulation | In-process | U.S. equities | Backtest and paper trading |

> **No foreign equity gateway.** The local (Asian) leg in live trading is executed via the operator's own Asian brokerage account. The platform generates the order instruction (`BUY_LOCAL`) and logs it; actual local execution is external. In backtest mode the `SimulatedForeignExchange` handles local fills.

**Internal Modules**

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

---

### 3.11 ROCE / RUCE Calculator

**Responsibilities**
- Subscribe to closed round-trip fill pairs (ADR fill + local fill)
- Compute ROCE and RUCE per trade as per §2.4
- Tag each trade with a `LiquidityBucket` for post-hoc attribution
- Accumulate per-pair and aggregate distribution statistics matching paper Table 7-B format
- Integrated into the backtest tearsheet

```python
class RoceRuceCalculator:
    def on_round_trip_closed(
        self,
        adr_short_fill:  FillEvent,
        adr_cover_fill:  FillEvent,
        local_buy_fill:  FillEvent,
        local_sell_fill: FillEvent,
    ) -> RoceRuceResult:
        p_adr_open  = adr_short_fill.fill_price
        p_adr_close = adr_cover_fill.fill_price
        p_loc_open  = local_buy_fill.fill_price_usd
        p_loc_close = local_sell_fill.fill_price_usd

        local_return = (p_loc_close - p_loc_open) / p_loc_open
        adr_return   = (p_adr_open  - p_adr_close) / p_adr_open   # profit when ADR falls

        roce = local_return + adr_return
        ruce = local_return + Decimal("2") * adr_return            # 0.5× denom = 2× return

        duration_days = (
            local_sell_fill.timestamp_exchange.date()
            - local_buy_fill.timestamp_exchange.date()
        ).days

        return RoceRuceResult(
            pair_id=adr_short_fill.pair_id,
            roce=roce,
            ruce=ruce,
            local_return=local_return,
            adr_return=adr_return,
            duration_days=duration_days,
            liquidity_bucket=self._assign_bucket(adr_short_fill.pair_id),
            was_force_closed=adr_cover_fill.metadata.get("reason") == "force_close",
            was_aborted=False,
        )
```

**Distribution statistics computed per run** (matching paper Table 7-B):
- Per-pair and aggregate: mean, std, max, p90, p75, median, p25, p10, min
- Trades per firm per year
- Median duration (days) and IQ range
- Roll effective spread (transaction cost estimate)
- Liquidity bucket breakdown

---

### 3.12 Backtesting / Replay Engine

**Responsibilities**
- Load U.S. ADR, Asian underlying, and FX daily prices from Parquet cache
- Replay events in chronological order via merged min-heap stream
- Enforce point-in-time pair registry (no look-ahead on pair selection dates)
- Simulate U.S. equity fills with next-bar execution (models overnight gap)
- Simulate Asian equity fills with next-bar execution at local open price
- Apply realistic cost model: commission, SEC fee, short borrow, local levy
- Produce tearsheet with ROCE/RUCE distributions, Sharpe, max drawdown, per-pair attribution

**Multi-Feed Replay Loop**

```python
class BacktestEngine:
    async def run(self, start: date, end: date):
        bus   = MemoryBus()
        clock = SimulatedClock(start)

        pair_registry  = self.pair_registry_loader.load_as_of(start)
        hs_engine      = HongSusmelEngine(bus, clock, pair_registry)
        risk           = RiskEngine(bus, clock, self.risk_config)
        position       = PositionEngine(bus, clock)
        sequencer      = AsianExecutionSequencer(bus, clock)
        us_exchange    = SimulatedUSExchange(bus, clock, self.us_cost_model)
        foreign_exch   = SimulatedForeignExchange(bus, clock, self.foreign_cost_model)
        roce_ruce      = RoceRuceCalculator(bus, clock, pair_registry)

        async for event in self._merge_streams(
            self.us_loader.stream(start, end),
            self.foreign_loader.stream(start, end),
            self.fx_loader.stream(start, end),
        ):
            clock.advance_to(event.timestamp_exchange)
            if clock.date_changed:
                new_registry = self.pair_registry_loader.load_as_of(clock.date)
                await bus.publish("pair-registry", PairRegistryUpdateEvent(registry=new_registry))
            topic = self._route_event_topic(event)
            await bus.publish(topic, event)
            await bus.flush()
```

**Cost Model**

```python
class EquitiesCostModel:
    """
    U.S. ADR leg:
        commission   = max($1.00, $0.005 × shares)
        sec_fee      = notional × 0.0000278  (sell only)
        short_borrow = notional × borrow_rate / 252

    Asian underlying leg:
        commission   = max(min_commission, rate_per_share × shares)
        stamp_duty   = notional × stamp_rate  (e.g., HK 0.10%, AU 0.00%)
        local_levy   = exchange-specific transaction levy
    """
```

**Internal Modules**

```
backtest/
├── engine.py                     # Main replay loop (multi-feed: US, foreign, FX)
├── clock.py                      # SimulatedClock
├── data_loader.py                # U.S. ADR Datastream Parquet streaming
├── foreign_data_loader.py        # Datastream Global Parquet streaming
├── fx_data_loader.py             # Datastream FX history streaming (trdstrm.ds2fxrate)
├── pair_registry_loader.py       # Point-in-time registry snapshots
├── simulated_us_exchange.py      # Virtual U.S. order book
├── simulated_foreign_exchange.py # Virtual Asian order book (backtest only)
├── slippage_models.py            # Half-spread, sqrt-impact
├── cost_model.py                 # Commission + SEC + borrow + stamp duty + local levy
├── roce_ruce_calculator.py       # ROCE/RUCE per trade; aggregate distributions
└── report.py                     # HTML tearsheet: paper-benchmarked distributions
```

---

### 3.13 Monitoring / Logging / Alerting

**Responsibilities**
- Collect structured logs from all components
- Expose Prometheus metrics (spread z-scores, duration distribution, ROCE/RUCE live, zero-return day rates)
- Send alerts via Slack/PagerDuty on critical events (overnight abort, force-close cluster, large drawdown)
- Provide OpenTelemetry traces for cross-component request tracking

**Internal Modules**

```
monitoring/
├── logger.py     # Structured JSON logger (structlog)
├── metrics.py    # Prometheus metric definitions
├── tracing.py    # OpenTelemetry tracer setup
├── alerting.py   # Alert rule engine and dispatcher
└── health.py     # Health check endpoints
```

---

## 4. Architecture Principles

| Principle | Application |
|-----------|-------------|
| **Point-in-time correctness** | Pair selection never uses future data; ADR ratio history enforced as-of selection date |
| **Research / production parity** | Same H&S Engine, Risk Engine, and Position Engine in research, backtest, and live |
| **β = ratio, never estimated** | ADR conversion ratio is legally fixed; OLS estimation introduces spurious time-varying hedges inconsistent with the law-of-one-price framework |
| **Dollar spread, not log-price** | Hong & Susmel use raw dollar spread; log-spread distorts threshold geometry for high-ratio pairs |
| **One-sided entry by design** | Asian short-selling restrictions (Indonesia, Taiwan, China, India, Korea) eliminate the symmetric trade; `HSSignal` enum has no `LONG_ADR` variant |
| **No FX hedge** | Overnight gap makes simultaneous FX spot submission impossible; FX is conversion-only |
| **Event immutability** | All events are frozen Pydantic models; mutable state lives only in engines |
| **Graceful degradation** | FX data stale → skip spread computation; bar gap → skip signal; zero-return excess → block entry; never crash |
| **Next-bar fill discipline** | Simulated broker always fills on the bar after order submission; models the overnight gap correctly |
| **RUCE as live tracking metric** | RUCE (0.9% / day net) is the economically realistic measure; ROCE is reported for comparison |

---

## 5. Technology Stack

### Core Language
- **Python 3.12+**: asyncio, type hints, `match` statements, Polars / pandas for research

### Data Access

| Need | Technology | Notes |
|------|-----------|-------|
| U.S. ADR OHLCV | **WRDS Datastream (tr_ds_equities.wrds_ds2dsf)** | `typecode='ADR'`, `region='US'` filters |
| Asian underlying prices | **Datastream (tr_ds_equities.wrds_ds2dsf)** | Historical Asian equity OHLCV |
| ADR reference data | **Datastream (tr_ds_equities.wrds_ds_names, typecode='ADR')** | Underlying mapping via `dscompcode` join; ADR ratio sourced separately |
| Historical FX rates (daily) | **WRDS Datastream (trdstrm.ds2fxrate / trdstrm.ds2fxcode)** | SPOT rates (CCY per 1 USD) inverted to USD per CCY on output |
| Live U.S. ADR data | **Polygon.io / Alpaca** | WebSocket; ADR tickers; daily bars |
| Live Asian data | **Asian Market Feed** | Provider TBD per exchange |

### Messaging

| Option | Use Case | Notes |
|--------|----------|-------|
| **MemoryBus** | Backtest + unit tests | Synchronous; deterministic |
| asyncio.Queue | Live single-process prototype | Phase 1–2 only |
| **Kafka** | Production event bus | Best durability + replay |

### Databases

| Need | Technology | Notes |
|------|-----------|-------|
| Time-series OHLCV + FX | **TimescaleDB** | Hypertables, fast range scans |
| Pair registry / orders | **PostgreSQL 16** | ADR pair metadata, order/fill history |
| Hot state / cache | **Redis 7** | Latest spreads, z-scores |
| Historical data store | **MinIO** (local) / **S3** (cloud) | Datastream + FX Parquets |

### Quantitative Libraries

| Purpose | Library |
|---------|---------|
| ADF / PP cointegration tests | `statsmodels` |
| Rolling stats | `WelfordRollingStats` (internal) |
| DataFrame processing | `polars` (fast) / `pandas` (compat.) |
| Backtest tearsheets | `quantstats` |

---

## 6. Project Structure

```
asian-adr-strategy/
│
├── pyproject.toml
├── uv.lock
├── .env.example
├── Makefile
│
├── config/
│   ├── base.toml
│   ├── development.toml
│   ├── production.toml
│   └── pairs/
│       ├── asian_adr_pairs.json        # AsianADRApprovedPair registry
│       └── pair_registry.toml
│
├── src/
│   └── asian_adr/
│       │
│       ├── core/
│       │   ├── clock.py
│       │   ├── events.py               # All event dataclasses
│       │   ├── instruments.py          # AsianADRPairSpec
│       │   ├── types.py                # PairId, Ticker, CurrencyCode, ADRRatio
│       │   └── exceptions.py
│       │
│       ├── event_bus/
│       │   ├── base.py
│       │   ├── memory_bus.py
│       │   ├── asyncio_bus.py
│       │   └── kafka_bus.py
│       │
│       ├── feed_handler/
│       │   ├── connector.py
│       │   ├── normalizer.py
│       │   ├── staleness_monitor.py
│       │   └── connectors/
│       │       ├── polygon.py          # U.S. ADR live feed
│       │       ├── alpaca_data.py      # U.S. ADR alternative
│       │       └── asian_feed.py       # Asian underlying feed
│       │
│       ├── fx_handler/
│       │   ├── rate_cache.py
│       │   ├── normalizer.py
│       │   ├── staleness_monitor.py
│       │   └── connectors/
│       │       └── oanda.py            # OANDA daily FX rates
│       │
│       ├── strategy/
│       │   └── hong_susmel/
│       │       ├── engine.py           # HongSusmelEngine
│       │       ├── state.py            # HSPairState, WelfordRollingStats
│       │       ├── signal_factory.py
│       │       ├── execution_sequencer.py  # AsianExecutionSequencer
│       │       ├── sequencer_state.py
│       │       └── liquidity_bucket.py
│       │
│       ├── risk/
│       │   ├── engine.py
│       │   ├── state.py
│       │   └── rules/
│       │       ├── base.py
│       │       ├── notional_limits.py
│       │       ├── drawdown_limits.py
│       │       ├── rate_of_loss.py
│       │       ├── holding_period_force_close.py
│       │       ├── zero_return_day_filter.py
│       │       ├── overnight_abort_cover.py
│       │       ├── short_locate.py
│       │       ├── country_concentration.py
│       │       └── kill_switch.py
│       │
│       ├── position/
│       │   ├── engine.py
│       │   └── pnl_calculator.py
│       │
│       ├── gateways/
│       │   ├── base.py
│       │   ├── alpaca/
│       │   │   ├── rest_gateway.py
│       │   │   └── ws_gateway.py
│       │   ├── interactive_brokers/
│       │   │   ├── tws_gateway.py      # U.S. ADR equity only
│       │   │   └── short_locate.py
│       │   └── simulation/
│       │       ├── simulated_us_gateway.py
│       │       └── simulated_foreign_gateway.py
│       │
│       ├── backtest/
│       │   ├── engine.py
│       │   ├── clock.py
│       │   ├── data_loader.py
│       │   ├── foreign_data_loader.py
│       │   ├── fx_data_loader.py
│       │   ├── pair_registry_loader.py
│       │   ├── simulated_us_exchange.py
│       │   ├── simulated_foreign_exchange.py
│       │   ├── slippage_models.py
│       │   ├── cost_model.py
│       │   ├── roce_ruce_calculator.py
│       │   └── report.py
│       │
│       ├── persistence/
│       │   ├── timescale.py
│       │   ├── postgres.py
│       │   ├── redis_cache.py
│       │   └── s3_store.py
│       │
│       ├── monitoring/
│       │   ├── logger.py
│       │   ├── metrics.py
│       │   ├── tracing.py
│       │   └── alerting.py
│       │
│       └── runners/
│           ├── live_runner.py
│           ├── backtest_runner.py
│           └── data_recorder.py
│
├── tests/
│   ├── unit/
│   │   ├── test_hs_engine.py
│   │   ├── test_execution_sequencer.py
│   │   ├── test_roce_ruce.py
│   │   ├── test_risk_rules.py
│   │   └── test_rolling_stats.py
│   ├── integration/
│   │   ├── test_hs_signal_flow.py
│   │   ├── test_fx_handler_hs_engine.py
│   │   └── test_sequencer_backtest_fills.py
│   ├── system/
│   │   └── test_asian_adr_backtest_e2e.py
│   └── conftest.py
│
├── datastream/                            # All offline data + research scripts (no src/ dependency)
│   ├── data/
│   │   ├── parquet/                       # Local Parquet cache (dev); S3/MinIO in production
│   │   │   ├── adr/
│   │   │   │   ├── adr_prices.parquet     # infocode, marketdate, OHLCV, adj_factor, ticker, isin
│   │   │   │   └── adr_reference.parquet  # adr_ticker, underlying_ticker, exchange, currency, adr_ratio (NULL)
│   │   │   ├── global/
│   │   │   │   └── global_prices.parquet  # infocode, marketdate, OHLCV, adj_factor, ticker, exchange, currency
│   │   │   └── fx/
│   │   │       └── fx_rates.parquet       # date, base_currency, quote_currency, mid (USD/CCY), provider
│   │   └── backtest/                      # Backtest run outputs
│   │       └── run_YYYYMMDD_HHMMSS/
│   │           ├── trades.parquet
│   │           ├── summary.json
│   │           ├── distribution.json
│   │           └── tearsheet.html
│   ├── config/
│   │   └── pairs/
│   │       ├── asian_adr_pairs.json       # Active approved pair registry
│   │       ├── screening_diagnostics.json # Per-candidate accept/reject log
│   │       ├── backups/                   # Timestamped registry snapshots (rescreen.py)
│   │       └── changelogs/               # Per-run added/dropped/changed diffs (rescreen.py)
│   ├── fetch_datastream_adr_data.py       # WRDS: U.S. ADR OHLCV + reference → adr/
│   ├── fetch_datastream_global_data.py    # WRDS: Asian underlying OHLCV → global/
│   ├── fetch_fx_history.py               # WRDS: SPOT FX rates (inverted USD/CCY) → fx/
│   ├── run_asian_adr_screening.py         # Full pair selection pipeline → asian_adr_pairs.json
│   ├── rescreen.py                        # Incremental re-screening orchestrator (Phase 7)
│   ├── run_backtest.py                    # End-to-end backtest → data/backtest/
│   ├── backtest_report.py                 # HTML tearsheet from backtest output
│   └── healthcheck.py
│
├── notebooks/
│   ├── adr_spread_analysis.ipynb
│   ├── cointegration_validation.ipynb
│   ├── roll_spread_calibration.ipynb
│   └── backtest_tearsheet.ipynb
│
├── docker/
│   ├── Dockerfile.trading
│   ├── Dockerfile.research
│   └── docker-compose.yml
│
└── docs/
    ├── architecture.md
    ├── strategy_methodology.md
    ├── runbooks/
    └── adr/
```

**Key structural rules:**
- `core/` has zero external dependencies — only stdlib + pydantic
- `datastream/` scripts are standalone — they never import from `src/asian_adr/`
- `strategy/` never imports from `risk/`, `position/`, or `gateways/` directly
- `notebooks/` never imported by `src/` (enforced by ruff rule)

---

## 7. Internal APIs and Data Contracts

All events inherit from `BaseEvent` and are immutable frozen Pydantic models.

### 7.1 Base Event

```python
class BaseEvent(BaseModel):
    model_config = {"frozen": True}

    event_id:             UUID     = Field(default_factory=uuid4)
    event_type:           str
    timestamp_exchange:   datetime
    timestamp_received:   datetime
    timestamp_processed:  datetime | None = None
```

### 7.2 Market Events

```python
class BarEvent(BaseEvent):
    event_type:   Literal["bar"] = "bar"
    ticker:       str
    open:         Decimal
    high:         Decimal
    low:          Decimal
    close:        Decimal
    volume:       int
    bar_interval: str            # "1d"
    is_adjusted:  bool
    exchange:     str            # "NYSE", "TSE", "HKEX", "KRX", etc.
    currency:     str            # "USD", "JPY", "HKD", "KRW", etc.
```

### 7.3 FX Rate Event

```python
class FXRateEvent(BaseEvent):
    event_type:     Literal["fx_rate"] = "fx_rate"
    currency_pair:  str       # e.g., "JPY_USD"
    base_currency:  str       # "JPY"
    quote_currency: str       # "USD"
    mid:            Decimal   # USD per 1 unit of base currency
    provider:       str       # "datastream" (historical cache) | "oanda" (live feed)
    is_stale:       bool = False
```

### 7.4 Hong & Susmel Signal Event

```python
class HSSignal(str, Enum):
    SHORT_ADR   = "short_adr"    # SELL ADR short; buy local next Asia open
    EXIT        = "exit"          # Spread converged; close both legs
    FORCE_CLOSE = "force_close"   # Holding period expired

class HongSusmelSignalEvent(BaseEvent):
    event_type:         Literal["hs_signal"] = "hs_signal"
    pair_id:            str
    adr_ticker:         str
    underlying_ticker:  str
    signal:             HSSignal
    spread:             Decimal     # dollar spread at signal bar
    mu:                 Decimal     # rolling mean
    sigma:              Decimal     # rolling std (ddof=1)
    kappa_open:         Decimal     # µ + k0·σ
    kappa_close:        Decimal     # µ + kc·σ
    z_score:            Decimal     # (spread − µ) / σ
    days_held:          int         # 0 for SHORT_ADR signals
```

### 7.5 Risk Decision

```python
class RiskDecisionStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"

class RiskDecision(BaseEvent):
    event_type:       Literal["risk_decision"] = "risk_decision"
    signal_id:        UUID
    pair_id:          str
    status:           RiskDecisionStatus
    approved_notional: Decimal | None
    rejected_reason:   str | None
    risk_rule_results: list[dict]
```

### 7.6 Order Request

```python
class OrderRequest(BaseEvent):
    event_type:      Literal["order_request"] = "order_request"
    pair_id:         str
    risk_decision_id: UUID
    ticker:          str
    side:            Literal["buy", "sell"]
    quantity:        Decimal
    order_type:      OrderType
    limit_price:     Decimal | None
    venue:           Literal["us_equity", "foreign_equity"]
    currency:        str
    is_short_sale:   bool = False
    time_in_force:   str = "DAY"
```

### 7.7 Fill Event

```python
class FillEvent(BaseEvent):
    event_type:         Literal["fill"] = "fill"
    fill_id:            str
    client_order_id:    UUID
    broker_order_id:    str
    pair_id:            str | None
    ticker:             str
    side:               Literal["buy", "sell"]
    fill_price:         Decimal         # in native currency
    fill_price_usd:     Decimal         # USD-converted
    fill_quantity:      Decimal
    remaining_quantity: Decimal
    commission:         Decimal
    commission_currency: str
    sec_fee:            Decimal
    stamp_duty:         Decimal
    short_borrow_fee:   Decimal
    fx_rate_used:       Decimal | None
    is_short_sale:      bool
    venue:              Literal["us_equity", "foreign_equity"]
    exchange:           str
    metadata:           dict = {}
```

### 7.8 Position Update

```python
class PositionUpdateEvent(BaseEvent):
    event_type:            Literal["position_update"] = "position_update"
    pair_id:               str
    ticker:                str
    venue:                 Literal["us_equity", "foreign_equity"]
    net_quantity:          Decimal
    average_entry_price:   Decimal
    average_entry_price_usd: Decimal
    unrealized_pnl_usd:    Decimal
    realized_pnl_usd:      Decimal
    mark_price:            Decimal
    mark_price_usd:        Decimal
    notional_value_usd:    Decimal
    fx_rate:               Decimal | None
    is_short:              bool
    trigger:               Literal["fill", "mark_to_market", "fx_update", "reconciliation"]
```

### 7.9 Asian ADR Approved Pair (Registry)

```python
class AsianADRApprovedPair(BaseModel):
    model_config = {"frozen": True}

    pair_id:              str
    adr_ticker:           str
    underlying_ticker:    str
    underlying_exchange:  str           # "TSE", "HKEX", "KRX", "ASX", etc.
    underlying_currency:  str           # "JPY", "HKD", "KRW", "AUD", etc.
    adr_ratio:            Decimal       # Local shares per 1 ADR; fixed structural constant
    estimation_days:      int = 60      # T: rolling window for µ/σ
    holding_days:         int = 90      # H: max holding period before force-close
    k0:                   Decimal = Decimal("2.0")
    kc:                   Decimal = Decimal("0.0")
    zero_return_pct_adr:  Decimal       # liquidity metric (Bekaert et al. 2007)
    roll_spread_local:    Decimal       # Roll (1984) effective spread, local leg
    roll_spread_adr:      Decimal       # Roll (1984) effective spread, ADR leg
    fx_hedge_required:    bool = False
    withholding_tax_rate: Decimal = Decimal("0.0")
    approved_date:        date
    expiry_date:          date
    is_active:            bool = True
```

### 7.10 ROCE / RUCE Result

```python
class LiquidityBucket(str, Enum):
    HIGH        = "high"
    HIGH_MEDIUM = "high_medium"
    MEDIUM_LOW  = "medium_low"
    LOW         = "low"

class RoceRuceResult(BaseModel):
    model_config = {"frozen": True}

    pair_id:          str
    trade_open_date:  date
    trade_close_date: date
    duration_days:    int
    local_return:     Decimal
    adr_return:       Decimal
    roce:             Decimal
    ruce:             Decimal
    roll_cost_pct:    Decimal     # estimated round-trip cost (Roll 1984)
    roce_net:         Decimal     # roce − roll_cost_pct
    ruce_net:         Decimal     # ruce − roll_cost_pct
    liquidity_bucket: LiquidityBucket
    was_force_closed: bool
    was_aborted:      bool        # True if local leg never established (overnight reversal)
```

---

## 8. Concurrency and Runtime Model

### 8.1 Single-Process asyncio Architecture (Phase 1–2)

```python
async def main():
    bus   = AsyncioBus()
    clock = LiveClock()

    pair_registry = await AsianADRRegistry.load_active(db_url=config.database_url)

    all_adr_tickers        = pair_registry.all_adr_tickers()
    all_underlying_tickers = pair_registry.all_underlying_tickers()
    all_currencies         = pair_registry.all_currencies()

    us_feed_handler      = PolygonFeedHandler(bus, clock, tickers=all_adr_tickers)
    asian_feed_handler   = AsianFeedHandler(bus, clock, tickers=all_underlying_tickers)
    fx_handler           = OANDAFXHandler(bus, clock, currency_pairs=all_currencies)

    hs_engine       = HongSusmelEngine(bus, clock, pair_registry)
    risk_engine     = RiskEngine(bus, clock, risk_config)
    position_engine = PositionEngine(bus, clock)
    sequencer       = AsianExecutionSequencer(bus, clock)
    us_gateway      = InteractiveBrokersGateway(bus, clock, tws_port=config.ib_tws_port)
    roce_ruce       = RoceRuceCalculator(bus, clock, pair_registry)

    await bus.subscribe("market-data",    hs_engine.on_daily_bar)
    await bus.subscribe("fx-rates",       hs_engine.on_fx_rate)
    await bus.subscribe("signals",        risk_engine.on_signal)
    await bus.subscribe("risk-decisions", sequencer.on_signal)
    await bus.subscribe("fills",          sequencer.on_fill)
    await bus.subscribe("market-data",    sequencer.on_bar)
    await bus.subscribe("fills",          position_engine.on_fill)
    await bus.subscribe("market-data",    position_engine.on_bar)
    await bus.subscribe("fx-rates",       position_engine.on_fx_rate)
    await bus.subscribe("positions",      risk_engine.on_position_update)
    await bus.subscribe("pair-registry",  hs_engine.on_registry_update)
    await bus.subscribe("fills",          roce_ruce.on_fill)

    sequencer.register_gateway("us_equity", us_gateway)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(us_feed_handler.run())
        tg.create_task(asian_feed_handler.run())
        tg.create_task(fx_handler.run())
        tg.create_task(hs_engine.run())
        tg.create_task(risk_engine.run())
        tg.create_task(position_engine.run())
        tg.create_task(sequencer.run())
        tg.create_task(us_gateway.run())

asyncio.run(main())
```

### 8.2 Backpressure Handling

```python
class AsyncioBus:
    def __init__(self, max_queue_size: int = 10_000):
        self._queues: dict[str, asyncio.Queue] = defaultdict(
            lambda: asyncio.Queue(maxsize=max_queue_size)
        )

    async def publish(self, topic: str, event: BaseEvent):
        try:
            self._queues[topic].put_nowait(event)
        except asyncio.QueueFull:
            metrics.event_bus_dropped_total.labels(topic=topic).inc()
            logger.warning("bus_queue_full", topic=topic)
            if topic in {"orders", "fills", "risk-decisions"}:
                await self._queues[topic].put(event)   # block on critical topics
```

### 8.3 Multi-Process Scaling (Phase 3+)

Replace `AsyncioBus` with `KafkaBus`. Each component becomes its own Docker container / Kubernetes pod.

---

## 9. Backtesting Design

### 9.1 Deterministic Replay

See §3.12 for the full multi-feed replay implementation. The `MemoryBus` (synchronous, no `asyncio`) is used in backtest mode for maximum speed and determinism.

### 9.2 Slippage Models

```python
class HalfSpreadSlippageModel:
    """U.S. ADR — estimated half-spread from OHLCV bar."""
    def calculate(self, bar: BarEvent, side: str, qty: Decimal) -> Decimal:
        estimated_spread = (bar.high - bar.low) * Decimal("0.1")
        direction = 1 if side == "buy" else -1
        return bar.close + direction * estimated_spread / 2

class SqrtImpactSlippageModel:
    """Asian underlying — sqrt of participation rate as market impact."""
    def calculate(self, bar: BarEvent, side: str, qty: Decimal) -> Decimal:
        adv = bar.volume * bar.close
        if adv == 0:
            return bar.close
        participation = qty * bar.close / adv
        direction = 1 if side == "buy" else -1
        return bar.close * (1 + direction * Decimal("0.1") * participation.sqrt())
```

### 9.3 ROCE / RUCE Distribution Output

The backtest report produces per-pair and aggregate distribution tables matching paper Table 7-B format:

```
Metric                    Mean     Std    Max    p90    p75    Median  p25    p10    Min
────────────────────────────────────────────────────────────────────────────────────────
ROCE per trade            0.046   0.057  0.488  0.088  0.053  0.028  0.011  -0.005 -0.382
RUCE per trade            0.076   0.097  0.877  0.159  0.095  0.053  0.020  -0.009 -0.624
Duration (days)           5.91    9.25   89     13     6      3      1       1      0
Trades per firm per year  11.7    4.98   42     16     14     11.6   8.7    5.8    0
Roll cost (round trip)    0.027   0.015  0.11   0.034  0.021  0.016  0.008  0.005  0.001
```

### 9.4 Validation Targets (Paper Benchmarks)

Before any backtest result is considered valid:

| Check | Target |
|-------|--------|
| Median ROCE | ~2.8% (±0.5%) |
| Median RUCE | ~5.3% (±0.5%) |
| Median duration | 3 days |
| IQ range of duration | 1–6 days |
| ADR leg contribution | ~90% of total return |
| Liquidity bucket monotonicity | Low bucket > High bucket (ROCE) |
| Overnight abort rate | < 30% of initiated positions |

---

## 10. Risk and Reliability

### 10.1 Kill Switch

```python
class KillSwitch:
    async def trigger(self, reason: str, operator: str = "system"):
        if self._triggered:
            return
        self._triggered = True
        logger.critical("KILL_SWITCH_TRIGGERED", reason=reason, operator=operator)
        await self.bus.publish("system", KillSwitchEvent(reason=reason))
        await self.sequencer.close_all_pairs(order_type=OrderType.MARKET)
        await self.sequencer.cancel_all_orders()
        await self.alerting.send_critical(f"Kill switch: {reason}")
        await self.audit_logger.log_kill_switch(reason, operator)
```

### 10.2 Pre-Trade Risk Limits

| Limit Type | Default Value | Scope |
|-----------|--------------|-------|
| Max simultaneous open pairs | 20 | Global |
| Max notional per leg | $100,000 | Per pair |
| Max gross portfolio notional | $2,000,000 | Global |
| Max daily loss | 2% of AUM | Global |
| Max drawdown kill switch | 5% of AUM | Global |
| ADR zero-return-day entry block | 50% | Per pair |
| Holding period force-close | 90 days | Per pair |
| Max country concentration | 30% of gross | Per country |
| Rate of loss | $25,000 / 30 bars | Global |

### 10.3 Circuit Breakers

```python
class AsianADRCircuitBreakers:
    """
    1. Foreign Data Gap: if Asian underlying data has gaps > 3 bars,
       mark affected pairs data-unavailable; skip signals.
    2. ADR Ratio Discrepancy: if live ratio differs from registered by > 5%,
       suspend pair and alert operator.
    3. Zero-Return Cluster: if > 50% of active pairs trigger ZeroReturnDayFilter
       on the same bar, flag potential data-provider issue; alert operator.
    4. Overnight Abort Cluster: if > 20% of AWAITING_LOCAL positions abort on
       the same morning (mass spread reversal), alert operator and throttle new entries.
    """
```

### 10.4 Reconnect Logic

```python
async def reconnect_with_backoff(connect_fn, max_retries=10):
    for attempt in range(max_retries):
        try:
            return await connect_fn()
        except ConnectionError:
            if attempt == max_retries - 1:
                raise
            delay = min(2 ** attempt + random.uniform(0, 1), 60)
            logger.warning("reconnect_attempt", attempt=attempt, delay=delay)
            await asyncio.sleep(delay)
```

### 10.5 Audit Logging

Every order, fill, risk decision, spread computation, FX rate event, pair registry change, overnight abort event, and system event is written to an append-only audit log:
- UTC timestamp, component ID, event type, full JSON payload, operator identity

Stored in: PostgreSQL `audit_log` table + S3 archive (never deleted).

---

## 11. Observability

### 11.1 Structured Logging

```python
import structlog
logger = structlog.get_logger()

logger.info("hs_signal_generated",
    pair_id=signal.pair_id,
    signal=signal.signal,
    z_score=str(signal.z_score),
    spread=str(signal.spread),
    days_held=signal.days_held)
```

### 11.2 Prometheus Metrics

```python
spread_z_score = Gauge(
    "hs_spread_z_score",
    "Current z-score of dollar spread",
    ["pair_id", "adr_ticker"],
)
days_held = Gauge(
    "hs_position_days_held",
    "Days current position has been open",
    ["pair_id"],
)
overnight_aborts_total = Counter(
    "hs_overnight_aborts_total",
    "Positions aborted due to overnight spread reversal",
    ["pair_id"],
)
force_closes_total = Counter(
    "hs_force_closes_total",
    "Positions closed due to holding period expiry",
    ["pair_id"],
)
roce_per_trade = Histogram(
    "hs_roce_per_trade",
    "ROCE per closed round-trip",
    ["pair_id", "liquidity_bucket"],
    buckets=[-.10, -.05, -.02, 0, .01, .02, .03, .05, .08, .10, .15, .20],
)
zero_return_pct = Gauge(
    "hs_adr_zero_return_pct",
    "Rolling zero-return-day percentage for ADR",
    ["pair_id", "adr_ticker"],
)
```

### 11.3 Key Dashboards (Grafana)

| Dashboard | Key Panels |
|-----------|-----------|
| **Portfolio Overview** | Net PnL, gross notional, pairs active, daily fills, overnight aborts |
| **Spread Monitor** | Z-score heatmap per pair, spread vs thresholds time series |
| **ROCE / RUCE Tracker** | Live distribution histogram; rolling median vs paper benchmark |
| **Liquidity Monitor** | Zero-return-day rates per pair; bucket assignment changes |
| **System Health** | Latency, queue depths, feed reconnects, FX staleness |
| **Risk Monitor** | Drawdown meter, country concentration, kill switch status |

### 11.4 Latency Targets

```
Daily bar received from U.S. feed:           baseline
Daily bar received from Asian feed:          baseline
FX rate received (daily close):              baseline
Feed → Event Bus:                            target < 50ms   (daily bar; not HFT)
Event Bus → H&S Engine:                      target < 5ms
H&S Engine spread computation:               target < 2ms
Signal → Risk evaluation:                    target < 5ms
Risk Decision → Execution Sequencer:         target < 2ms
Sequencer → U.S. gateway:                    target < 10ms
Gateway → Broker API:                        target < 100ms
──────────────────────────────────────────────────────────
Total signal-to-order (end-of-day batch):    target < 200ms
```

---

## 12. Agile Implementation Roadmap

### Phase 1: Research Foundation — WRDS Data + Pair Selection

**Objectives**: Validate Asian ADR pair selection end-to-end; no live trading

**Architecture Decisions**:
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

---

### Phase 2: Backtesting Engine

**Objectives**: Validate strategy on historical WRDS Datastream data; reproduce paper benchmarks

**Architecture Decisions**:
- `MemoryBus` (synchronous) for speed and determinism
- `SimulatedClock` driven by merged U.S. / Asian / FX daily streams
- Point-in-time pair registry
- Next-bar fill model correctly simulates overnight gap
- Full cost model: commission, SEC fee, short borrow, stamp duty, local levy

**Deliverables**:
- `core/events.py` — all event dataclasses
- `strategy/hong_susmel/engine.py` — `HongSusmelEngine`
- `strategy/hong_susmel/execution_sequencer.py` — `AsianExecutionSequencer`
- `strategy/hong_susmel/state.py` — `HSPairState`, `WelfordRollingStats(ddof=1)`
- `risk/` — `HoldingPeriodForceCloseRule`, `ZeroReturnDayFilter`, `OvernightAbortCoverRule`
- `backtest/` — complete module with multi-feed replay
- `backtest/roce_ruce_calculator.py` — ROCE/RUCE per trade; aggregate distributions
- `backtest/report.py` — HTML tearsheet matching paper Table 7-B format
- Paper validation: median ROCE ≈ 2.8%, median RUCE ≈ 5.3%, median duration ≈ 3 days

---

### Phase 3: Live Data Feed Integration

**Objectives**: Connect to Polygon.io (U.S. ADR) and Asian feeds; run in paper mode

**Deliverables**:
- `feed_handler/connectors/polygon.py` — U.S. ADR end-of-day WebSocket
- `feed_handler/connectors/asian_feed.py` — Asian underlying daily bars
- `fx_handler/connectors/oanda.py` — OANDA daily FX close rates
- `runners/live_runner.py` — wires all components (paper mode)
- Integration tests replaying captured feed sessions

---

### Phase 4: Persistence + Event Bus

**Objectives**: Survive restarts; persist fills, positions, and pair state

**Deliverables**:
- `persistence/` — complete module
- `event_bus/kafka_bus.py`
- Alembic migrations (pair registry, orders, fills, positions, audit log)
- State recovery on startup (reload open positions; rebuild rolling stats from bar history)
- Docker Compose with all infrastructure services

---

### Phase 5: Live Broker Connectivity

**Objectives**: Execute real U.S. ADR short/cover trades

**Deliverables**:
- `gateways/interactive_brokers/tws_gateway.py` — primary U.S. ADR gateway
- `gateways/interactive_brokers/short_locate.py` — verify locate before every SELL
- `runners/data_recorder.py` — record live bars for subsequent backtesting
- Integration tests against Interactive Brokers paper trading account

---

### Phase 6: Full Risk Controls + Observability

**Objectives**: Production-grade risk management and full observability stack

**Deliverables**:
- `risk/rules/` — all rule implementations
- Kill switch REST endpoint
- `monitoring/` — complete module with spread and ROCE/RUCE metrics
- Grafana dashboards: spread monitor, ROCE/RUCE tracker, liquidity monitor
- Alertmanager rules: overnight abort cluster, force-close cluster, zero-return-day spike
- Runbooks for all alert types

---

### Phase 7: Dynamic Pair Rotation + Kubernetes

**Objectives**: Automate weekly re-screening; production Kubernetes deployment

**Deliverables**:
- Weekly cron jobs: re-run Asian ADR screener and registry update
- Helm chart for Kubernetes deployment
- CI/CD pipeline (GitHub Actions → ArgoCD)
- Full type coverage; 80%+ test coverage

---

## 13. Deployment Strategy

### 13.1 Local Development Setup

```bash
# Prerequisites: Python 3.12+, uv, Docker Desktop, WRDS account, OANDA account, Interactive Brokers account (TWS/Gateway running)

git clone https://github.com/your-org/asian-adr-strategy
cd asian-adr-strategy
uv sync

cp .env.example .env
# Edit .env: add WRDS credentials, POLYGON, ALPACA, OANDA settings

docker compose up -d postgres redis
uv run alembic upgrade head

# Fetch historical data
uv run python datastream/fetch_datastream_adr_data.py --start 2000-01-01 --end 2011-12-31
uv run python datastream/fetch_datastream_global_data.py --start 2000-01-01 --end 2011-12-31
uv run python datastream/fetch_fx_history.py --start 2000-01-01 --end 2011-12-31

# Run pair selection
uv run python datastream/run_asian_adr_screening.py --as-of 2002-01-01

# Run backtest (reproduces paper)
uv run python datastream/run_backtest.py --start 2002-01-01 --end 2011-12-31

# Run in paper trading mode
uv run python -m asian_adr.runners.live_runner --config config/development.toml
```

### 13.2 Docker Compose Architecture

```yaml
services:
  postgres:
    image: timescale/timescaledb:latest-pg16
    volumes: [postgres_data:/var/lib/postgresql/data]
    environment:
      POSTGRES_DB: asian_adr
      POSTGRES_PASSWORD: ${DB_PASSWORD}

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes

  kafka:
    image: confluentinc/cp-kafka:7.6.0
    depends_on: [zookeeper]
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092

  prometheus:
    image: prom/prometheus:latest
    volumes: [./prometheus.yml:/etc/prometheus/prometheus.yml]

  grafana:
    image: grafana/grafana:latest
    volumes: [./grafana/dashboards:/etc/grafana/dashboards]
    depends_on: [prometheus]

  us-feed-handler:
    build: {context: .., dockerfile: docker/Dockerfile.trading}
    command: python -m asian_adr.runners.us_feed_handler_runner
    environment: {FEED_PROVIDER: polygon}
    depends_on: [kafka]

  asian-feed-handler:
    build: {context: .., dockerfile: docker/Dockerfile.trading}
    command: python -m asian_adr.runners.asian_feed_handler_runner
    depends_on: [kafka]

  fx-handler:
    build: {context: .., dockerfile: docker/Dockerfile.trading}
    command: python -m asian_adr.runners.fx_handler_runner
    environment: {FX_PROVIDER: oanda}
    depends_on: [kafka]

  hs-engine:
    build: {context: .., dockerfile: docker/Dockerfile.trading}
    command: python -m asian_adr.runners.hs_engine_runner
    depends_on: [kafka, redis, postgres]

  risk-engine:
    build: {context: .., dockerfile: docker/Dockerfile.trading}
    command: python -m asian_adr.runners.risk_runner
    depends_on: [kafka, redis, postgres]

  sequencer:
    build: {context: .., dockerfile: docker/Dockerfile.trading}
    command: python -m asian_adr.runners.sequencer_runner
    depends_on: [kafka, postgres]

  position-engine:
    build: {context: .., dockerfile: docker/Dockerfile.trading}
    command: python -m asian_adr.runners.position_runner
    depends_on: [kafka, postgres, redis]

  research-scheduler:
    build: {context: .., dockerfile: docker/Dockerfile.research}
    command: python datastream/rescreen.py --as-of today
    depends_on: [postgres]
    environment:
      WRDS_USERNAME: ${WRDS_USERNAME}
      WRDS_PASSWORD: ${WRDS_PASSWORD}
      OANDA_API_KEY:  ${OANDA_API_KEY}

volumes:
  postgres_data:
```

### 13.3 Production Deployment Evolution

| Phase | Infrastructure | When to Use |
|-------|---------------|-------------|
| Phase 1–2 | Local laptop / single VM | Research, backtesting |
| Phase 3–5 | Docker Compose on dedicated VM | Paper trading, small live deployment |
| Phase 6 | Docker Compose + managed DBs (RDS, ElastiCache) | Production live trading |
| Phase 7 | Kubernetes (EKS/GKE) + Helm + ArgoCD | High availability |

---

## 14. Engineering Best Practices

### 14.1 Typing Strategy

```toml
[tool.mypy]
strict = true
python_version = "3.12"
disallow_untyped_defs = true
warn_return_any = true
```

Domain primitives: `PairId = NewType("PairId", str)`, `ADRRatio = NewType("ADRRatio", Decimal)`, `ZScore = NewType("ZScore", Decimal)`.

### 14.2 Linting and Formatting

```toml
[tool.ruff]
target-version = "py312"
line-length = 100
select = ["E", "W", "F", "I", "B", "UP", "SIM", "TCH", "RUF"]
```

### 14.3 Testing Hierarchy

```
tests/
├── unit/           # No I/O; pure function tests
│                   # Synthetic spread series with known µ/σ/threshold properties
│                   # Target: < 200ms total; 90%+ coverage of core/ and strategy/
│
├── integration/    # Uses MemoryBus; wires real components together
│                   # Covers H&S signal → risk → sequencer → simulated broker path
│                   # Covers overnight abort path explicitly
│                   # Target: < 30s total
│
└── system/         # Full backtest on WRDS-sourced Parquet files
                    # Golden output: median ROCE within ±0.5% of paper benchmark
                    # Target: < 10 min total
```

Shared `conftest.py` provides: `memory_bus`, `simulated_clock`, `mock_us_gateway`, `mock_asian_gateway`, `sample_pair_registry`, `synthetic_spread_series`, `synthetic_fx_series`.

### 14.4 Configuration

```toml
# config/base.toml

[trading]
environment = "development"
log_level   = "INFO"
broker      = "interactive_brokers"

[risk]
max_open_pairs          = 20
max_notional_per_leg    = 100_000
max_daily_loss_pct      = 0.02
drawdown_kill_switch_pct= 0.05
max_zero_return_pct     = 0.50
max_country_conc_pct    = 0.30

[hong_susmel]
estimation_days = 60       # T
holding_days    = 90       # H
k0              = 2.0      # entry multiplier
kc              = 0.0      # exit multiplier (law-of-one-price convergence)
return_metric   = "RUCE"   # ROCE | RUCE for live P&L tracking

[research]
min_non_zero_return_pct  = 0.50
min_trading_years        = 2
adr_rescreen_frequency   = "weekly"
```

### 14.5 Dependencies

```toml
[project]
name = "asian-adr-strategy"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "structlog>=24.0",
    "wrds>=3.2",
    "polars>=0.20",
    "pyarrow>=16.0",
    "statsmodels>=0.14",
    "numpy>=1.26",
    "scipy>=1.12",
    "asyncpg>=0.29",
    "sqlalchemy[asyncio]>=2.0",
    "redis[hiredis]>=5.0",
    "aiokafka>=0.11",
    "websockets>=12.0",
    "ib_insync>=0.9",
    "alpaca-py>=0.20",
    "oandapyV20>=0.7",
    "prometheus-client>=0.20",
    "opentelemetry-sdk>=1.24",
]

[project.optional-dependencies]
dev      = ["ruff", "mypy", "pytest", "pytest-asyncio", "pytest-cov"]
research = ["jupyter", "plotly>=5.0", "quantstats>=0.0.62"]
```

### 14.6 Architecture Decision Records

```markdown
# ADR-001: Dollar Spread, Not Percentage Premium
## Status: Accepted
## Decision
Signal is the raw dollar spread: P_ADR − (P_local × FX) / ratio.
Percentage premium normalisation distorts the threshold geometry for high-ratio
pairs (e.g., ratio=10) and introduces unnecessary division by a potentially
noisy parity price.
```

```markdown
# ADR-002: β = ADR Ratio, Never OLS-Estimated
## Status: Accepted
## Decision
The cointegrating vector is determined by the legal ADR conversion ratio.
OLS estimation introduces look-ahead bias and produces time-varying hedges
that are inconsistent with the law-of-one-price framework.
```

```markdown
# ADR-003: No FX Hedge
## Status: Accepted
## Decision
The overnight gap (U.S. close → Asia open) makes simultaneous FX spot submission
impossible. The paper accepts residual FX exposure. fx_hedge_required = False on
all registry entries. FX rates are used only to convert local prices to USD for
spread computation.
```

```markdown
# ADR-004: One-Sided Direction Only
## Status: Accepted
## Decision
Indonesia, Taiwan, China, India, and Korea all had short-selling restrictions on
domestic shares during the 2000–2011 study period (Bris, Goetzmann & Zhu 2007).
HSSignal enum has no LONG_ADR variant. The engine never emits a long-local / short-
ADR-cover signal regardless of spread sign.
```

```markdown
# ADR-005: Next-Bar Fill Model (Overnight Gap)
## Status: Accepted
## Decision
The simulated broker always fills on the bar after order submission. This correctly
models: (a) the ADR short leg filled at U.S. close, (b) the local leg conditionally
filled at the next Asian open after re-checking the spread. Same-bar fills would
produce systematically optimistic backtest results.
```

### 14.7 Reproducibility Checklist

Before any backtest result is considered valid:

- [ ] Configuration file committed and tagged
- [ ] `uv.lock` committed (exact dependency versions)
- [ ] Datastream ADR Parquet snapshot referenced by content hash
- [ ] Datastream Global Parquet snapshot referenced by content hash
- [ ] OANDA FX Parquet snapshot referenced by content hash
- [ ] Point-in-time ADR pair registry snapshot used (no look-ahead)
- [ ] ADR ratio values as-of backtest start date used
- [ ] Git commit SHA recorded in backtest report
- [ ] Golden output tests pass in CI
- [ ] Median ROCE within ±0.5% of paper benchmark (Table 4 Panel A)
- [ ] Median duration within ±1 day of paper benchmark (3 days)
- [ ] Liquidity bucket medians monotonically increasing (High < Low)
- [ ] Overnight abort events logged and counted
- [ ] Force-close events logged and counted

### 14.8 Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `ENVIRONMENT` | Yes | `development` / `staging` / `production` |
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `REDIS_URL` | Yes | Redis connection string |
| `KAFKA_BOOTSTRAP` | Phase 4+ | Kafka broker addresses |
| `WRDS_USERNAME` | Yes | WRDS portal username |
| `WRDS_PASSWORD` | Yes | WRDS portal password |
| `POLYGON_API_KEY` | Yes | Polygon.io API key (U.S. ADR live feed) |
| `IB_TWS_PORT` | Yes | Interactive Brokers TWS port (7496 live / 7497 paper) |
| `IB_CLIENT_ID` | Yes | Interactive Brokers client ID for TWS connection |
| `ALPACA_API_KEY` | No | Alpaca API key (fallback paper trading only) |
| `ALPACA_API_SECRET` | No | Alpaca API secret (fallback paper trading only) |
| `OANDA_API_KEY` | Yes | OANDA v20 API key (FX daily rates) |
| `SLACK_WEBHOOK_URL` | Phase 6+ | Slack alerting webhook |
| `KILL_SWITCH_SECRET` | Phase 6+ | Auth token for manual kill switch API |
| `S3_BUCKET` | Phase 3+ | S3/MinIO bucket for Parquet cache |

---

*This document is a living specification. Update it via PR with each significant architectural change. Archive superseded decisions in `docs/adr/`. Re-run the Asian ADR pair selection pipeline at least weekly and re-validate backtest golden outputs after any change to the Hong & Susmel Engine, Risk Engine, or Position Engine.*
