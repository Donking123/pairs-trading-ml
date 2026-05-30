# Technology Stack

## Core Language

- **Python 3.12+**: asyncio, type hints, `match` statements, Polars / pandas for research

## Data Access

| Need | Technology | Notes |
|------|-----------|-------|
| U.S. ADR OHLCV (historical) | **WRDS Datastream** (`tr_ds_equities.wrds_ds2dsf`) | `typecode='ADR'`, `region='US'` |
| Asian underlying prices (historical) | **WRDS Datastream** (`tr_ds_equities.wrds_ds2dsf`) | Filtered to Asian exchange mnemonics |
| ADR reference data | **WRDS Datastream** (`tr_ds_equities.wrds_ds_names`) | Underlying mapping via `dscompcode` join; ADR ratio sourced separately |
| Historical FX rates (daily) | **WRDS Datastream** (`trdstrm.ds2fxrate` / `trdstrm.ds2fxcode`) | SPOT rates CCY/USD; inverted to USD/CCY on output |
| Live U.S. ADR data | **Polygon.io / Alpaca** | WebSocket; ADR tickers; daily bars |
| Live Asian data | **Asian Market Feed** | Provider TBD per exchange |
| Live FX rates | **OANDA REST v20** | Daily close rates |

## Messaging / Event Bus

| Option | Use Case | Notes |
|--------|----------|-------|
| **MemoryBus** | Backtest + unit tests | Synchronous; deterministic |
| asyncio.Queue | Live single-process prototype | Phase 1–2 only |
| **Kafka** | Production event bus | Best durability + replay |

## Databases

| Need | Technology | Notes |
|------|-----------|-------|
| Time-series OHLCV + FX | **TimescaleDB** | Hypertables, fast range scans |
| Pair registry / orders / fills | **PostgreSQL 16** | ADR pair metadata, order history, audit log |
| Hot state / spread cache | **Redis 7** | Latest spreads, z-scores per pair |
| Historical data store | **MinIO** (local) / **S3** (cloud) | Datastream + FX Parquet files |

## Quantitative Libraries

| Purpose | Library |
|---------|---------|
| ADF / PP cointegration tests | `statsmodels` |
| Rolling stats (Welford, ddof=1) | `WelfordRollingStats` (internal) |
| DataFrame processing | `polars` (fast) / `pandas` (compat.) |
| Backtest tearsheets | `quantstats` |
| Array operations | `numpy` |
| Statistical distributions | `scipy` |

## Observability Stack

| Concern | Technology |
|---------|-----------|
| Metrics | **Prometheus** + `prometheus_client` |
| Dashboards | **Grafana** |
| Logging | **structlog** → stdout → Loki |
| Tracing | **OpenTelemetry** → Jaeger |
| Alerting | **Alertmanager** + PagerDuty / Slack |

## Python Dependencies

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

## Deployment / Orchestration

| Phase | Tool |
|-------|------|
| 1–2 | Local laptop / single VM |
| 3–5 | Docker Compose on dedicated VM |
| 6 | Docker Compose + managed DBs (RDS, ElastiCache) |
| 7 | Kubernetes (EKS/GKE) + Helm + ArgoCD |
