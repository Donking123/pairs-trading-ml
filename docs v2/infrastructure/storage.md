# Storage & Persistence

## Storage Decision Matrix

| Data Type | Storage | Retention |
|-----------|---------|-----------|
| ADR + global OHLCV (Parquet) | S3 / MinIO | Unlimited |
| FX rates (Parquet) | S3 / MinIO | Unlimited |
| Pair registry | PostgreSQL | Versioned |
| Orders / fills | PostgreSQL | Unlimited |
| Audit log | PostgreSQL + S3 | Never deleted |
| Position snapshots | Redis + PostgreSQL | Hot: 24h |
| Rolling spread stats | Redis | Rebuilt on restart |
| Backtest results | S3 / MinIO + local | Unlimited |
| Event stream | Kafka | 7 days (Phase 3+) |

## Module Structure

```
persistence/
├── timescale.py        # TimescaleDB hypertable adapter (OHLCV time-series)
├── postgres.py         # Orders, fills, pair registry, audit log
├── redis_cache.py      # Hot state: latest spreads, z-scores, position snapshots
└── s3_store.py         # Parquet archive: historical bars, backtest results
```

## Technology Choices

| Need | Technology | Notes |
|------|-----------|-------|
| Time-series OHLCV + FX | **TimescaleDB** | PostgreSQL-compatible; hypertables; fast range scans |
| Pair registry / orders / fills | **PostgreSQL 16** | ADR pair metadata, order history, audit log |
| Hot state / spread cache | **Redis 7** | Latest spreads, z-scores, position summaries |
| Historical data store | **MinIO** (local) / **S3** (cloud) | Datastream + FX Parquet files; backtest outputs |

## Local Parquet Layout

```
datastream/data/parquet/
├── adr/
│   ├── adr_prices.parquet       # infocode, marketdate, OHLCV, adj_factor, ticker, isin
│   └── adr_reference.parquet    # adr_ticker, underlying_ticker, exchange, currency, ratio
├── global/
│   └── global_prices.parquet    # infocode, marketdate, OHLCV, adj_factor, ticker, exchange, currency
└── fx/
    └── fx_rates.parquet         # date, base_currency, quote_currency, mid (USD/CCY), provider
```

In production these files are stored in S3/MinIO and streamed by the backtest data loaders.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `REDIS_URL` | Yes | Redis connection string |
| `KAFKA_BOOTSTRAP` | Phase 3+ | Kafka broker addresses |
| `S3_BUCKET` | Phase 3+ | S3/MinIO bucket for Parquet cache |
