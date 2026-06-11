# Deployment

## Local Development Setup

```bash
# Prerequisites: Python 3.12+, uv, Docker Desktop
# WRDS account, OANDA account, Interactive Brokers account (TWS/Gateway running)

git clone https://github.com/your-org/asian-adr-strategy
cd asian-adr-strategy
uv sync

cp .env.example .env
# Edit .env: add WRDS credentials, POLYGON, ALPACA, OANDA, IB settings

docker compose up -d postgres redis
uv run alembic upgrade head

# Fetch historical data (Phase 1)
uv run python datastream/fetch_datastream_adr_data.py --start 2000-01-01 --end 2011-12-31
uv run python datastream/fetch_datastream_global_data.py --start 2000-01-01 --end 2011-12-31
uv run python datastream/fetch_fx_history.py --start 2000-01-01 --end 2011-12-31

# Run pair selection (Phase 1)
uv run python datastream/run_asian_adr_screening.py --as-of 2002-01-01

# Run backtest (Phase 2)
uv run python datastream/run_backtest.py --start 2002-01-01 --end 2011-12-31

# Run in paper trading mode (Phase 3+)
uv run python -m asian_adr.runners.live_runner --config config/development.toml
```

## Docker Compose

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

## CI/CD Pipeline (GitHub Actions)

```yaml
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres: {image: timescale/timescaledb:latest-pg16}
      redis: {image: redis:7-alpine}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run ruff check src/ tests/
      - run: uv run mypy src/
      - run: uv run pytest tests/unit/ tests/integration/ -v --cov=asian_adr

  backtest-regression:
    needs: test
    steps:
      - run: uv run pytest tests/system/ -v  # Golden output tests

  build:
    needs: [test, backtest-regression]
    steps:
      - run: docker build -t asian-adr:${{ github.sha }} -f docker/Dockerfile.trading .
      - run: docker push ghcr.io/${{ github.repository }}/asian-adr:${{ github.sha }}

  deploy-staging:
    needs: build
    if: github.ref == 'refs/heads/main'
    steps:
      - run: argocd app sync asian-adr-staging
```

## Production Deployment Evolution

| Phase | Infrastructure | When to Use |
|-------|---------------|-------------|
| Phase 1–2 | Local laptop / single VM | Research, backtesting |
| Phase 3–5 | Docker Compose on dedicated VM | Paper trading, small live deployment |
| Phase 6 | Docker Compose + managed DBs (RDS, ElastiCache) | Production live trading |
| Phase 7 | Kubernetes (EKS/GKE) + Helm + ArgoCD | High availability, multi-strategy |

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `ENVIRONMENT` | Yes | `development` / `staging` / `production` |
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `REDIS_URL` | Yes | Redis connection string |
| `KAFKA_BOOTSTRAP` | Phase 4+ | Kafka broker addresses |
| `WRDS_USERNAME` | Yes | WRDS portal username |
| `WRDS_PASSWORD` | Yes | WRDS portal password |
| `POLYGON_API_KEY` | Phase 3+ | Polygon.io API key (U.S. ADR live feed) |
| `IB_TWS_PORT` | Phase 5+ | Interactive Brokers TWS port (7496 live / 7497 paper) |
| `IB_CLIENT_ID` | Phase 5+ | Interactive Brokers client ID |
| `ALPACA_API_KEY` | No | Alpaca API key (fallback paper trading only) |
| `ALPACA_API_SECRET` | No | Alpaca API secret |
| `OANDA_API_KEY` | Yes | OANDA v20 API key (FX daily rates) |
| `SLACK_WEBHOOK_URL` | Phase 6+ | Slack alerting webhook |
| `KILL_SWITCH_SECRET` | Phase 6+ | Auth token for manual kill switch API |
| `S3_BUCKET` | Phase 3+ | S3/MinIO bucket for Parquet cache |
