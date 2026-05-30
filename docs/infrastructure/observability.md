# Observability

## Structured Logging

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

All log output is JSON, shipped to Loki via Promtail.

## Module Structure

```
monitoring/
├── logger.py     # Structured JSON logger (structlog)
├── metrics.py    # Prometheus metric definitions
├── tracing.py    # OpenTelemetry tracer setup
├── alerting.py   # Alert rule engine and dispatcher
└── health.py     # Health check endpoints
```

## Prometheus Metrics

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

## Grafana Dashboards

| Dashboard | Key Panels |
|-----------|-----------|
| **Portfolio Overview** | Net PnL, gross notional, pairs active, daily fills, overnight aborts |
| **Spread Monitor** | Z-score heatmap per pair, spread vs thresholds time series |
| **ROCE / RUCE Tracker** | Live distribution histogram; rolling median vs paper benchmark |
| **Liquidity Monitor** | Zero-return-day rates per pair; bucket assignment changes |
| **System Health** | Latency, queue depths, feed reconnects, FX staleness |
| **Risk Monitor** | Drawdown meter, country concentration, kill switch status |

## Latency Targets

```
Daily bar received from U.S. feed:           baseline
Daily bar received from Asian feed:          baseline
FX rate received (daily close):              baseline
Feed → Event Bus:                            target < 50ms
Event Bus → H&S Engine:                     target < 5ms
H&S Engine spread computation:              target < 2ms
Signal → Risk evaluation:                   target < 5ms
Risk Decision → Execution Sequencer:        target < 2ms
Sequencer → U.S. gateway:                   target < 10ms
Gateway → Broker API:                       target < 100ms
──────────────────────────────────────────────────────────
Total signal-to-order (end-of-day batch):   target < 200ms
```

## Stack

| Concern | Technology |
|---------|-----------|
| Metrics | **Prometheus** + `prometheus_client` |
| Dashboards | **Grafana** |
| Logging | **structlog** → stdout → Loki |
| Tracing | **OpenTelemetry** → Jaeger |
| Alerting | **Alertmanager** + PagerDuty / Slack |

## Alert Rules

| Alert | Trigger | Severity |
|-------|---------|----------|
| Overnight abort cluster | > 20% of AWAITING_LOCAL positions abort same morning | Critical |
| Force-close cluster | > 5 force-closes in one bar | Warning |
| Zero-return-day spike | > 50% of pairs trigger ZeroReturnDayFilter same bar | Warning |
| FX staleness | FX rate age > 2 business days for any active currency | Critical |
| Kill switch triggered | Any | Critical |
| Feed gap | Asian or U.S. data gap > 3 bars | Warning |
