# Backtesting & Replay Engine

## Responsibilities

- Load U.S. ADR, Asian underlying, and FX daily prices from Parquet cache
- Replay events in chronological order via merged min-heap stream
- Enforce point-in-time pair registry (no look-ahead on pair selection dates)
- Simulate U.S. equity fills with next-bar execution (models overnight gap)
- Simulate Asian equity fills with next-bar execution at local open price
- Apply realistic cost model: commission, SEC fee, short borrow, local levy
- Produce tearsheet with ROCE/RUCE distributions matching paper Table 7-B format

## Deterministic Replay Loop

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

`flush()` after each event enforces strict causality — no component can react to a bar that has not been published yet.

## Cost Model

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

## Slippage Models

| Model | Application |
|-------|-------------|
| `HalfSpreadSlippageModel` | U.S. ADR — estimated half-spread from OHLCV bar; `(high − low) × 0.1 / 2` |
| `SqrtImpactSlippageModel` | Asian underlying — sqrt of participation rate as market impact |

```python
class HalfSpreadSlippageModel:
    def calculate(self, bar: BarEvent, side: str, qty: Decimal) -> Decimal:
        estimated_spread = (bar.high - bar.low) * Decimal("0.1")
        direction = 1 if side == "buy" else -1
        return bar.close + direction * estimated_spread / 2

class SqrtImpactSlippageModel:
    def calculate(self, bar: BarEvent, side: str, qty: Decimal) -> Decimal:
        adv = bar.volume * bar.close
        if adv == 0:
            return bar.close
        participation = qty * bar.close / adv
        direction = 1 if side == "buy" else -1
        return bar.close * (1 + direction * Decimal("0.1") * participation.sqrt())
```

## ROCE / RUCE Distribution Output

The backtest report produces per-pair and aggregate distribution tables matching paper Table 7-B:

```
Metric                    Mean     Std    Max    p90    p75    Median  p25    p10    Min
────────────────────────────────────────────────────────────────────────────────────────
ROCE per trade            0.046   0.057  0.488  0.088  0.053  0.028  0.011  -0.005 -0.382
RUCE per trade            0.076   0.097  0.877  0.159  0.095  0.053  0.020  -0.009 -0.624
Duration (days)           5.91    9.25   89     13     6      3      1       1      0
Trades per firm per year  11.7    4.98   42     16     14     11.6   8.7    5.8    0
Roll cost (round trip)    0.027   0.015  0.11   0.034  0.021  0.016  0.008  0.005  0.001
```

## Validation Targets (Paper Benchmarks)

Before any backtest result is considered valid:

| Check | Target |
|-------|--------|
| Median ROCE | ~2.8% (±0.5%) |
| Median RUCE | ~5.3% (±0.5%) |
| Median duration | 3 days (±1) |
| IQ range of duration | 1–6 days |
| ADR leg contribution | ~90% of total return |
| Liquidity bucket monotonicity | Low bucket > High bucket (ROCE) |
| Overnight abort rate | < 30% of initiated positions |

## Reconnect Logic

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

## Determinism Guarantees

- `SimulatedClock` never reads wall-clock time
- `MemoryBus` is synchronous — no concurrency, no race conditions
- All parameters and the git SHA are recorded in the report header
- Golden-output CI tests: re-run on identical Parquet snapshots must produce byte-identical results

## Reproducibility Checklist

See [engineering-practices.md](engineering-practices.md) for the full checklist.
