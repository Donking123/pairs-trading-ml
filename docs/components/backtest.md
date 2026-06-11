# Backtest Engine

**Package**: `asian_adr.backtest`

## Responsibilities

- Load U.S. ADR, Asian underlying, and FX daily prices from Parquet cache
- Replay events in chronological order via merged min-heap stream across all three feeds
- Enforce point-in-time pair registry (no look-ahead on pair selection dates)
- Drive identical H&S Engine, Risk Engine, and Position Engine as live trading
- Produce ROCE/RUCE tearsheet matching paper Table 7-B format

**Inputs**: Parquet cache (`adr_prices`, `global_prices`, `fx_rates`), `asian_adr_pairs.json`
**Outputs**: `trades.parquet`, `summary.json`, `distribution.json`, `tearsheet.html`

## Module Structure

```
backtest/
├── engine.py                     # Main replay loop (multi-feed: US, foreign, FX)
├── clock.py                      # SimulatedClock
├── data_loader.py                # U.S. ADR Datastream Parquet streaming
├── foreign_data_loader.py        # Datastream Global Parquet streaming
├── fx_data_loader.py             # Datastream FX history streaming
├── pair_registry_loader.py       # Point-in-time registry snapshots
├── simulated_us_exchange.py      # Virtual U.S. order book
├── simulated_foreign_exchange.py # Virtual Asian order book (backtest only)
├── slippage_models.py            # Half-spread, sqrt-impact
├── cost_model.py                 # Commission + SEC fee + short borrow + stamp duty + levy
├── roce_ruce_calculator.py       # ROCE/RUCE per trade; aggregate distributions
└── report.py                     # HTML tearsheet: paper-benchmarked distributions
```

## Key Design

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
            await bus.publish(self._route_event_topic(event), event)
            await bus.flush()
```

## Fill Model

- **U.S. ADR leg**: fills at next-bar close price after order submission
- **Local (Asian) leg**: fills at next-bar open price after order submission
- Both use next-bar semantics (ADR-005) to correctly model the overnight gap

## Simulated Cost Model

| Cost | U.S. ADR Leg | Asian Local Leg |
|------|-------------|-----------------|
| Commission | max($1.00, $0.005 × shares) | max(min_commission, rate × shares) |
| Regulatory fee | SEC fee: notional × 0.0000278 (sell only) | Local levy (exchange-specific) |
| Short borrow | notional × borrow_rate / 252 | — |
| Stamp duty | — | notional × stamp_rate (e.g., HK 0.10%, AU 0.00%) |

## Determinism Guarantees

- `MemoryBus` is synchronous — no concurrency, no race conditions
- `SimulatedClock` never reads wall-clock time
- All parameters and git SHA recorded in tearsheet header
- Re-run on identical Parquet snapshots produces byte-identical results

See [backtesting.md](../backtesting.md) for validation targets and reproducibility checklist.
