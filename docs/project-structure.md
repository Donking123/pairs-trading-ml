# Project Structure

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
│       │   ├── clock.py                # AbstractClock, LiveClock, SimulatedClock
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
    ├── architecture/
    ├── runbooks/
    └── adr/
```

## Structural Rules

- `core/` has zero external dependencies — only stdlib + pydantic
- `datastream/` scripts are standalone — they never import from `src/asian_adr/`
- `strategy/` never imports from `risk/`, `position/`, or `gateways/` directly
- `notebooks/` never imported by `src/` (enforced by ruff rule)
- Each component under `src/asian_adr/` can evolve into its own microservice package

## Live Runner Wiring

```python
async def main():
    bus   = AsyncioBus()
    clock = LiveClock()

    pair_registry = await AsianADRRegistry.load_active(db_url=config.database_url)

    us_feed_handler    = PolygonFeedHandler(bus, clock, tickers=pair_registry.all_adr_tickers())
    asian_feed_handler = AsianFeedHandler(bus, clock, tickers=pair_registry.all_underlying_tickers())
    fx_handler         = OANDAFXHandler(bus, clock, currency_pairs=pair_registry.all_currencies())

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
