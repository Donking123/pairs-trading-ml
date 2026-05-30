# Engineering Practices

## Type System

- **All source code**: strict type hints; no `Any` unless unavoidable
- Use `typing.Protocol` for duck-typed interfaces
- Use `NewType` for domain primitives: `PairId = NewType("PairId", str)`, `ADRRatio = NewType("ADRRatio", Decimal)`, `ZScore = NewType("ZScore", Decimal)`
- Use `TypeAlias` for complex types: `Price: TypeAlias = Decimal`

```toml
[tool.mypy]
strict = true
python_version = "3.12"
disallow_untyped_defs = true
warn_return_any = true
```

## Linting & Formatting

```toml
[tool.ruff]
target-version = "py312"
line-length = 100
select = ["E", "W", "F", "I", "B", "UP", "SIM", "TCH", "RUF"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

## Testing Hierarchy

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

## Configuration Management

```toml
# config/base.toml

[trading]
environment = "development"
log_level   = "INFO"
broker      = "interactive_brokers"

[risk]
max_open_pairs           = 20
max_notional_per_leg     = 100_000
max_daily_loss_pct       = 0.02
drawdown_kill_switch_pct = 0.05
max_zero_return_pct      = 0.50
max_country_conc_pct     = 0.30

[hong_susmel]
estimation_days = 60       # T
holding_days    = 90       # H
k0              = 2.0      # entry multiplier
kc              = 0.0      # exit multiplier (law-of-one-price convergence)
return_metric   = "RUCE"   # ROCE | RUCE for live P&L tracking

[research]
min_non_zero_return_pct = 0.50
min_trading_years       = 2
adr_rescreen_frequency  = "weekly"
```

## Secrets Handling

- **Never** commit secrets to git (enforced by `git-secrets` pre-commit hook)
- Local development: `.env` file (gitignored)
- Staging/Production: environment variables injected by CI/CD or Kubernetes Secrets

```bash
# .env (gitignored)
WRDS_USERNAME=xxx
WRDS_PASSWORD=xxx
OANDA_API_KEY=xxx
IB_TWS_PORT=7497
DATABASE_URL=postgresql+asyncpg://user:password@localhost/asian_adr
REDIS_URL=redis://localhost:6379/0
```

## Dependency Management

- **Tool**: `uv` (fast, lockfile-based, workspace support)
- **Policy**: pin all transitive dependencies in `uv.lock`
- **Security**: `uv audit` in CI to detect vulnerabilities
- **Updates**: Dependabot weekly PRs; human review before merge

## Architecture Decision Records (ADRs)

All significant architectural choices are documented in `docs/adr/`. Implemented decisions:

| ADR | Title | Status |
|-----|-------|--------|
| ADR-001 | Dollar Spread, Not Percentage Premium | Accepted |
| ADR-002 | β = ADR Ratio, Never OLS-Estimated | Accepted |
| ADR-003 | No FX Hedge | Accepted |
| ADR-004 | One-Sided Direction Only | Accepted |
| ADR-005 | Next-Bar Fill Model (Overnight Gap) | Accepted |

## Reproducibility Checklist

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
