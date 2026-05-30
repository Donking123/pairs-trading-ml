# Hong & Susmel Engine

**Package**: `asian_adr.strategy.hong_susmel`

## Responsibilities

- Subscribe to `BarEvent` for each ADR and its Asian underlying (daily bars)
- Subscribe to `FXRateEvent` for USD conversion of local price
- Maintain per-pair `WelfordRollingStats(ddof=1)` over estimation window `T`
- Compute dollar spread on every daily bar update when both legs are fresh
- Emit `HongSusmelSignalEvent(signal=SHORT_ADR)` when `spread_t > κ_open`
- Emit `HongSusmelSignalEvent(signal=EXIT)` when `spread_t < κ_close`
- Emit `HongSusmelSignalEvent(signal=FORCE_CLOSE)` when holding days ≥ H
- Enforce stale-leg guard: both legs must carry the same bar date before computing spread
- Direction is permanently one-sided: `SHORT_ADR` only (ADR-004)

**Inputs**: `BarEvent` (topic: `market-data`), `FXRateEvent` (topic: `fx-rates`), `PairRegistryUpdateEvent` (topic: `pair-registry`)
**Outputs**: `HongSusmelSignalEvent` (topic: `signals`)

## Module Structure

```
strategy/hong_susmel/
├── engine.py               # HongSusmelEngine: spread computation, signal emission
├── state.py                # HSPairState: prices, rolling stats, holding-period counter
├── signal_factory.py       # Constructs HongSusmelSignalEvent
└── liquidity_bucket.py     # Assigns High/High-Med/Med-Low/Low bucket for attribution
```

## Engine Implementation

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

## WelfordRollingStats

Rolling mean and standard deviation computed in O(1) per update using Welford's online algorithm with `ddof=1` (sample standard deviation), matching the paper's estimation methodology.

## Design Rules

- Engine never imports from `risk/`, `position/`, or `gateways/`
- Engine emits signals only — it cannot submit orders directly
- `HSSignal` enum has no `LONG_ADR` variant; entry is permanently one-sided
- All time references use the injected `Clock`, never `datetime.now()`
- Spread is the raw dollar spread — log-spread or percentage premium are never used (ADR-001)
- `adr_ratio` is always taken from the pair registry — never re-estimated (ADR-002)
