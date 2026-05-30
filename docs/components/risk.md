# Risk Management Engine

**Package**: `asian_adr.risk`

## Responsibilities

- Validate every `HongSusmelSignalEvent` against pre-trade risk rules before order submission
- Enforce position limits, drawdown limits, and holding-period force-close
- Block entries when ADR zero-return-day percentage exceeds threshold
- Detect FX conversion data staleness; suspend affected pairs
- Trigger kill switch on breach of critical thresholds
- Write every decision to the append-only audit log

**Inputs**: `HongSusmelSignalEvent` (topic: `signals`), `PositionUpdateEvent` (topic: `positions`)
**Outputs**: `RiskDecision` (topic: `risk-decisions`), `KillSwitchEvent` (topic: `alerts`)

## Module Structure

```
risk/
├── engine.py
├── state.py
└── rules/
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

## Risk Rule Interface

```python
class AbstractRiskRule(ABC):
    @abstractmethod
    def evaluate(
        self,
        signal: HongSusmelSignalEvent,
        state: HSPairState,
        config: RiskConfig,
    ) -> RiskRuleResult: ...
    # RiskRuleResult: passed=bool, severity=INFO|WARN|BLOCK|KILL, reason=str
```

## Key Rules

### Holding Period Force-Close

```python
class HoldingPeriodForceCloseRule(AbstractRiskRule):
    def evaluate(self, state: HSPairState, pair: AsianADRApprovedPair) -> bool:
        days_held = (self._clock.date() - state.entry_date).days
        return days_held >= pair.holding_days
```

### Zero Return Day Filter

```python
class ZeroReturnDayFilter(AbstractRiskRule):
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

### Overnight Abort Cover

Automatically covers a naked ADR short if the spread reversed overnight and no local leg was placed. Prevents one-sided inventory from overnight gap reversals.

## Pre-Trade Limits

| Limit | Default |
|-------|---------|
| Max simultaneous open pairs | 20 |
| Max notional per leg | $100,000 |
| Max gross portfolio notional | $2,000,000 |
| Max daily loss | 2% of AUM |
| Max drawdown kill switch | 5% of AUM |
| ADR zero-return-day entry block | 50% |
| Holding period force-close | 90 days |
| Max country concentration | 30% of gross |
| Rate of loss | $25,000 / 30 bars |
| ADR short-locate required | Yes |

## Kill Switch

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

**Reset**: requires explicit operator confirmation — never auto-reset in production.

## Circuit Breakers

| Breaker | Trigger | Action |
|---------|---------|--------|
| Foreign Data Gap | Asian underlying data gaps > 3 bars | Mark affected pairs data-unavailable; skip signals |
| ADR Ratio Discrepancy | Live ratio differs from registered by > 5% | Suspend pair; alert operator |
| Zero-Return Cluster | > 50% of active pairs trigger ZeroReturnDayFilter on same bar | Flag data-provider issue; alert operator |
| Overnight Abort Cluster | > 20% of AWAITING_LOCAL positions abort same morning | Alert operator; throttle new entries |

## Audit Logging

Every order, fill, risk decision, spread computation, FX rate event, pair registry change, overnight abort event, and system event is written to an append-only audit log:
- UTC timestamp, component ID, event type, full JSON payload, operator identity

Storage: PostgreSQL `audit_log` table + S3 archive (never deleted).
