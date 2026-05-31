"""Holding-period force-close monitor rule (architecture §3.7).

A hard stop independent of spread level: if a pair has been held at least
``holding_days``, it must be force-closed. This is a *monitor* rule — it does
not gate new entries; the engine evaluates it against open positions and emits
a ``FORCE_CLOSE`` when it fires. A non-passing result here means "force-close
condition met", not "order rejected".
"""

from __future__ import annotations

from asian_adr.core import HongSusmelSignalEvent

from .base import AbstractRiskRule, RiskContext, RiskRuleResult, Severity

__all__ = ["HoldingPeriodForceCloseRule"]


class HoldingPeriodForceCloseRule(AbstractRiskRule):
    name = "holding_period_force_close"

    def evaluate(
        self, signal: HongSusmelSignalEvent | None, ctx: RiskContext
    ) -> RiskRuleResult:
        limit = ctx.pair.holding_days if ctx.pair else ctx.config.max_holding_days
        if ctx.days_held >= limit:
            return RiskRuleResult.block(
                self.name,
                f"held {ctx.days_held}d ≥ holding limit {limit}d → force-close",
                severity=Severity.WARN,
            )
        return RiskRuleResult.ok(self.name)
