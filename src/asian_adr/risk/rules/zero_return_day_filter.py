"""Zero-return-day entry filter (architecture §3.7).

Blocks new entries for pairs whose ADR zero-return-day percentage over the
estimation window exceeds the configured ceiling, preventing stale-price false
signals from illiquid ADRs. The percentage is supplied on the context by the
engine's injected ``zero_return_provider`` (so this rule needs no reference to
the strategy's per-pair state).
"""

from __future__ import annotations

from asian_adr.core import HongSusmelSignalEvent

from .base import AbstractRiskRule, RiskContext, RiskRuleResult

__all__ = ["ZeroReturnDayFilter"]


class ZeroReturnDayFilter(AbstractRiskRule):
    name = "zero_return_day_filter"

    def evaluate(
        self, signal: HongSusmelSignalEvent | None, ctx: RiskContext
    ) -> RiskRuleResult:
        if ctx.zero_return_pct > ctx.config.max_zero_return_pct:
            return RiskRuleResult.block(
                self.name,
                f"ADR zero-return {ctx.zero_return_pct:.1%} exceeds limit "
                f"{ctx.config.max_zero_return_pct:.1%}",
            )
        return RiskRuleResult.ok(self.name)
