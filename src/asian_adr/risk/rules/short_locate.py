"""ADR short-locate verification.

Every ADR entry is a short sale; the strategy must confirm a borrow/locate is
available before approving the SELL. Locate availability is resolved upstream
(broker query in live, always-available in backtest) and supplied on the
context.
"""

from __future__ import annotations

from asian_adr.core import HongSusmelSignalEvent

from .base import AbstractRiskRule, RiskContext, RiskRuleResult

__all__ = ["ShortLocateRule"]


class ShortLocateRule(AbstractRiskRule):
    name = "short_locate"

    def evaluate(
        self, signal: HongSusmelSignalEvent | None, ctx: RiskContext
    ) -> RiskRuleResult:
        if ctx.config.require_short_locate and not ctx.short_locate_available:
            adr = signal.adr_ticker if signal else (ctx.pair.adr_ticker if ctx.pair else "?")
            return RiskRuleResult.block(
                self.name, f"no short-locate available for ADR {adr}"
            )
        return RiskRuleResult.ok(self.name)
