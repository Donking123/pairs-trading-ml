"""Rate-of-loss limit: max dollar loss over a rolling window of bars.

Catches fast bleed that the daily-loss check would miss intra-day. The engine
maintains a rolling sum of realised+unrealised PnL deltas over the last
``rate_of_loss_bars`` bars and supplies it as ``loss_window_usd`` (signed).
"""

from __future__ import annotations

from asian_adr.core import HongSusmelSignalEvent

from .base import AbstractRiskRule, RiskContext, RiskRuleResult

__all__ = ["RateOfLossRule"]


class RateOfLossRule(AbstractRiskRule):
    name = "rate_of_loss"

    def evaluate(
        self, signal: HongSusmelSignalEvent | None, ctx: RiskContext
    ) -> RiskRuleResult:
        if ctx.loss_window_usd <= -ctx.config.rate_of_loss_limit:
            return RiskRuleResult.block(
                self.name,
                f"loss ${-ctx.loss_window_usd:,.0f} over {ctx.config.rate_of_loss_bars} "
                f"bars exceeds limit ${ctx.config.rate_of_loss_limit:,.0f}",
            )
        return RiskRuleResult.ok(self.name)
