"""Daily-loss and peak-to-trough drawdown limits.

Two thresholds, both expressed as a fraction of AUM:

* **Daily loss** beyond ``max_daily_loss_pct`` blocks new entries for the day.
* **Peak-to-trough drawdown** beyond ``max_drawdown_pct`` escalates to the kill
  switch (``KILL`` severity), halting all new risk.
"""

from __future__ import annotations

from asian_adr.core import HongSusmelSignalEvent

from .base import AbstractRiskRule, RiskContext, RiskRuleResult, Severity

__all__ = ["DrawdownLimitsRule"]


class DrawdownLimitsRule(AbstractRiskRule):
    name = "drawdown_limits"

    def evaluate(
        self, signal: HongSusmelSignalEvent | None, ctx: RiskContext
    ) -> RiskRuleResult:
        # Peak-to-trough drawdown → kill switch (checked first; harder stop).
        if ctx.drawdown_pct >= ctx.config.max_drawdown_pct:
            return RiskRuleResult.block(
                self.name,
                f"drawdown {ctx.drawdown_pct:.1%} hit kill-switch limit "
                f"{ctx.config.max_drawdown_pct:.1%}",
                severity=Severity.KILL,
            )
        # Daily loss → block new entries.
        daily_loss_limit = ctx.config.max_daily_loss_pct * ctx.config.aum_usd
        if ctx.daily_pnl_usd <= -daily_loss_limit:
            return RiskRuleResult.block(
                self.name,
                f"daily loss ${-ctx.daily_pnl_usd:,.0f} exceeds limit "
                f"${daily_loss_limit:,.0f}",
            )
        return RiskRuleResult.ok(self.name)
