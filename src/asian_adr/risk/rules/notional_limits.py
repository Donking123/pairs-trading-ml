"""Notional and open-pair count limits."""

from __future__ import annotations

from asian_adr.core import HongSusmelSignalEvent

from .base import AbstractRiskRule, RiskContext, RiskRuleResult

__all__ = ["NotionalLimitsRule", "MaxOpenPairsRule"]


class NotionalLimitsRule(AbstractRiskRule):
    """Per-leg notional and gross-portfolio notional ceilings."""

    name = "notional_limits"

    def evaluate(
        self, signal: HongSusmelSignalEvent | None, ctx: RiskContext
    ) -> RiskRuleResult:
        if ctx.proposed_notional > ctx.config.max_notional_per_leg:
            return RiskRuleResult.block(
                self.name,
                f"per-leg notional ${ctx.proposed_notional:,.0f} exceeds "
                f"limit ${ctx.config.max_notional_per_leg:,.0f}",
            )
        projected_gross = ctx.gross_notional_usd + ctx.proposed_notional
        if projected_gross > ctx.config.max_gross_notional:
            return RiskRuleResult.block(
                self.name,
                f"gross notional ${projected_gross:,.0f} would exceed "
                f"limit ${ctx.config.max_gross_notional:,.0f}",
            )
        return RiskRuleResult.ok(self.name)


class MaxOpenPairsRule(AbstractRiskRule):
    """Caps the number of simultaneously open pairs."""

    name = "max_open_pairs"

    def evaluate(
        self, signal: HongSusmelSignalEvent | None, ctx: RiskContext
    ) -> RiskRuleResult:
        if ctx.open_pairs >= ctx.config.max_open_pairs:
            return RiskRuleResult.block(
                self.name,
                f"{ctx.open_pairs} open pairs at limit {ctx.config.max_open_pairs}",
            )
        return RiskRuleResult.ok(self.name)
