"""Overnight abort-cover monitor rule (architecture §3.7).

If the ADR leg filled (short) but the spread reversed overnight and no local
leg was placed, the naked ADR short must be covered. The execution sequencer
already performs this cover inline at the Asia open; this rule is the
independent risk-side safety check that the same condition is flagged, so a
stuck ``AWAITING_ASIA_OPEN`` position cannot silently carry naked-short
inventory.

It is a *monitor* rule (not an entry gate): a non-passing result means
"naked short detected — cover required".
"""

from __future__ import annotations

from dataclasses import dataclass

from asian_adr.core import HongSusmelSignalEvent

from .base import AbstractRiskRule, RiskContext, RiskRuleResult, Severity

__all__ = ["OvernightAbortCoverRule", "NakedShortState"]


@dataclass
class NakedShortState:
    """Minimal view the rule needs of a pair awaiting its local leg."""

    pair_id: str
    adr_filled: bool          # ADR short confirmed
    local_filled: bool        # local leg established
    spread_reversed: bool     # spread fell below kappa_close overnight


class OvernightAbortCoverRule(AbstractRiskRule):
    name = "overnight_abort_cover"

    def evaluate_state(self, state: NakedShortState) -> RiskRuleResult:
        """Evaluate a specific awaiting-local position for naked-short risk."""
        if state.adr_filled and not state.local_filled and state.spread_reversed:
            return RiskRuleResult.block(
                self.name,
                f"pair {state.pair_id}: ADR short filled, local leg never "
                f"placed, spread reversed → cover naked short",
                severity=Severity.WARN,
            )
        return RiskRuleResult.ok(self.name)

    def evaluate(
        self, signal: HongSusmelSignalEvent | None, ctx: RiskContext
    ) -> RiskRuleResult:
        # No naked-short context at signal time; the stateful path is
        # :meth:`evaluate_state`, called by the monitor loop.
        return RiskRuleResult.ok(self.name)
