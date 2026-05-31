"""Risk-rule framework: config, evaluation context, result, and base class.

Every pre-trade rule implements one method — :meth:`AbstractRiskRule.evaluate`
— taking the signal under consideration and a :class:`RiskContext` snapshot of
portfolio state, and returning a :class:`RiskRuleResult`. The engine runs the
rules in order, aggregates the results into a :class:`~asian_adr.core.RiskDecision`,
and escalates to the kill switch on any ``KILL`` result.

Keeping a single uniform signature (rather than the per-rule signatures sketched
in the spec) lets the engine treat the rule set as a homogeneous pipeline and
makes rules trivially composable and testable.

Depends only on the standard library, pydantic, and ``asian_adr.core``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import IntEnum

from pydantic import BaseModel, ConfigDict

from asian_adr.core import AsianADRApprovedPair, HongSusmelSignalEvent

__all__ = ["Severity", "RiskConfig", "RiskContext", "RiskRuleResult", "AbstractRiskRule"]


class Severity(IntEnum):
    """Ordered severity of a rule result; higher blocks harder.

    ``INFO``/``WARN`` are advisory and never reject an order. ``BLOCK`` rejects
    the order. ``KILL`` rejects *and* escalates to the global kill switch.
    """

    INFO = 0
    WARN = 1
    BLOCK = 2
    KILL = 3


class RiskConfig(BaseModel):
    """Pre-trade risk limits (architecture §10.2 defaults)."""

    model_config = ConfigDict(frozen=True)

    max_open_pairs: int = 20
    max_notional_per_leg: Decimal = Decimal("100000")
    max_gross_notional: Decimal = Decimal("2000000")
    max_zero_return_pct: Decimal = Decimal("0.50")
    max_holding_days: int = 90
    max_daily_loss_pct: Decimal = Decimal("0.02")        # of AUM
    max_drawdown_pct: Decimal = Decimal("0.05")          # of AUM → kill switch
    require_short_locate: bool = True
    rate_of_loss_limit: Decimal = Decimal("25000")       # dollars
    rate_of_loss_bars: int = 30
    max_country_concentration_pct: Decimal = Decimal("0.30")  # of gross
    aum_usd: Decimal = Decimal("1000000")
    # Notional granted to an approved entry (per leg).
    entry_notional_usd: Decimal = Decimal("100000")


@dataclass(frozen=True)
class RiskContext:
    """Immutable portfolio snapshot a rule evaluates a signal against.

    The engine assembles this from :class:`~asian_adr.risk.state.RiskState`
    plus the injected zero-return provider, so rules never reach into live
    engine internals (and ``risk`` never imports ``strategy``).
    """

    config: RiskConfig
    pair: AsianADRApprovedPair | None = None
    proposed_notional: Decimal = Decimal("0")
    zero_return_pct: Decimal = Decimal("0")
    open_pairs: int = 0
    gross_notional_usd: Decimal = Decimal("0")
    country: str | None = None
    country_notional_usd: Decimal = Decimal("0")
    daily_pnl_usd: Decimal = Decimal("0")
    drawdown_pct: Decimal = Decimal("0")
    loss_window_usd: Decimal = Decimal("0")   # signed; losses are negative
    short_locate_available: bool = True
    kill_switch_active: bool = False
    days_held: int = 0


class RiskRuleResult(BaseModel):
    """Outcome of a single rule evaluation."""

    model_config = ConfigDict(frozen=True)

    rule: str
    passed: bool
    reason: str = ""
    severity: Severity = Severity.INFO

    @classmethod
    def ok(cls, rule: str) -> "RiskRuleResult":
        return cls(rule=rule, passed=True)

    @classmethod
    def block(
        cls, rule: str, reason: str, severity: Severity = Severity.BLOCK
    ) -> "RiskRuleResult":
        return cls(rule=rule, passed=False, reason=reason, severity=severity)

    @property
    def is_blocking(self) -> bool:
        return not self.passed and self.severity >= Severity.BLOCK


class AbstractRiskRule(ABC):
    """Base class for every risk rule.

    Subclasses set :attr:`name` and implement :meth:`evaluate`.
    """

    name: str = "abstract_rule"

    @abstractmethod
    def evaluate(
        self, signal: HongSusmelSignalEvent | None, ctx: RiskContext
    ) -> RiskRuleResult:
        """Return a result; ``passed=True`` means the rule permits the order."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={self.name!r})"
