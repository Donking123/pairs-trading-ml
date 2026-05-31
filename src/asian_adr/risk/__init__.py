"""Risk layer: pre-trade gating engine, portfolio state, and rules.

The :class:`RiskEngine` sits between the strategy and the execution sequencer:
it consumes ``signals`` + ``positions`` and emits ``risk-decisions``. It imports
only ``core`` and ``event_bus`` — the strategy's zero-return percentage and the
broker's short-locate availability are supplied via injected providers, so the
risk layer never depends on ``strategy`` or ``gateways``.
"""

from __future__ import annotations

from .engine import RiskEngine
from .rules import (
    AbstractRiskRule,
    CountryConcentrationRule,
    DrawdownLimitsRule,
    HoldingPeriodForceCloseRule,
    KillSwitch,
    KillSwitchEvent,
    KillSwitchRule,
    MaxOpenPairsRule,
    NotionalLimitsRule,
    OvernightAbortCoverRule,
    RateOfLossRule,
    RiskConfig,
    RiskContext,
    RiskRuleResult,
    Severity,
    ShortLocateRule,
    ZeroReturnDayFilter,
    country_of,
)
from .state import RiskState

__all__ = [
    "RiskEngine",
    "RiskState",
    "RiskConfig",
    "RiskContext",
    "RiskRuleResult",
    "Severity",
    "AbstractRiskRule",
    "NotionalLimitsRule",
    "MaxOpenPairsRule",
    "DrawdownLimitsRule",
    "RateOfLossRule",
    "HoldingPeriodForceCloseRule",
    "ZeroReturnDayFilter",
    "OvernightAbortCoverRule",
    "ShortLocateRule",
    "CountryConcentrationRule",
    "KillSwitch",
    "KillSwitchRule",
    "KillSwitchEvent",
    "country_of",
]
