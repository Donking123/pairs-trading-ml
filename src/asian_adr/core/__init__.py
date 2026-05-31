"""Core domain layer: types, events, instruments, clocks, exceptions.

This package is the shared vocabulary of the platform and has **zero external
dependencies beyond the standard library and pydantic**. Nothing here imports
from any other ``asian_adr`` subpackage, so every component can depend on
``core`` without creating cycles.
"""

from __future__ import annotations

from .clock import Clock, LiveClock, SimulatedClock
from .events import (
    BarEvent,
    BaseEvent,
    FillEvent,
    FXRateEvent,
    HongSusmelSignalEvent,
    OrderRequest,
    PositionUpdateEvent,
    RiskDecision,
    RoceRuceResult,
)
from .exceptions import (
    ADRRatioUnresolvedError,
    AsianADRError,
    ConfigurationError,
    DataError,
    ExecutionError,
    GatewayError,
    KillSwitchTriggered,
    MissingBarError,
    MissingFXRateError,
    OrderRejected,
    OvernightAbort,
    PairNotFoundError,
    PairRegistryError,
    RiskError,
    RiskRuleViolation,
    SequencerStateError,
    ShortLocateUnavailable,
    StaleDataError,
)
from .instruments import (
    ASIAN_EXCHANGES,
    STANDARD_ADR_RATIOS,
    AsianADRApprovedPair,
    AsianADRPairSpec,
)
from .types import (
    BUY,
    FOREIGN_EQUITY,
    SELL,
    US_EQUITY,
    ADRRatio,
    CurrencyCode,
    ExchangeCode,
    FillId,
    HSPosition,
    HSSignal,
    LiquidityBucket,
    OrderId,
    OrderType,
    PairId,
    RiskDecisionStatus,
    Side,
    Ticker,
    Venue,
)

__all__ = [
    # clock
    "Clock",
    "LiveClock",
    "SimulatedClock",
    # events
    "BaseEvent",
    "BarEvent",
    "FXRateEvent",
    "HongSusmelSignalEvent",
    "RiskDecision",
    "OrderRequest",
    "FillEvent",
    "PositionUpdateEvent",
    "RoceRuceResult",
    # instruments
    "AsianADRApprovedPair",
    "AsianADRPairSpec",
    "ASIAN_EXCHANGES",
    "STANDARD_ADR_RATIOS",
    # types: scalars
    "PairId",
    "Ticker",
    "CurrencyCode",
    "ExchangeCode",
    "OrderId",
    "FillId",
    "ADRRatio",
    # types: literals / enums
    "Side",
    "Venue",
    "BUY",
    "SELL",
    "US_EQUITY",
    "FOREIGN_EQUITY",
    "HSSignal",
    "HSPosition",
    "RiskDecisionStatus",
    "OrderType",
    "LiquidityBucket",
    # exceptions
    "AsianADRError",
    "ConfigurationError",
    "DataError",
    "StaleDataError",
    "MissingBarError",
    "MissingFXRateError",
    "PairRegistryError",
    "PairNotFoundError",
    "ADRRatioUnresolvedError",
    "RiskError",
    "RiskRuleViolation",
    "KillSwitchTriggered",
    "ExecutionError",
    "SequencerStateError",
    "OvernightAbort",
    "GatewayError",
    "ShortLocateUnavailable",
    "OrderRejected",
]
