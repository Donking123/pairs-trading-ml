"""Domain exception hierarchy.

Every error raised inside the platform derives from :class:`AsianADRError`,
so a single ``except AsianADRError`` at a runner boundary catches all
strategy-specific failures while letting genuine programming errors
(``KeyError``, ``TypeError`` …) propagate.

The hierarchy mirrors the layered architecture:

    AsianADRError
    ├── ConfigurationError
    ├── DataError
    │   ├── StaleDataError
    │   ├── MissingBarError
    │   └── MissingFXRateError
    ├── PairRegistryError
    │   ├── PairNotFoundError
    │   └── ADRRatioUnresolvedError
    ├── RiskError
    │   ├── RiskRuleViolation
    │   └── KillSwitchTriggered
    ├── ExecutionError
    │   ├── SequencerStateError
    │   └── OvernightAbort
    └── GatewayError
        ├── ShortLocateUnavailable
        └── OrderRejected

Only the standard library is used here.
"""

from __future__ import annotations

__all__ = [
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


class AsianADRError(Exception):
    """Base class for every error raised by the platform."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigurationError(AsianADRError):
    """A required setting is missing, malformed, or mutually inconsistent."""


# ---------------------------------------------------------------------------
# Data / feed
# ---------------------------------------------------------------------------


class DataError(AsianADRError):
    """Base class for market-data, FX, and reference-data problems."""


class StaleDataError(DataError):
    """A price or FX value is older than its permitted freshness window.

    Per the graceful-degradation principle the engine usually *skips* the
    affected computation rather than raising; this exception exists for the
    paths that must hard-fail (e.g. a pair held past the FX staleness ceiling).
    """


class MissingBarError(DataError):
    """No bar is available for a ticker on a date where one was expected."""


class MissingFXRateError(DataError):
    """No FX rate is cached for a currency required to convert a local price."""


# ---------------------------------------------------------------------------
# Pair registry / research
# ---------------------------------------------------------------------------


class PairRegistryError(AsianADRError):
    """Base class for pair-registry lookup and validation failures."""


class PairNotFoundError(PairRegistryError):
    """A referenced ``pair_id`` is not present in the active registry."""


class ADRRatioUnresolvedError(PairRegistryError):
    """The ADR conversion ratio could not be resolved (NULL or non-positive).

    WRDS reference data never carries the ratio, so it must be estimated or
    injected externally; an unresolved ratio means the pair is rejected /
    suspended rather than traded with a guessed hedge.
    """


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


class RiskError(AsianADRError):
    """Base class for risk-management failures."""


class RiskRuleViolation(RiskError):
    """A pre-trade risk rule blocked an order.

    Parameters
    ----------
    rule:
        Name of the rule that produced the block.
    reason:
        Human-readable explanation suitable for audit logs and alerts.
    """

    def __init__(self, rule: str, reason: str) -> None:
        self.rule = rule
        self.reason = reason
        super().__init__(f"{rule}: {reason}")


class KillSwitchTriggered(RiskError):
    """The global kill switch has fired; no new orders may be submitted."""


# ---------------------------------------------------------------------------
# Execution / sequencing
# ---------------------------------------------------------------------------


class ExecutionError(AsianADRError):
    """Base class for order-sequencing and execution failures."""


class SequencerStateError(ExecutionError):
    """An event arrived that is invalid for the sequencer's current phase."""


class OvernightAbort(ExecutionError):
    """The local leg was abandoned after an overnight spread reversal.

    Raised (or surfaced as an event) when an ADR short filled but the spread
    reverted below ``kappa_close`` before the Asian open, so the naked short
    is covered and no local leg is ever placed.
    """


# ---------------------------------------------------------------------------
# Gateway / broker
# ---------------------------------------------------------------------------


class GatewayError(AsianADRError):
    """Base class for broker-gateway failures."""


class ShortLocateUnavailable(GatewayError):
    """No borrow / locate is available for an ADR short sale."""


class OrderRejected(GatewayError):
    """The broker rejected an order request.

    Parameters
    ----------
    ticker:
        Symbol of the rejected order.
    reason:
        Broker-supplied or synthesised rejection reason.
    """

    def __init__(self, ticker: str, reason: str) -> None:
        self.ticker = ticker
        self.reason = reason
        super().__init__(f"order rejected for {ticker}: {reason}")
