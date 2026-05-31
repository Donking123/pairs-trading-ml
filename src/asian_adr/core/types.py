"""Domain type aliases and enumerations.

This module defines the shared vocabulary used across every component:

* Branded scalar aliases (``PairId``, ``Ticker``, ``CurrencyCode`` …) that make
  function signatures self-documenting without imposing runtime cost.
* The closed enumerations referenced by the event contracts in §7 of the
  architecture spec (``HSSignal``, ``RiskDecisionStatus``, ``OrderType``,
  ``LiquidityBucket``).
* ``Side`` / ``Venue`` literal aliases that mirror the broker-facing contracts.

Only the standard library is imported here — ``core`` carries no third-party
dependency other than pydantic (which is not needed in this file).
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Literal, NewType

__all__ = [
    # Branded scalars
    "PairId",
    "Ticker",
    "CurrencyCode",
    "ExchangeCode",
    "OrderId",
    "FillId",
    "ADRRatio",
    # Literal aliases
    "Side",
    "Venue",
    "BUY",
    "SELL",
    "US_EQUITY",
    "FOREIGN_EQUITY",
    # Enumerations
    "HSSignal",
    "HSPosition",
    "RiskDecisionStatus",
    "OrderType",
    "LiquidityBucket",
]


# ---------------------------------------------------------------------------
# Branded scalar aliases
# ---------------------------------------------------------------------------
# ``NewType`` gives static type-checkers a distinct nominal type while
# remaining a plain ``str`` / ``Decimal`` at runtime (no boxing, no overhead).

PairId = NewType("PairId", str)
"""Stable identifier for an approved ADR / underlying pair, e.g. ``"TM_7203.T"``."""

Ticker = NewType("Ticker", str)
"""A tradable symbol — either the U.S. ADR or its Asian underlying."""

CurrencyCode = NewType("CurrencyCode", str)
"""ISO-4217 currency code, e.g. ``"USD"``, ``"JPY"``, ``"HKD"``."""

ExchangeCode = NewType("ExchangeCode", str)
"""Exchange mnemonic, e.g. ``"NYSE"``, ``"TSE"``, ``"HKEX"``, ``"KRX"``."""

OrderId = NewType("OrderId", str)
"""Internal client-side order identifier."""

FillId = NewType("FillId", str)
"""Broker-assigned execution / fill identifier."""

ADRRatio = NewType("ADRRatio", Decimal)
"""Local shares represented by one ADR — a fixed structural constant.

The ratio is **never** OLS-estimated (architecture rule ADR-002); it is the
``β`` used in the dollar-spread reconstruction.
"""


# ---------------------------------------------------------------------------
# Literal aliases (broker-facing contracts)
# ---------------------------------------------------------------------------

Side = Literal["buy", "sell"]
Venue = Literal["us_equity", "foreign_equity"]

# Named constants so call sites can avoid bare string literals.
BUY: Side = "buy"
SELL: Side = "sell"
US_EQUITY: Venue = "us_equity"
FOREIGN_EQUITY: Venue = "foreign_equity"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class HSSignal(str, Enum):
    """Direction of a Hong & Susmel signal.

    The strategy is one-sided by design: Asian short-selling restrictions
    eliminate the symmetric trade, so there is deliberately **no** ``LONG_ADR``
    member.
    """

    SHORT_ADR = "short_adr"      # SELL ADR short; buy local at next Asia open
    EXIT = "exit"                # spread converged — close both legs
    FORCE_CLOSE = "force_close"  # holding period expired — hard close


class HSPosition(str, Enum):
    """Per-pair position state held inside the Hong & Susmel engine."""

    FLAT = "flat"
    OPEN = "open"


class RiskDecisionStatus(str, Enum):
    """Outcome of the pre-trade risk evaluation for a signal."""

    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


class OrderType(str, Enum):
    """Supported order types routed to the broker gateway."""

    MARKET = "market"
    LIMIT = "limit"
    MARKET_ON_CLOSE = "market_on_close"
    MARKET_ON_OPEN = "market_on_open"


class LiquidityBucket(str, Enum):
    """Liquidity tier used for post-hoc ROCE/RUCE attribution (paper §2.5).

    Illiquidity rises from ``HIGH`` to ``LOW``; the paper documents higher
    median profit in less-liquid buckets (limits-to-arbitrage premium).
    """

    HIGH = "high"
    HIGH_MEDIUM = "high_medium"
    MEDIUM_LOW = "medium_low"
    LOW = "low"
