"""Immutable event contracts (architecture spec §7).

Every message that crosses the event bus is a **frozen** pydantic model
deriving from :class:`BaseEvent`. Immutability is a hard invariant: mutable
state lives only inside engines, never inside the events they exchange.

All monetary and price quantities are ``Decimal`` to avoid binary
floating-point drift in spread and PnL arithmetic.

This module depends only on the standard library and pydantic.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from .types import (
    HSSignal,
    LiquidityBucket,
    OrderType,
    RiskDecisionStatus,
    Side,
    Venue,
)

__all__ = [
    "BaseEvent",
    "BarEvent",
    "FXRateEvent",
    "HongSusmelSignalEvent",
    "RiskDecision",
    "OrderRequest",
    "FillEvent",
    "PositionUpdateEvent",
    "RoceRuceResult",
    # re-exported enums for convenience
    "HSSignal",
    "RiskDecisionStatus",
    "OrderType",
    "LiquidityBucket",
]


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------


class BaseEvent(BaseModel):
    """Common envelope for every event on the bus.

    The three timestamps capture the canonical latency path: when the
    exchange stamped the bar (``timestamp_exchange``), when the platform
    ingested it (``timestamp_received``), and when a component finished
    handling it (``timestamp_processed``, set downstream).
    """

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    event_type: str
    timestamp_exchange: datetime
    timestamp_received: datetime
    timestamp_processed: datetime | None = None


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


class BarEvent(BaseEvent):
    """A single daily OHLCV bar for an ADR or its Asian underlying."""

    event_type: Literal["bar"] = "bar"
    ticker: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    bar_interval: str = "1d"
    is_adjusted: bool
    exchange: str            # "NYSE", "TSE", "HKEX", "KRX", …
    currency: str            # "USD", "JPY", "HKD", "KRW", …


class FXRateEvent(BaseEvent):
    """Daily FX close used only for USD conversion of the local price.

    ``mid`` is quoted as **USD per 1 unit of the base currency**. There is no
    FX hedge anywhere in the system — the overnight gap makes simultaneous FX
    submission impossible, so FX is conversion-only.
    """

    event_type: Literal["fx_rate"] = "fx_rate"
    currency_pair: str       # e.g. "JPY_USD"
    base_currency: str       # "JPY"
    quote_currency: str      # "USD"
    mid: Decimal             # USD per 1 unit of base currency
    provider: str            # "datastream" (historical) | "oanda" (live)
    is_stale: bool = False


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


class HongSusmelSignalEvent(BaseEvent):
    """A trading signal emitted by the Hong & Susmel engine.

    Direction is permanently one-sided (``SHORT_ADR`` only on entry); ``EXIT``
    and ``FORCE_CLOSE`` close an open pair. ``days_held`` is ``0`` for entry
    signals.
    """

    event_type: Literal["hs_signal"] = "hs_signal"
    pair_id: str
    adr_ticker: str
    underlying_ticker: str
    signal: HSSignal
    spread: Decimal          # dollar spread at the signal bar
    mu: Decimal              # rolling mean
    sigma: Decimal           # rolling std (ddof=1)
    kappa_open: Decimal      # µ + k0·σ
    kappa_close: Decimal     # µ + kc·σ
    z_score: Decimal         # (spread − µ) / σ
    days_held: int = 0


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


class RiskDecision(BaseEvent):
    """Outcome of the pre-trade risk evaluation for one signal."""

    event_type: Literal["risk_decision"] = "risk_decision"
    signal_id: UUID
    pair_id: str
    status: RiskDecisionStatus
    approved_notional: Decimal | None = None
    rejected_reason: str | None = None
    risk_rule_results: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Orders & fills
# ---------------------------------------------------------------------------


class OrderRequest(BaseEvent):
    """An order instruction routed from the sequencer to a broker gateway."""

    event_type: Literal["order_request"] = "order_request"
    pair_id: str
    risk_decision_id: UUID
    ticker: str
    side: Side
    quantity: Decimal
    order_type: OrderType
    limit_price: Decimal | None = None
    venue: Venue
    currency: str
    is_short_sale: bool = False
    time_in_force: str = "DAY"


class FillEvent(BaseEvent):
    """A confirmed execution, carrying native- and USD-denominated prices.

    ``fill_price`` is in the instrument's native currency; ``fill_price_usd``
    is the USD-converted price the ROCE/RUCE calculator consumes directly.
    """

    event_type: Literal["fill"] = "fill"
    fill_id: str
    client_order_id: UUID
    broker_order_id: str
    pair_id: str | None = None
    ticker: str
    side: Side
    fill_price: Decimal           # native currency
    fill_price_usd: Decimal       # USD-converted
    fill_quantity: Decimal
    remaining_quantity: Decimal
    commission: Decimal
    commission_currency: str
    sec_fee: Decimal = Decimal("0")
    stamp_duty: Decimal = Decimal("0")
    short_borrow_fee: Decimal = Decimal("0")
    fx_rate_used: Decimal | None = None
    is_short_sale: bool = False
    venue: Venue
    exchange: str
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


class PositionUpdateEvent(BaseEvent):
    """Mark-to-market snapshot for one leg of a pair, published on any change."""

    event_type: Literal["position_update"] = "position_update"
    pair_id: str
    ticker: str
    venue: Venue
    net_quantity: Decimal
    average_entry_price: Decimal
    average_entry_price_usd: Decimal
    unrealized_pnl_usd: Decimal
    realized_pnl_usd: Decimal
    mark_price: Decimal
    mark_price_usd: Decimal
    notional_value_usd: Decimal
    fx_rate: Decimal | None = None
    is_short: bool
    trigger: Literal["fill", "mark_to_market", "fx_update", "reconciliation"]


# ---------------------------------------------------------------------------
# Performance result
# ---------------------------------------------------------------------------


class RoceRuceResult(BaseModel):
    """Per-trade ROCE / RUCE result (paper §2.4).

    ``ruce`` doubles the ADR-leg contribution to reflect the Reg-T 50% margin
    on the short ADR leg. Net figures subtract the Roll (1984) round-trip cost
    estimate. Not an event — produced by the calculator and aggregated into
    the tearsheet — but frozen for the same immutability guarantee.
    """

    model_config = ConfigDict(frozen=True)

    pair_id: str
    trade_open_date: date
    trade_close_date: date
    duration_days: int
    local_return: Decimal
    adr_return: Decimal
    roce: Decimal
    ruce: Decimal
    roll_cost_pct: Decimal       # estimated round-trip cost (Roll 1984)
    roce_net: Decimal            # roce − roll_cost_pct
    ruce_net: Decimal            # ruce − roll_cost_pct
    liquidity_bucket: LiquidityBucket
    was_force_closed: bool = False
    was_aborted: bool = False    # local leg never established (overnight reversal)
