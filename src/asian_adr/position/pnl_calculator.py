"""Average-cost position accounting and multi-currency PnL math.

A :class:`LegPosition` holds one leg (the ADR or the local underlying) of a
pair. :class:`PnLCalculator` applies fills with average-cost basis and computes
realised / unrealised PnL **in USD**, which is how everything downstream
(risk, tearsheet) consumes it.

Sign convention: ``net_quantity`` is signed — positive long, negative short —
so a single formula handles both directions::

    unrealized_usd = (mark_usd − avg_entry_price_usd) × net_quantity

For a short (``net_quantity < 0``) a falling mark makes ``(mark − entry)``
negative, which times the negative quantity yields a positive profit.

Depends only on the standard library and ``asian_adr.core``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from asian_adr.core import FillEvent

__all__ = ["LegPosition", "PnLCalculator"]


@dataclass
class LegPosition:
    """Mutable state for one leg of a pair."""

    pair_id: str
    ticker: str
    venue: str
    currency: str
    net_quantity: Decimal = Decimal("0")
    avg_entry_price: Decimal = Decimal("0")       # native currency
    avg_entry_price_usd: Decimal = Decimal("0")   # USD at entry
    realized_pnl_usd: Decimal = Decimal("0")
    last_mark: Decimal | None = None              # native
    last_fx: Decimal = Decimal("1")               # USD per 1 unit native

    @property
    def is_short(self) -> bool:
        return self.net_quantity < 0

    @property
    def is_flat(self) -> bool:
        return self.net_quantity == 0


class PnLCalculator:
    """Stateless helpers that mutate / read :class:`LegPosition`."""

    @staticmethod
    def _fees_usd(fill: FillEvent) -> Decimal:
        fees_native = (
            fill.commission + fill.sec_fee + fill.stamp_duty + fill.short_borrow_fee
        )
        if fill.commission_currency == "USD" or fill.fx_rate_used is None:
            return fees_native
        return fees_native * fill.fx_rate_used

    @classmethod
    def apply_fill(cls, leg: LegPosition, fill: FillEvent) -> None:
        """Update ``leg`` for ``fill`` with average-cost basis and realised PnL."""
        delta = fill.fill_quantity if fill.side == "buy" else -fill.fill_quantity
        q0 = leg.net_quantity
        price = fill.fill_price
        price_usd = fill.fill_price_usd

        if q0 == 0 or (q0 > 0) == (delta > 0):
            # opening or adding in the same direction → re-weight the average
            new_abs = abs(q0) + abs(delta)
            leg.avg_entry_price = (
                leg.avg_entry_price * abs(q0) + price * abs(delta)
            ) / new_abs
            leg.avg_entry_price_usd = (
                leg.avg_entry_price_usd * abs(q0) + price_usd * abs(delta)
            ) / new_abs
            leg.net_quantity = q0 + delta
        else:
            # reducing / closing / flipping → realise PnL on the closed quantity
            closed = min(abs(delta), abs(q0))
            if q0 > 0:  # long reduced by a sell
                leg.realized_pnl_usd += (price_usd - leg.avg_entry_price_usd) * closed
            else:       # short reduced by a buy (cover)
                leg.realized_pnl_usd += (leg.avg_entry_price_usd - price_usd) * closed
            leg.net_quantity = q0 + delta
            if leg.net_quantity == 0:
                leg.avg_entry_price = Decimal("0")
                leg.avg_entry_price_usd = Decimal("0")
            elif (q0 > 0) != (leg.net_quantity > 0):
                # flipped through zero → new basis at the fill price
                leg.avg_entry_price = price
                leg.avg_entry_price_usd = price_usd
            # otherwise the average basis of the remaining quantity is unchanged

        leg.realized_pnl_usd -= cls._fees_usd(fill)
        leg.last_mark = price
        leg.last_fx = Decimal("1") if fill.venue == "us_equity" else (
            fill.fx_rate_used if fill.fx_rate_used is not None else leg.last_fx
        )

    @staticmethod
    def unrealized_pnl_usd(leg: LegPosition, mark_native: Decimal, fx: Decimal) -> Decimal:
        mark_usd = mark_native * fx
        return (mark_usd - leg.avg_entry_price_usd) * leg.net_quantity

    @staticmethod
    def notional_usd(leg: LegPosition, mark_native: Decimal, fx: Decimal) -> Decimal:
        return abs(leg.net_quantity) * mark_native * fx
