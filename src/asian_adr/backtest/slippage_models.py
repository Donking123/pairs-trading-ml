"""Slippage models for simulated fills (architecture §9.2).

* :class:`HalfSpreadSlippageModel` — U.S. ADR leg; a fraction of the bar's
  high-low range proxies the half-spread, applied against the trade direction.
* :class:`SqrtImpactSlippageModel` — Asian underlying leg; square-root of the
  participation rate models market impact.

Both return an execution price in the bar's native currency.

Depends only on the standard library and ``asian_adr.core``.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol, runtime_checkable

from asian_adr.core import BarEvent, Side

__all__ = ["SlippageModel", "HalfSpreadSlippageModel", "SqrtImpactSlippageModel"]


@runtime_checkable
class SlippageModel(Protocol):
    def calculate(self, bar: BarEvent, side: Side, qty: Decimal) -> Decimal: ...


class HalfSpreadSlippageModel:
    """U.S. ADR — estimated half-spread from the OHLCV bar."""

    def __init__(self, spread_fraction: Decimal = Decimal("0.1")) -> None:
        self._spread_fraction = spread_fraction

    def calculate(self, bar: BarEvent, side: Side, qty: Decimal) -> Decimal:
        estimated_spread = (bar.high - bar.low) * self._spread_fraction
        direction = Decimal(1) if side == "buy" else Decimal(-1)
        return bar.close + direction * estimated_spread / Decimal(2)


class SqrtImpactSlippageModel:
    """Asian underlying — sqrt of participation rate as market impact."""

    def __init__(self, impact_coefficient: Decimal = Decimal("0.1")) -> None:
        self._impact = impact_coefficient

    def calculate(self, bar: BarEvent, side: Side, qty: Decimal) -> Decimal:
        adv = Decimal(bar.volume) * bar.close
        if adv == 0:
            return bar.close
        participation = qty * bar.close / adv
        direction = Decimal(1) if side == "buy" else Decimal(-1)
        return bar.close * (1 + direction * self._impact * participation.sqrt())
