"""Simulated foreign (Asian) gateway — backtest only (architecture §3.10).

There is no live foreign gateway: the local Asian leg is executed via the
operator's own brokerage in production. For backtest/paper, local fills are
handled by the backtest :class:`SimulatedForeignExchange`, which this module
re-exports under the gateway name so the structure is complete without
duplicating the fill logic.
"""

from __future__ import annotations

from asian_adr.backtest import SimulatedForeignExchange as SimulatedForeignGateway

__all__ = ["SimulatedForeignGateway"]
