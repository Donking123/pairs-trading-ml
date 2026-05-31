"""In-process simulated U.S. gateway for paper trading (architecture §3.10).

Wraps the backtest :class:`SimulatedUSExchange` (DRY — same fill logic the
backtest uses) behind the gateway contract: it runs the short-locate check from
:class:`AbstractGateway`, then delegates the fill to the wrapped exchange, which
fills at the most recent bar with slippage + the U.S. cost model and publishes
the :class:`FillEvent`.

Use it to replace ``LiveRunner.enable_simulated_execution`` with a gateway-shaped
component, or for paper trading without a broker connection.
"""

from __future__ import annotations

from asian_adr.backtest import EquitiesCostModel, SimulatedUSExchange
from asian_adr.backtest.slippage_models import SlippageModel
from asian_adr.core import BarEvent, Clock, OrderRequest
from asian_adr.event_bus import AbstractEventBus

from ..base import AbstractGateway, ShortLocateProvider

__all__ = ["SimulatedUSGateway"]


class SimulatedUSGateway(AbstractGateway):
    """Paper-trading U.S. ADR gateway backed by the simulated exchange."""

    def __init__(
        self,
        bus: AbstractEventBus,
        clock: Clock,
        *,
        cost_model: EquitiesCostModel | None = None,
        slippage_model: SlippageModel | None = None,
        short_locate: ShortLocateProvider | None = None,
    ) -> None:
        super().__init__(bus, clock, short_locate=short_locate, name="sim_us")
        self._exchange = SimulatedUSExchange(
            bus, clock, cost_model or EquitiesCostModel(), slippage_model
        )

    async def on_bar(self, bar: BarEvent) -> None:
        """Track the latest bar for fill pricing (wire to ``market-data``)."""
        await self._exchange.on_bar(bar)

    async def submit(self, order: OrderRequest) -> None:
        # Locate already verified by AbstractGateway.on_order; delegate the fill.
        await self._exchange.on_order(order)
