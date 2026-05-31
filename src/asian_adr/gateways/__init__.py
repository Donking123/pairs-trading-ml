"""Broker gateways — U.S. ADR execution (architecture §3.10).

Consume ``OrderRequest`` (ADR short SELL / cover BUY), verify short-locate,
submit to the broker, and publish ``FillEvent``. Interactive Brokers is the
primary gateway; Alpaca is the commission-free / paper fallback; the simulation
gateway is for backtest and paper trading. No live foreign-equity gateway — the
Asian leg is executed via the operator's own brokerage.

Each gateway is a ``Producer`` (``run()``) and exposes ``on_order`` for the
``orders`` topic. Broker SDKs (``ib_insync``, ``httpx``, ``websockets``) are
imported lazily, so this package is import-safe without them. Imports only
``core``, ``event_bus``, and (for the simulation gateway) ``backtest``.
"""

from __future__ import annotations

from .alpaca import AlpacaRestGateway, AlpacaWsGateway
from .base import (
    AbstractGateway,
    AlwaysAvailableLocate,
    LocateResult,
    OrderRejectedEvent,
    ShortLocateProvider,
    reconnect_with_backoff,
)
from .interactive_brokers import IBShortLocate, InteractiveBrokersGateway
from .simulation import SimulatedForeignGateway, SimulatedUSGateway

__all__ = [
    "AbstractGateway",
    "LocateResult",
    "ShortLocateProvider",
    "AlwaysAvailableLocate",
    "OrderRejectedEvent",
    "reconnect_with_backoff",
    "InteractiveBrokersGateway",
    "IBShortLocate",
    "AlpacaRestGateway",
    "AlpacaWsGateway",
    "SimulatedUSGateway",
    "SimulatedForeignGateway",
]
