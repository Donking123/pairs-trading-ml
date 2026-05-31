"""Live FX rate feed (architecture §3.3).

Fetches daily FX close rates for the registry's currencies (OANDA REST live;
the Parquet loader handles backtest), normalises each to a ``FXRateEvent``
(USD per 1 unit of base), publishes to ``fx-rates``, and falls back to the prior
day's rate with ``is_stale=True`` when a rate is missing — emitting staleness /
suspend alerts. FX is conversion-only; no hedging.

Each handler is a ``Producer`` ready for ``LiveRunner.add_producer``. Imports
only ``core`` and ``event_bus``; the OANDA HTTP client is imported lazily.
"""

from __future__ import annotations

from .connectors import (
    USD_QUOTE_CCYS,
    OANDAConnector,
    OANDAFXHandler,
    oanda_instrument,
)
from .handler import AbstractFXConnector, FXRateHandler, IterableFXConnector
from .normalizer import FXNormalizer, OANDANormalizer
from .rate_cache import FXRateCache
from .staleness_monitor import FXStaleEvent, FXStalenessMonitor, business_days_between

__all__ = [
    "FXRateCache",
    "FXRateHandler",
    "AbstractFXConnector",
    "IterableFXConnector",
    "FXNormalizer",
    "OANDANormalizer",
    "FXStalenessMonitor",
    "FXStaleEvent",
    "business_days_between",
    "OANDAConnector",
    "OANDAFXHandler",
    "oanda_instrument",
    "USD_QUOTE_CCYS",
]
