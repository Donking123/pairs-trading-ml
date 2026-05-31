"""Live market-data feed handlers (architecture §3.2).

Connects to provider feeds (Polygon/Alpaca for U.S. ADRs, an Asian provider for
underlyings), normalises every message to a ``BarEvent`` before it enters the
bus, rebuilds subscriptions from ``PairRegistryUpdateEvent``s, and flags
zero-return days / feed gaps as alerts. Each handler is a ``Producer`` (``async
run()``) ready to plug into the live runner via ``add_producer``.

Imports only ``core`` and ``event_bus`` — never ``strategy`` / ``risk`` /
``gateways``. Provider SDKs are imported lazily, so this package is import-safe
without them.
"""

from __future__ import annotations

from .connector import (
    AbstractConnector,
    FeedHandler,
    IterableConnector,
    reconnect_with_backoff,
)
from .connectors import (
    AlpacaConnector,
    AlpacaFeedHandler,
    AsianConnector,
    AsianFeedHandler,
    PolygonConnector,
    PolygonFeedHandler,
    ticker_meta_from_pairs,
)
from .normalizer import (
    AlpacaNormalizer,
    AsianNormalizer,
    BarNormalizer,
    PolygonNormalizer,
    build_bar,
)
from .staleness_monitor import FeedGapEvent, StalenessMonitor, ZeroReturnEvent
from .subscription_manager import SubscriptionManager

__all__ = [
    # orchestration
    "FeedHandler",
    "AbstractConnector",
    "IterableConnector",
    "reconnect_with_backoff",
    # normalizers
    "BarNormalizer",
    "build_bar",
    "PolygonNormalizer",
    "AlpacaNormalizer",
    "AsianNormalizer",
    # subscription / staleness
    "SubscriptionManager",
    "StalenessMonitor",
    "ZeroReturnEvent",
    "FeedGapEvent",
    # connectors / handlers
    "PolygonConnector",
    "PolygonFeedHandler",
    "AlpacaConnector",
    "AlpacaFeedHandler",
    "AsianConnector",
    "AsianFeedHandler",
    "ticker_meta_from_pairs",
]
