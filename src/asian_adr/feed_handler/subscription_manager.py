"""Registry-driven subscription management (architecture §3.2).

Tracks the set of tickers a feed handler should be subscribed to, rebuilding it
whenever a ``PairRegistryUpdateEvent`` arrives. The registry event is
duck-typed (``event.registry``) so this package depends on neither the research
nor strategy layers.

A handler serves one leg — the ADR (U.S.) side or the underlying (Asian) side —
selected by ``role``.

Depends only on the standard library.
"""

from __future__ import annotations

import logging
from typing import Iterable, Literal

logger = logging.getLogger(__name__)

__all__ = ["SubscriptionManager", "Role"]

Role = Literal["adr", "underlying"]


class SubscriptionManager:
    """Maintains the active ticker set for one feed leg."""

    def __init__(self, role: Role, initial_tickers: Iterable[str] = ()) -> None:
        self._role = role
        self._tickers: set[str] = set(initial_tickers)

    @property
    def role(self) -> Role:
        return self._role

    @property
    def tickers(self) -> frozenset[str]:
        return frozenset(self._tickers)

    def set_tickers(self, tickers: Iterable[str]) -> None:
        self._tickers = set(tickers)

    def _select(self, pair) -> str:
        return pair.adr_ticker if self._role == "adr" else pair.underlying_ticker

    def update_from_registry(self, event: object) -> tuple[set[str], set[str]]:
        """Rebuild the ticker set from a registry update.

        Returns ``(added, removed)`` so the handler can resubscribe minimally.
        Accepts either a ``PairRegistryUpdateEvent`` (``.registry``) or a raw
        iterable of pairs.
        """
        pairs = getattr(event, "registry", event)
        new = {self._select(p) for p in pairs}
        added = new - self._tickers
        removed = self._tickers - new
        self._tickers = new
        if added or removed:
            logger.info(
                "subscription %s update: +%d -%d (now %d)",
                self._role, len(added), len(removed), len(new),
            )
        return added, removed
