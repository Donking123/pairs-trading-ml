"""Point-in-time pair-registry loader (architecture §3.12, §4).

Loads the approved-pair registry JSON once and serves *as-of* snapshots so the
backtest never sees a pair before its ``approved_date`` or after its
``expiry_date`` — enforcing point-in-time correctness (no look-ahead on pair
selection). The :class:`PairRegistryUpdateEvent` is published on the
``pair-registry`` topic whenever the active set changes on a date rollover.

Uses only the standard library and ``asian_adr.core``.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Iterable, Literal

from asian_adr.core import AsianADRApprovedPair, BaseEvent

logger = logging.getLogger(__name__)

__all__ = ["PairRegistryUpdateEvent", "PairRegistryLoader"]


class PairRegistryUpdateEvent(BaseEvent):
    """Published when the active pair registry changes (point-in-time)."""

    event_type: Literal["pair_registry_update"] = "pair_registry_update"
    registry: list[AsianADRApprovedPair]


class PairRegistryLoader:
    """Serves as-of snapshots of the approved-pair registry."""

    def __init__(self, pairs: Iterable[AsianADRApprovedPair]) -> None:
        self._all: list[AsianADRApprovedPair] = list(pairs)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "PairRegistryLoader":
        raw = json.loads(Path(path).read_text())
        entries = raw if isinstance(raw, list) else raw.get("pairs", [])
        pairs: list[AsianADRApprovedPair] = []
        for entry in entries:
            try:
                pairs.append(AsianADRApprovedPair.model_validate(entry))
            except Exception as exc:  # skip malformed / unresolved-ratio pairs
                logger.warning("skipping pair %s: %s", entry.get("pair_id"), exc)
        logger.info("loaded %d pairs from %s", len(pairs), path)
        return cls(pairs)

    # -- queries -------------------------------------------------------------

    def load_as_of(self, as_of: date) -> list[AsianADRApprovedPair]:
        """Active pairs whose validity window contains ``as_of``."""
        return [p for p in self._all if p.is_active_as_of(as_of)]

    @property
    def all_pairs(self) -> list[AsianADRApprovedPair]:
        return list(self._all)

    def all_adr_tickers(self) -> set[str]:
        return {p.adr_ticker for p in self._all}

    def all_underlying_tickers(self) -> set[str]:
        return {p.underlying_ticker for p in self._all}

    def all_currencies(self) -> set[str]:
        return {p.underlying_currency for p in self._all}
