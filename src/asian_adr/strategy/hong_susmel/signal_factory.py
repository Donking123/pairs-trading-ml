"""Constructs immutable :class:`HongSusmelSignalEvent` objects.

Centralising signal construction keeps the threshold geometry (``κ_open``,
``κ_close``, ``z``) in one place and guarantees every emitted signal carries a
consistent set of fields. The dollar spread, ``µ`` and ``σ`` are computed by the
engine; the factory derives the thresholds and z-score and stamps timestamps.

Depends only on the standard library and ``asian_adr.core``.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from asian_adr.core import (
    AsianADRApprovedPair,
    Clock,
    HongSusmelSignalEvent,
    HSSignal,
)

__all__ = ["SignalFactory"]


class SignalFactory:
    """Builds signal events with derived thresholds and z-score."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    def build(
        self,
        pair: AsianADRApprovedPair,
        signal: HSSignal,
        spread: Decimal,
        mu: Decimal,
        sigma: Decimal,
        timestamp_exchange: datetime,
        days_held: int = 0,
    ) -> HongSusmelSignalEvent:
        """Assemble a :class:`HongSusmelSignalEvent`.

        ``κ_open = µ + k0·σ`` and ``κ_close = µ + kc·σ``; ``z = (spread − µ)/σ``.
        A zero ``σ`` (degenerate flat window) yields ``z = 0`` rather than a
        division error — such a pair cannot breach a threshold anyway.
        """
        z_score = (spread - mu) / sigma if sigma != 0 else Decimal(0)
        return HongSusmelSignalEvent(
            timestamp_exchange=timestamp_exchange,
            timestamp_received=self._clock.now(),
            pair_id=pair.pair_id,
            adr_ticker=pair.adr_ticker,
            underlying_ticker=pair.underlying_ticker,
            signal=signal,
            spread=spread,
            mu=mu,
            sigma=sigma,
            kappa_open=mu + pair.k0 * sigma,
            kappa_close=mu + pair.kc * sigma,
            z_score=z_score,
            days_held=days_held,
        )
