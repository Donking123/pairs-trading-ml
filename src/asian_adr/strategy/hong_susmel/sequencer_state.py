"""State for the Asian execution sequencer.

Defines the per-pair phase enum and mutable record the sequencer advances as it
bridges the overnight gap, plus the alert event emitted when an overnight
reversal forces a naked-short abort.

Depends only on the standard library and ``asian_adr.core``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal

from asian_adr.core import BaseEvent, FillEvent, HongSusmelSignalEvent

__all__ = ["SeqPhase", "SequencerState", "AdrOvernightAbortEvent"]


class SeqPhase(str, Enum):
    """Lifecycle of one pair inside the sequencer.

    ``IDLE`` → ``AWAITING_LOCAL`` (ADR short submitted) → ``AWAITING_ASIA_OPEN``
    (ADR fill confirmed) → ``OPEN`` (local leg established) → ``IDLE`` (closed).
    An overnight reversal sends ``AWAITING_ASIA_OPEN`` straight back to ``IDLE``
    (abort, naked short covered).
    """

    IDLE = "idle"
    AWAITING_LOCAL = "awaiting_local"
    AWAITING_ASIA_OPEN = "awaiting_asia_open"
    OPEN = "open"


@dataclass
class SequencerState:
    """Mutable per-pair sequencer record."""

    pair_id: str
    adr_ticker: str
    underlying_ticker: str
    phase: SeqPhase = SeqPhase.IDLE
    adr_fill: FillEvent | None = None
    signal_event: HongSusmelSignalEvent | None = None

    def transition(
        self,
        new_phase: SeqPhase,
        *,
        signal_event: HongSusmelSignalEvent | None = None,
    ) -> None:
        self.phase = new_phase
        if signal_event is not None:
            self.signal_event = signal_event

    def reset(self) -> None:
        """Return to IDLE and clear the round-trip's working state."""
        self.phase = SeqPhase.IDLE
        self.adr_fill = None
        self.signal_event = None


class AdrOvernightAbortEvent(BaseEvent):
    """Published to ``alerts`` when a naked ADR short is covered after a reversal.

    Raised when the ADR leg filled but the spread reverted below ``κ_close``
    before the Asian open, so no local leg is ever placed and the short is
    covered — preventing one-sided overnight inventory.
    """

    event_type: Literal["adr_overnight_abort"] = "adr_overnight_abort"
    pair_id: str
    reason: str = "overnight_spread_reversal"


def _now_placeholder() -> datetime:  # pragma: no cover - helper for callers
    """Convenience for constructing the abort event when no clock is handy."""
    from datetime import timezone

    return datetime.now(timezone.utc)
