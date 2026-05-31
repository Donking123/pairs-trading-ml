"""Hong & Susmel strategy: signal engine, state, and overnight execution.

Exposes the signal-generation engine, its per-pair state and rolling stats, the
signal factory, the liquidity-bucket classifier, and the Asian execution
sequencer with its phase model. Imports only from ``core`` and ``event_bus`` —
never from ``risk``, ``position``, or ``gateways``.
"""

from __future__ import annotations

from .engine import HongSusmelEngine
from .execution_sequencer import AsianExecutionSequencer
from .liquidity_bucket import (
    assign_bucket,
    assign_by_adv,
    assign_by_zero_return,
)
from .sequencer_state import AdrOvernightAbortEvent, SeqPhase, SequencerState
from .signal_factory import SignalFactory
from .state import HSPairState, WelfordRollingStats

__all__ = [
    "HongSusmelEngine",
    "AsianExecutionSequencer",
    "SignalFactory",
    "HSPairState",
    "WelfordRollingStats",
    "SeqPhase",
    "SequencerState",
    "AdrOvernightAbortEvent",
    "assign_bucket",
    "assign_by_zero_return",
    "assign_by_adv",
]
